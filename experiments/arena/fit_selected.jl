#=Refit each LLM Arena model with its own validation-selected settings.

Each dataset has a separate full-data train/validation search in
`output/arena/<dataset>/hyperparameter_search/`. This script refits the
selected configuration on train plus validation, then evaluates the held-out
test partition.  It does not transfer Expert-5k hyperparameters to 140k.
=#

include(joinpath(@__DIR__, "..", "..", "src", "datasets", "arena.jl"))
isdefined(@__MODULE__, :RatingBaselines) || include(joinpath(@__DIR__, "..", "..", "src", "evaluation", "baselines.jl"))

using CSV
using DataFrames
using Printf
using Random
using Statistics

const OUTER_MAX_ITERS = 60
const INNER_MAX_ITERS = 60
const RB = RatingBaselines

"""Full-data validation winners, recorded with their source grid."""
const SELECTED_SETTINGS = Dict(
    :expert_5k => (
        vanilla=(beta=20.0, nu=NaN,
            source="refined_full validation NLL"),
        robust=(beta=24.0, nu=32.0,
            source="refined_full validation NLL"),
        consistent=(beta=20.0, nu=128.0,
            source="refined_full validation NLL"),
    ),
    :human_preference_140k => (
        vanilla=(beta=24.0, nu=NaN,
            source="session-held-out validation NLL"),
        robust=(beta=24.0, nu=10.0,
            source="session-held-out validation NLL"),
        consistent=(beta=40.0, nu=2.5,
            source="session-held-out validation NLL"),
    ),
    :search_arena_24k => (
        vanilla=(beta=24.0, nu=NaN,
            source="coarse validation NLL"),
        robust=(beta=24.0, nu=6.0,
            source="coarse validation NLL"),
        consistent=(beta=24.0, nu=16.0,
            source="coarse validation NLL"),
    ),
)

function _record_metrics!(rows, name, split_name, posteriors, matches, draws, beta, epsilon;
    model::Symbol, precisions=nothing, nu=nothing)
    metrics = _model_predictive_metrics(posteriors, matches, draws, beta, epsilon;
        model=model, precisions=precisions, nu=nu, quadrature_points=16)
    push!(rows, (model=name, split=split_name, beta=beta, nu=nu === nothing ? NaN : nu,
        nll=metrics.nll, mean_log_probability=metrics.mean_log_probability,
        mean_correct_probability=metrics.mean_correct_probability, accuracy=metrics.accuracy))
end

_rank_descending(values) = sortperm(sortperm(-collect(values)))

function _ranking_sanity_checks(names, vanilla, robust, consistent, elo, bradley_terry)
    report = DataFrame(
        model=names,
        vanilla_skill=TSG.mean.(vanilla),
        robust_skill=TSG.mean.(robust),
        consistent_skill=TSG.mean.(consistent),
        elo=elo,
        bradley_terry=bradley_terry,
    )
    for column in (:vanilla_skill, :robust_skill, :consistent_skill, :elo, :bradley_terry)
        report[!, Symbol("$(column)_rank")] = _rank_descending(report[!, column])
    end
    correlations = DataFrame(model=String[], baseline=String[], spearman_rank_correlation=Float64[])
    for model in (:vanilla_skill, :robust_skill, :consistent_skill), baseline in (:elo, :bradley_terry)
        push!(correlations, (String(model), String(baseline),
            cor(report[!, Symbol("$(model)_rank")], report[!, Symbol("$(baseline)_rank")])))
    end
    return report, correlations
end

function run_fixed_predictive_llm_arena(; dataset::Symbol, min_games::Int, output_dir::AbstractString,
    settings_override=nothing)
    tables_dir = joinpath(output_dir, "data")
    figures_dir = joinpath(output_dir, "overview")
    diagnostics_dir = joinpath(output_dir, "models", "robust", "diagnostics")
    logs_dir = joinpath(output_dir, "logs")
    foreach(mkpath, (tables_dir, figures_dir, diagnostics_dir, logs_dir))
    settings = settings_override === nothing ? get(SELECTED_SETTINGS, dataset, nothing) : settings_override
    settings === nothing && error("No validation-selected settings for $dataset")
    battles = _load_battles(nothing; dataset=dataset, min_games=min_games)
    # Match the validation search exactly: grouped data are split in their
    # loader order; pre-shuffling changes unique(group_id) and changes the test set.
    battles = _prepare_battle_split(battles; seed=42)
    names, index, matches, draws = _index_battles(battles)
    train, validation, test = _split_battle_indices(battles, 0.70, 0.15; seed=42)
    trainval = vcat(train, validation)
    prior_s = TSG.Gaussian1DFromMeanVariance(25.0, (25.0 / 3)^2)
    rows = DataFrame(model=String[], split=String[], beta=Float64[], nu=Float64[],
        nll=Float64[], mean_log_probability=Float64[], mean_correct_probability=Float64[], accuracy=Float64[])

    # The draw margin is calibrated from all outcomes available during model
    # selection and fitting (train plus validation), never from the test split.
    calibration_draw_rate = mean(draws[trainval])
    epsilon_vanilla = _draw_margin(settings.vanilla.beta, calibration_draw_rate)
    epsilon_robust = _draw_margin(settings.robust.beta, calibration_draw_rate)
    epsilon_consistent = _draw_margin(settings.consistent.beta, calibration_draw_rate)

    @printf("%s: %d battles, %d models, %.2f%% draws\n", String(dataset), length(matches), length(names), 100mean(draws))
    vanilla = _fit_vanilla(prior_s, matches[trainval], draws[trainval], settings.vanilla.beta, epsilon_vanilla, OUTER_MAX_ITERS, true)
    robust, robust_b = _fit_robust(prior_s, matches[trainval], draws[trainval], settings.robust.beta, epsilon_robust,
        settings.robust.nu, INNER_MAX_ITERS, OUTER_MAX_ITERS, true)
    consistent, precisions = TS.consistent_trueskill(
        TSDC.DistributionBag(prior_s, length(names)), TSDC.DistributionBag(_precision_prior(settings.consistent.nu), length(names)),
        matches[trainval], epsilon_consistent; s2=settings.consistent.beta^2, draw_mask=draws[trainval],
        inner_max_iters=INNER_MAX_ITERS, outer_max_iters=OUTER_MAX_ITERS, verbose=true)
    # Fit simple baselines on the same information available to the final
    # TrueSkill refits. They are used only as a ranking sanity check in plots.
    elo = RB.fit_elo(matches[trainval], draws[trainval], length(names))
    bradley_terry = RB.fit_bradley_terry(matches[trainval], draws[trainval], length(names))

    for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test))
        _record_metrics!(rows, "vanilla", split_name, vanilla, matches[split], draws[split], settings.vanilla.beta, epsilon_vanilla; model=:vanilla)
        _record_metrics!(rows, "robust", split_name, robust, matches[split], draws[split], settings.robust.beta, epsilon_robust; model=:robust, nu=settings.robust.nu)
        _record_metrics!(rows, "consistent", split_name, consistent, matches[split], draws[split], settings.consistent.beta, epsilon_consistent;
            model=:consistent, precisions=precisions, nu=settings.consistent.nu)
    end

    report = DataFrame(model=names,
        vanilla_skill=TSG.mean.(vanilla), vanilla_skill_sd=sqrt.(TSG.variance.(vanilla)),
        robust_skill=TSG.mean.(robust), robust_skill_sd=sqrt.(TSG.variance.(robust)),
        consistent_skill=TSG.mean.(consistent), consistent_skill_sd=sqrt.(TSG.variance.(consistent)),
        consistent_precision=TSGamma.mean.(precisions),
        consistent_performance_sd=settings.consistent.beta .* sqrt.(TSGamma.expected_inverse_precision.(precisions)))
    sort!(report, :consistent_skill, rev=true)
    selected_settings = DataFrame(model=["vanilla", "robust", "consistent"],
        beta=[settings.vanilla.beta, settings.robust.beta, settings.consistent.beta],
        nu=[NaN, settings.robust.nu, settings.consistent.nu],
        source=[settings.vanilla.source, settings.robust.source, settings.consistent.source])
    counts = Dict(name => count(==(name), battles.model_a) + count(==(name), battles.model_b) for name in names)
    robust_s2 = 2settings.robust.beta^2 * (settings.robust.nu - 2) / settings.robust.nu
    noisiest = _noisiest_robust_battles(battles[trainval, :], robust_b, robust_s2; n=10)
    split_summary = DataFrame(split=["train", "validation", "test", "train_validation"],
        n_battles=[length(train), length(validation), length(test), length(trainval)],
        n_draws=[count(draws[train]), count(draws[validation]), count(draws[test]), count(draws[trainval])])
    CSV.write(joinpath(tables_dir, "hyperparameters.csv"), selected_settings)
    CSV.write(joinpath(tables_dir, "metrics.csv"), rows)
    CSV.write(joinpath(tables_dir, "model_report.csv"), report)
    CSV.write(joinpath(tables_dir, "final_player_statistics.csv"), report)
    CSV.write(joinpath(tables_dir, "baseline_ratings.csv"), DataFrame(model=names, elo=elo, bradley_terry=bradley_terry))
    ranking_comparison, rank_correlations = _ranking_sanity_checks(names, vanilla, robust, consistent, elo, bradley_terry)
    CSV.write(joinpath(tables_dir, "ranking_comparison.csv"), ranking_comparison)
    CSV.write(joinpath(tables_dir, "rank_correlations.csv"), rank_correlations)
    CSV.write(joinpath(diagnostics_dir, "robust_noisiest_battles.csv"), noisiest)
    CSV.write(joinpath(tables_dir, "split_summary.csv"), split_summary)
    _plot_top_models(names, index, counts, vanilla, consistent, precisions, settings.consistent.beta, 15, joinpath(figures_dir, "top_models.png");
        elo=elo, bradley_terry=bradley_terry)
    _plot_all_models_comparison(names, index, counts, matches, draws, test, vanilla, robust, consistent, precisions,
        settings.consistent.beta, settings.vanilla.beta, epsilon_vanilla, joinpath(figures_dir, "all_models.png");
        elo=elo, bradley_terry=bradley_terry)
    _write_probability_matrix(output_dir, "vanilla", names, _pairwise_win_probability_matrix(names, index, vanilla, settings.vanilla.beta, epsilon_vanilla); ranking=_skill_ranking(names, index, vanilla))
    _write_probability_matrix(output_dir, "robust", names, _pairwise_win_probability_matrix(names, index, robust, settings.robust.beta, epsilon_robust); ranking=_skill_ranking(names, index, robust))
    _write_probability_matrix(output_dir, "consistent", names, _pairwise_win_probability_matrix(names, index, consistent, settings.consistent.beta, epsilon_consistent; precisions=precisions); ranking=_skill_ranking(names, index, consistent))
    open(joinpath(logs_dir, "run.log"), "w") do io
        split_unit = :group_id in propertynames(battles) ? "evaluation session" : "battle"
        println(io, "dataset=$dataset; models=$(length(names)); battles=$(length(matches)); validation-selected settings refit on train+validation; split_unit=$split_unit")
        println(io, "draw calibration rate=$calibration_draw_rate")
        println(io, "vanilla beta=$(settings.vanilla.beta); robust beta=$(settings.robust.beta) nu=$(settings.robust.nu); consistent beta=$(settings.consistent.beta) nu=$(settings.consistent.nu)")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fixed_predictive_llm_arena(dataset=:expert_5k, min_games=20,
        output_dir=joinpath(DEFAULT_OUTPUT_DIR, "expert_5k", "selected_validation_predictive"))
    run_fixed_predictive_llm_arena(dataset=:human_preference_140k, min_games=100,
        output_dir=joinpath(DEFAULT_OUTPUT_DIR, "human_preference_140k", "selected_validation_predictive"))
    run_fixed_predictive_llm_arena(dataset=:search_arena_24k, min_games=20,
        output_dir=joinpath(DEFAULT_OUTPUT_DIR, "search_arena_24k", "selected_validation_predictive"))
end
