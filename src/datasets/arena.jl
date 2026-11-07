#=Static TrueSkill analysis of LLM Arena human-preference battles.

Run from the repository root using the locked environment:
    make arena DATASET=human_preference_140k

The script downloads the public parquet shards only when no compact CSV is present.
The held-out test set evaluates prediction of an unseen human-preference
battle.  Human Preference-140k is split by evaluation session so that related
votes never cross partitions. ``tie`` and ``both_bad`` are both represented as
TrueSkill draws.
=#

isdefined(@__MODULE__, :TrueSkill) || include("../trueskill.jl")
isdefined(@__MODULE__, :PredictiveLikelihoods) || include("../evaluation/predictive.jl")
isdefined(@__MODULE__, :ComparisonPlots) || include("../plots/comparison.jl")

using CSV
using DataFrames
using Dates
using Distributions
using Downloads
using DuckDB
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

const TS = TrueSkill
const TSG = TrueSkill.Gaussian
const TSGamma = TrueSkill.Gamma
const TSDC = TrueSkill.DistributionCollections
const ComparisonPlotHelpers = ComparisonPlots
const PL = PredictiveLikelihoods

const DEFAULT_BATTLES_PATH = joinpath(@__DIR__, "..", "..", "data", "arena", "battles.csv")
const DEFAULT_PARQUET_DIR = get(ENV, "HTS_ARENA_DATA_DIR", joinpath(@__DIR__, "..", "..", "data", "arena", "parquet"))
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output", "arena")
const LLMARENA_DATASETS = Dict(
    :human_preference_140k => (repository="arena-human-preference-140k", n_shards=7, has_timestamp=true,
        group_column="evaluation_session_id", revision="6322995ab34d7c2693e3f47dd13fa5caa0789a74"),
    :expert_5k => (repository="arena-expert-5k", n_shards=1, has_timestamp=false, group_column=nothing, revision="171f77047d8d153c41b3f2c11bc910273f62f183"),
    :search_arena_24k => (repository="search-arena-24k", n_shards=1, has_timestamp=true, group_column=nothing, revision="b6488bd4cb3f4d85dfdfab7d63c5302601c68a95"),
)

function _draw_margin(beta::Float64, draw_rate::Float64)::Float64
    return PL.draw_margin(beta, draw_rate)
end

function _inverse_precision_variance(g::TSGamma.Gamma1D)::Float64
    (isfinite(g.β) && isfinite(g.λ) && g.λ > 0 && g.β > 1) || return NaN
    return g.λ^2 / (g.β^2 * (g.β - 1.0))
end

function _dataset_spec(dataset::Symbol)
    haskey(LLMARENA_DATASETS, dataset) || error("Unknown Arena dataset: $dataset.")
    return LLMARENA_DATASETS[dataset]
end

function _llm_arena_shards(parquet_dir::AbstractString; dataset::Symbol, download::Bool=true)::Vector{String}
    spec = _dataset_spec(dataset)
    dataset_dir = joinpath(parquet_dir, String(dataset))
    mkpath(dataset_dir)
    if dataset === :search_arena_24k
        path = joinpath(dataset_dir, "default", "test", "0000.parquet")
        if !isfile(path)
            download || error("Missing $path. Set download=true.")
            mkpath(dirname(path))
            println("Downloading Search Arena parquet")
            Downloads.download("https://huggingface.co/datasets/lmarena-ai/search-arena-24k/resolve/$(spec.revision)/default/test/0000.parquet", path)
        end
        return [path]
    end
    paths = String[]
    for shard in 0:(spec.n_shards - 1)
        name = @sprintf("train-%05d-of-%05d.parquet", shard, spec.n_shards)
        path = joinpath(dataset_dir, name)
        if !isfile(path)
            download || error("Missing $path. Set download=true or provide battles.csv.")
            println("Downloading LLM Arena $(dataset) shard $(shard + 1)/$(spec.n_shards)")
            Downloads.download("https://huggingface.co/datasets/lmarena-ai/$(spec.repository)/resolve/$(spec.revision)/data/$name", path)
        end
        push!(paths, path)
    end
    return paths
end

function _read_parquet_battles(DuckDB, DBInterface, parquet_dir::AbstractString; dataset::Symbol, download::Bool=true)::DataFrame
    spec = _dataset_spec(dataset)
    columns = spec.has_timestamp ? "timestamp, model_a, model_b, winner" : "model_a, model_b, winner"
    # A session can contain several votes over a shared conversation context.
    # Preserve it for group-wise splitting, rather than leaking related votes
    # across train/validation/test.  Other Arena releases do not expose a
    # compatible session identifier and retain their existing battle split.
    spec.group_column !== nothing && (columns *= ", $(spec.group_column) AS group_id")
    frames = DataFrame[]
    database = DuckDB.DB()
    try
        for path in _llm_arena_shards(parquet_dir; dataset=dataset, download=download)
            # DuckDB's parquet projection leaves the large nested conversation fields unread.
            sql_path = replace(path, '\\' => '/', "'" => "''")
            push!(frames, DataFrame(DBInterface.execute(database, "SELECT $columns FROM read_parquet('$sql_path')")))
        end
    finally
        DuckDB.close(database)
    end
    return vcat(frames...)
end

function _load_parquet_battles(parquet_dir::AbstractString; dataset::Symbol, download::Bool=true)::DataFrame
    return _read_parquet_battles(DuckDB, DuckDB.DBInterface, parquet_dir; dataset=dataset, download=download)
end

function _load_battles(path::Union{Nothing, AbstractString}; parquet_dir::AbstractString=DEFAULT_PARQUET_DIR,
    dataset::Symbol=:human_preference_140k, download::Bool=true, min_games::Int=100)::DataFrame
    path !== nothing && !isfile(path) && error("Missing preference CSV: $path")
    battles = path !== nothing ? DataFrame(CSV.File(path)) :
        _load_parquet_battles(parquet_dir; dataset=dataset, download=download)
    required = [:model_a, :model_b, :winner]
    isempty(setdiff(required, propertynames(battles))) || error("Preference CSV requires model_a, model_b, winner")
    if :evaluation_session_id in propertynames(battles) && :group_id ∉ propertynames(battles)
        rename!(battles, :evaluation_session_id => :group_id)
    end
    # Search Arena also contains unvoted conversations.  Retain only labelled
    # non-self battles, preserving both `tie` and `both_bad` as draws.
    filter!(row -> !ismissing(row.winner) && row.winner in ("model_a", "model_b", "tie", "both_bad") &&
        !ismissing(row.model_a) && !ismissing(row.model_b) && row.model_a != row.model_b, battles)
    counts = Dict{String, Int}()
    for row in eachrow(battles)
        for model in (String(row.model_a), String(row.model_b))
            counts[model] = get(counts, model, 0) + 1
        end
    end
    eligible = Set(model for (model, n) in counts if n >= min_games)
    filter!(row -> row.model_a in eligible && row.model_b in eligible, battles)
    :timestamp in propertynames(battles) && sort!(battles, :timestamp)
    return battles
end

function _index_battles(battles::DataFrame)
    names = sort!(unique(vcat(battles.model_a, battles.model_b)))
    index = Dict(name => i for (i, name) in enumerate(names))
    matches = Tuple{Int64, Int64}[]
    draws = BitVector()
    for row in eachrow(battles)
        a, b = index[row.model_a], index[row.model_b]
        if row.winner == "model_a"
            push!(matches, (a, b)); push!(draws, false)
        elseif row.winner == "model_b"
            push!(matches, (b, a)); push!(draws, false)
        else # Both `tie` and `both_bad` mean neither player wins.
            push!(matches, (a, b)); push!(draws, true)
        end
    end
    return names, index, matches, draws
end

_arena_string(x) = ismissing(x) ? "" : string(x)

"""Return the battles whose robust latent precision implies the most noise."""
function _noisiest_robust_battles(battles::DataFrame, robust_b, robust_s2::Float64; n::Int=10)
    rows = DataFrame(
        match_index=Int[], timestamp=String[], model_a=String[], model_b=String[], winner=String[], is_draw=Bool[],
        robust_mean_precision=Float64[], robust_expected_noise_variance=Float64[],
    )
    for i in 1:min(nrow(battles), length(robust_b))
        row = battles[i, :]
        push!(rows, (
            match_index=i,
            timestamp=:timestamp in propertynames(battles) ? _arena_string(row.timestamp) : "",
            model_a=_arena_string(row.model_a), model_b=_arena_string(row.model_b), winner=_arena_string(row.winner),
            is_draw=row.winner == "tie" || row.winner == "both_bad",
            robust_mean_precision=TSGamma.mean(robust_b[i]),
            robust_expected_noise_variance=robust_s2 * TSGamma.expected_inverse_precision(robust_b[i]),
        ))
    end
    sort!(rows, :robust_mean_precision)
    return first(rows, min(n, nrow(rows)))
end

function _split_indices(n::Int, train_fraction::Float64, validation_fraction::Float64)
    train_end = floor(Int, n * train_fraction)
    validation_end = floor(Int, n * (train_fraction + validation_fraction))
    return 1:train_end, (train_end + 1):validation_end, (validation_end + 1):n
end

"""Use the identical pre-split order in selection and final evaluation."""
function _prepare_battle_split(battles::DataFrame; seed::Int=42)
    return :group_id in propertynames(battles) ? battles :
        battles[randperm(MersenneTwister(seed), nrow(battles)), :]
end

"""Split preference battles without splitting an evaluation session.

When ``group_id`` is present (currently Human Preference-140k), all votes in a
session are assigned to one partition.  Group order is randomized by ``seed``;
the targets are numbers of battles, so split sizes are as close as possible to
the requested fractions while preserving session integrity.  Datasets without
a grouping variable retain the historical random battle-level split.
"""
function _split_battle_indices(battles::DataFrame, train_fraction::Float64, validation_fraction::Float64;
    seed::Int=42)
    0 < train_fraction < 1 || error("train_fraction must be in (0, 1)")
    0 < validation_fraction < 1 || error("validation_fraction must be in (0, 1)")
    train_fraction + validation_fraction < 1 || error("train and validation fractions must sum to less than one")
    :group_id ∉ propertynames(battles) && return _split_indices(nrow(battles), train_fraction, validation_fraction)

    # Missing identifiers are conservatively treated as one-observation groups.
    group_ids = [ismissing(value) || isempty(strip(string(value))) ? "__row_$(row)" : string(value)
        for (row, value) in enumerate(battles.group_id)]
    group_order = unique(group_ids)
    group_order = group_order[randperm(MersenneTwister(seed), length(group_order))]
    target_train = floor(Int, nrow(battles) * train_fraction)
    target_validation = floor(Int, nrow(battles) * (train_fraction + validation_fraction))
    group_sizes = Dict{String,Int}()
    for id in group_ids
        group_sizes[id] = get(group_sizes, id, 0) + 1
    end
    train_groups, validation_groups = Set{String}(), Set{String}()
    assigned = 0
    for id in group_order
        # Keep every group intact.  A boundary group may make a partition
        # slightly larger than its target, but is never silently discarded.
        if assigned < target_train
            push!(train_groups, id)
        elseif assigned < target_validation
            push!(validation_groups, id)
        end
        assigned += group_sizes[id]
    end
    train = findall(id -> id in train_groups, group_ids)
    validation = findall(id -> id in validation_groups, group_ids)
    test = findall(id -> !(id in train_groups || id in validation_groups), group_ids)
    isempty(train) && error("Group split produced an empty training partition")
    isempty(validation) && error("Group split produced an empty validation partition")
    isempty(test) && error("Group split produced an empty test partition")
    return train, validation, test
end

function _outcome_nll(posteriors, matches, draws, beta::Float64, epsilon::Float64;
    model::Symbol=:vanilla, precisions=nothing, nu=nothing)
    return _outcome_metrics(posteriors, matches, draws, beta, epsilon;
        model=model, precisions=precisions, nu=nu).nll
end

"""Backward-compatible adapter for historical Arena sweeps.

New code should use `_model_predictive_metrics`; this adapter now delegates to
the same dispatched evaluator instead of maintaining a separate Normal or
moment-matched likelihood implementation.
"""
function _outcome_metrics(posteriors, matches, draws, beta::Float64, epsilon::Float64;
    model::Symbol=:vanilla, precisions=nothing, nu=nothing)
    return _model_predictive_metrics(posteriors, matches, draws, beta, epsilon;
        model=model, precisions=precisions, nu=nu)
end

"""Posterior-predictive outcome likelihood evaluated by deterministic Gamma quadrature.

For a held-out battle, robust integrates a new match precision under its Gamma
prior; consistent integrates both players' inferred Gamma precision posteriors.
The Gaussian skill posterior is integrated analytically.  Thus the remaining
approximation is the fitted mean-field posterior, not a moment-matched
observation variance.
"""
function _model_predictive_metrics(posteriors, matches, draws, beta, epsilon;
    model::Symbol, precisions=nothing, nu::Union{Nothing, Float64}=nothing,
    robust_s2::Union{Nothing, Float64}=nothing, quadrature_points::Int=16)
    s2 = model === :robust ? (robust_s2 === nothing ?
        2beta^2 * ((nu::Float64) - 2) / (nu::Float64) : robust_s2) : 2beta^2
    predictor = model === :vanilla ? PL.VanillaPredictive(beta) :
        model === :robust ? PL.RobustPredictive(_precision_prior(nu::Float64), s2, quadrature_points) :
        model === :consistent ? PL.ConsistentPredictive(beta^2, quadrature_points) :
        error("Unknown predictive model: $model")
    return model === :consistent ? PL.model_metrics(predictor, posteriors, precisions, matches, draws, epsilon) :
        PL.model_metrics(predictor, posteriors, matches, draws, epsilon)
end

function _fit_vanilla(prior_s, matches, draws, beta, epsilon, outer_max_iters, verbose)
    n_players = isempty(matches) ? 0 : maximum(maximum(match) for match in matches)
    return TS.vanilla_trueskill(TSDC.DistributionBag(prior_s, n_players), beta^2, matches, epsilon;
                                draw_mask=draws, outer_max_iters=outer_max_iters, verbose=verbose)
end

# Gamma1D stores the Gamma shape minus one as its first parameter (see
# Gamma.shape).  This helper is therefore the paper's Gamma(nu/2, nu/2),
# expressed in the factor implementation's natural parameterization.
function _precision_prior(nu::Real)
    nu > 2 || error("The variance-matched precision prior requires nu > 2.")
    return TSGamma.Gamma1D(nu / 2 - 1, nu / 2)
end

function _fit_robust(prior_s, matches, draws, beta, epsilon, nu, inner_max_iters, outer_max_iters, verbose;
    schedule::Symbol=:interleaved, local_max_iters::Int=1, local_tol::Float64=1e-4)
    nu > 2 || error("Robust TrueSkill requires nu > 2 for variance matching.")
    n_players = isempty(matches) ? 0 : maximum(maximum(match) for match in matches)
    prior_b = _precision_prior(nu)
    # Match the marginal comparison-noise variance of vanilla TrueSkill.
    s2 = 2beta^2 * (nu - 2) / nu
    return TS.robust_trueskill(TSDC.DistributionBag(prior_s, n_players),
        TSDC.DistributionBag(prior_b, length(matches)), matches, epsilon;
        s2=s2, draw_mask=draws, inner_max_iters=inner_max_iters,
        outer_max_iters=outer_max_iters, schedule=schedule,
        local_max_iters=local_max_iters, local_tol=local_tol, verbose=verbose)
end

function _pairwise_win_probability_matrix(names, index, posteriors, beta, epsilon; precisions=nothing)
    n = length(names)
    probabilities = Matrix{Float64}(undef, n, n)
    for row in 1:n, column in 1:n
        i, j = index[names[row]], index[names[column]]
        if i == j
            probabilities[row, column] = NaN
            continue
        end
        mean_difference = TSG.mean(posteriors[i]) - TSG.mean(posteriors[j])
        skill_variance = TSG.variance(posteriors[i]) + TSG.variance(posteriors[j])
        performance_variance = precisions === nothing ? 2beta^2 : beta^2 * (
            TSGamma.expected_inverse_precision(precisions[i]) + TSGamma.expected_inverse_precision(precisions[j]))
        probabilities[row, column] = 1.0 - cdf(Normal(mean_difference, sqrt(skill_variance + performance_variance)), epsilon)
    end
    return probabilities
end

function _write_probability_matrix(output_dir, prefix, names, probabilities; ranking=nothing)
    if ranking !== nothing
        permutation = sortperm(ranking; rev=true)
        names = names[permutation]
        probabilities = probabilities[permutation, permutation]
    end
    table = DataFrame(model=names)
    for (column, name) in enumerate(names)
        table[!, Symbol(name)] = probabilities[:, column]
    end
    matrix_dir = joinpath(output_dir, "models", prefix)
    mkpath(matrix_dir)
    CSV.write(joinpath(matrix_dir, "$(prefix)_win_probability_matrix.csv"), table)
    n = length(names)
    pixels_per_model = 54
    figure = heatmap(probabilities; clims=(0, 1), c=:viridis, colorbar_title="P(row wins)",
        xticks=(1:n, names), yticks=(1:n, names), xrotation=90,
        tickfont=font(8), size=(max(2_400, pixels_per_model * n), max(2_200, pixels_per_model * n)),
        margin=18Plots.mm, left_margin=36Plots.mm, bottom_margin=42Plots.mm,
        title="$(uppercasefirst(prefix)) TrueSkill pairwise win probabilities", xlabel="opponent (column)", ylabel="model (row)")
    savefig(figure, joinpath(matrix_dir, "$(prefix)_win_probability_matrix.png"))
    savefig(figure, joinpath(matrix_dir, "$(prefix)_win_probability_matrix.pdf"))
end

_skill_ranking(names, index, posteriors) = [TSG.mean(posteriors[index[name]]) for name in names]

function _write_beta_nu_sweep_plots(rows, output_dir)
    betas = sort(unique(rows.beta))
    nus = sort(unique(filter(isfinite, rows.nu)))
    vanilla_validation = sort(filter(row -> row.model == "vanilla" && row.split == "validation", rows), :beta)
    vanilla_plot = plot(vanilla_validation.beta, vanilla_validation.nll; marker=:circle,
        xlabel="beta", ylabel="validation NLL", title="vanilla TrueSkill",
        legend=false, size=(900, 620), left_margin=14Plots.mm, bottom_margin=14Plots.mm)
    savefig(vanilla_plot, joinpath(output_dir, "vanilla_validation_nll_by_beta.png"))
    savefig(vanilla_plot, joinpath(output_dir, "vanilla_validation_nll_by_beta.pdf"))
    for model in ("robust", "consistent")
        values = [only(filter(row -> row.model == model && row.split == "validation" && row.beta == beta && row.nu == nu, rows).nll)
                  for beta in betas, nu in nus]
        heatmap_plot = heatmap(collect(eachindex(nus)), collect(eachindex(betas)), values;
            xlabel="nu", ylabel="beta", title="$(model) validation NLL",
            xticks=(collect(eachindex(nus)), string.(nus)), yticks=(collect(eachindex(betas)), string.(betas)),
            xrotation=45, tickfont=font(9), colorbar_title="NLL", yflip=true,
            size=(1100, 700), left_margin=18Plots.mm, bottom_margin=24Plots.mm)
        savefig(heatmap_plot, joinpath(output_dir, "$(model)_validation_nll_beta_nu_heatmap.png"))
        savefig(heatmap_plot, joinpath(output_dir, "$(model)_validation_nll_beta_nu_heatmap.pdf"))
    end
end

"""Plot selected TrueSkill estimates with optional standardized baseline scores.

Elo, Bradley--Terry, and TrueSkill scores live on different arbitrary scales.
For a visual ranking check, each baseline is therefore centered and scaled over
the displayed models; the ordering is unchanged and the axis is explicitly not
a common skill unit.
"""
function _plot_top_models(names, index, counts, vanilla_s, consistent_s, consistent_b, beta, top_n, path;
    elo=nothing, bradley_terry=nothing)
    top = sort(names; by=name -> (-TSG.mean(consistent_s[index[name]]), name))[1:min(top_n, length(names))]
    ordered = sort(top; by=name -> TSG.mean(consistent_s[index[name]]))
    ids, y = [index[name] for name in ordered], collect(1:length(ordered))
    labels = [@sprintf("%-48s %6d", name, get(counts, name, 0)) for name in ordered]
    labels = [labels; @sprintf("%-48s %6s", "model", "# battles")]
    ticks = ([y; maximum(y) + 1], labels)
    is_full = length(top) == length(names)
    row_height = is_full ? 24 : 32
    consistent_skill = scatter([TSG.mean(consistent_s[i]) for i in ids], y;
        xerr=[3sqrt(TSG.variance(consistent_s[i])) for i in ids], yticks=ticks,
        ylims=(0.5, length(y) + 1.5), xlabel="estimated skill ± 3σ", title="consistent TrueSkill skill", label="", markersize=5)
    vanilla_skill = scatter([TSG.mean(vanilla_s[i]) for i in ids], y;
        xerr=[3sqrt(TSG.variance(vanilla_s[i])) for i in ids], yticks=(y, fill("", length(y))),
        ylims=(0.5, length(y) + 1.5), xlabel="estimated skill ± 3σ", title="learned vanilla TrueSkill skill", label="", markersize=5)
    noise = [beta * sqrt(TSGamma.expected_inverse_precision(consistent_b[i])) for i in ids]
    noise_error = [3beta * sqrt(_inverse_precision_variance(consistent_b[i])) for i in ids]
    consistency = scatter(noise, y; xerr=noise_error, yticks=(y, fill("", length(y))),
        ylims=(0.5, length(y) + 1.5), xlabel="estimated performance s.d.", title="player inconsistency (lower is steadier)", label="", markersize=5)
    panels = Any[consistent_skill, vanilla_skill, consistency]
    function baseline_panel(scores, title)
        values = Float64[scores[i] for i in ids]
        scale = std(values)
        standardized = scale > 0 ? (values .- mean(values)) ./ scale : zeros(length(values))
        return scatter(standardized, y; yticks=(y, fill("", length(y))),
            ylims=(0.5, length(y) + 1.5), xlabel="standardized score", title=title,
            label="", markersize=5)
    end
    elo === nothing || push!(panels, baseline_panel(elo, "Elo ranking check"))
    bradley_terry === nothing || push!(panels, baseline_panel(bradley_terry, "Bradley--Terry ranking check"))
    plt = plot(panels...; layout=(1, length(panels)),
        size=(760 * length(panels), max(520, row_height * length(top) + 110)),
        margin=10Plots.mm, left_margin=52Plots.mm, thickness_scaling=is_full ? 0.65 : 1.0)
    savefig(plt, path)
    savefig(plt, replace(path, r"\.[^.]+$" => ".pdf"))
    return plt
end

"""Observed outcome rates plus held-out vanilla predictive surprisal by model."""
function _arena_row_statistics(names, index, matches, draws, test_indices, vanilla_s, vanilla_beta, vanilla_epsilon)
    statistics = Dict(name => (games=0, wins=0, draws=0, losses=0,
        test_wins=0, test_draws=0, test_losses=0,
        test_win_surprisal=0.0, test_draw_surprisal=0.0, test_loss_surprisal=0.0) for name in names)
    test_set = Set(test_indices)
    for (match_index, ((winner, loser), draw)) in enumerate(zip(matches, draws))
        winner_name, loser_name = names[winner], names[loser]
        if draw
            for player in (winner, loser)
                name = names[player]
                row = statistics[name]
                statistics[name] = merge(row, (games=row.games + 1, draws=row.draws + 1))
            end
        else
            winner_row, loser_row = statistics[winner_name], statistics[loser_name]
            statistics[winner_name] = merge(winner_row, (games=winner_row.games + 1,
                wins=winner_row.wins + 1))
            statistics[loser_name] = merge(loser_row, (games=loser_row.games + 1,
                losses=loser_row.losses + 1))
        end
        match_index in test_set || continue
        probabilities = PL.predict_outcomes(PL.VanillaPredictive(vanilla_beta), vanilla_s,
            winner, loser, vanilla_epsilon)
        if draw
            surprise = -log(clamp(probabilities.draw, 1e-12, 1.0))
            for name in (winner_name, loser_name)
                row = statistics[name]
                statistics[name] = merge(row, (test_draws=row.test_draws + 1,
                    test_draw_surprisal=row.test_draw_surprisal + surprise))
            end
        else
            winner_row, loser_row = statistics[winner_name], statistics[loser_name]
            statistics[winner_name] = merge(winner_row, (test_wins=winner_row.test_wins + 1,
                test_win_surprisal=winner_row.test_win_surprisal - log(clamp(probabilities.first, 1e-12, 1.0))))
            statistics[loser_name] = merge(loser_row, (test_losses=loser_row.test_losses + 1,
                test_loss_surprisal=loser_row.test_loss_surprisal - log(clamp(probabilities.second, 1e-12, 1.0))))
        end
    end
    return statistics
end

_rate_percent(numerator, denominator) = denominator == 0 ? "--" : @sprintf("%2.0f", 100numerator / denominator)
_rate_triplet(first, second, third) = @sprintf("%3s / %3s / %3s", first, second, third)
_display_model_name(name; max_chars::Int=34) = length(name) <= max_chars ? name : first(name, max_chars - 1) * "…"

"""Integer outcome percentages summing to 100 by largest-remainder rounding."""
function _outcome_percentages(wins, draws, losses)
    counts = [wins, draws, losses]
    total = sum(counts)
    total == 0 && return ("--", "--", "--")
    raw = 100 .* counts ./ total
    rounded = floor.(Int, raw)
    for index in sortperm(1:3; by=index -> (-(raw[index] - rounded[index]), index))[1:(100 - sum(rounded))]
        rounded[index] += 1
    end
    return string.(rounded)
end

"""All-model Arena comparison using the shared Chess-style panel layout."""
function _plot_all_models_comparison(names, index, counts, matches, draws, test_indices, vanilla_s, robust_s, consistent_s, consistent_b, beta, vanilla_beta, vanilla_epsilon, path;
    elo, bradley_terry)
    ordered = sort(names; by=name -> TSG.mean(consistent_s[index[name]]))
    ids = [index[name] for name in ordered]
    y = collect(1:length(ordered))
    statistics = _arena_row_statistics(names, index, matches, draws, test_indices, vanilla_s, vanilla_beta, vanilla_epsilon)
    labels = String[]
    for name in ordered
        row = statistics[name]
        outcome = _rate_triplet(_outcome_percentages(row.wins, row.draws, row.losses)...)
        surprise = _rate_triplet(
            row.test_wins == 0 ? "--" : @sprintf("%.2f", row.test_win_surprisal / row.test_wins),
            row.test_draws == 0 ? "--" : @sprintf("%.2f", row.test_draw_surprisal / row.test_draws),
            row.test_losses == 0 ? "--" : @sprintf("%.2f", row.test_loss_surprisal / row.test_losses))
        push!(labels, @sprintf("%-34s %5d  %15s  %15s", _display_model_name(name), get(counts, name, 0), outcome, surprise))
    end
    header = @sprintf("%-34s %5s  %15s  %15s", "model", "games", "W / D / L (%)", "test -log p (W/D/L)")
    skill_series = [
        (label="vanilla", means=[TSG.mean(vanilla_s[i]) for i in ids],
            errors=[3sqrt(TSG.variance(vanilla_s[i])) for i in ids], color=:steelblue),
        (label="robust", means=[TSG.mean(robust_s[i]) for i in ids],
            errors=[3sqrt(TSG.variance(robust_s[i])) for i in ids], color=:darkorange),
        (label="consistent", means=[TSG.mean(consistent_s[i]) for i in ids],
            errors=[3sqrt(TSG.variance(consistent_s[i])) for i in ids], color=:forestgreen),
    ]
    performance = (label="consistent", means=[beta * sqrt(TSGamma.expected_inverse_precision(consistent_b[i])) for i in ids],
        errors=[3beta * sqrt(_inverse_precision_variance(consistent_b[i])) for i in ids], color=:forestgreen)
    baselines = [
        (values=[elo[i] for i in ids], title="Elo ranking check"),
        (values=[bradley_terry[i] for i in ids], title="Bradley--Terry ranking check"),
    ]
    return ComparisonPlotHelpers.model_comparison_plot(y, (vcat(y, length(y) + 1), vcat(labels, header)), skill_series, performance, path;
        skill_title="skills (ordered by consistent TrueSkill)", performance_title="consistent player performance",
        skill_xlabel="skill mean +/- 3 s.d.", performance_xlabel="performance s.d.",
        ylabel="model", baseline_series=baselines, row_height=24)
end

function run_llm_arena_experiment(; battles_path::Union{Nothing, AbstractString}=nothing,
    parquet_dir::AbstractString=DEFAULT_PARQUET_DIR, dataset::Symbol=:human_preference_140k,
    download::Bool=true, split_seed::Int=42,
    output_dir::Union{Nothing, AbstractString}=nothing, min_games::Int=100, max_matches::Int=0,
    mu::Float64=25.0, sigma::Float64=25.0 / 3, beta_grid::Vector{Float64}=[4.0, 6.0, 8.0, 10.0, 12.0],
    nu_grid::Vector{Float64}=[4.0, 10.0, 20.0, 50.0, 100.0], train_fraction::Float64=0.70, validation_fraction::Float64=0.15,
    top_n::Int=15, outer_max_iters::Int=30, inner_max_iters::Int=30, verbose::Bool=true,
    fit_mode::Symbol=:sweep, fixed_beta::Union{Nothing, Float64}=nothing,
    fixed_robust_nu::Union{Nothing, Float64}=nothing, fixed_consistent_nu::Union{Nothing, Float64}=nothing)

    output_dir = output_dir === nothing ? joinpath(DEFAULT_OUTPUT_DIR, String(dataset)) : output_dir
    fit_mode in (:sweep, :fixed) || error("fit_mode must be :sweep or :fixed")
    if fit_mode === :fixed
        isnothing(fixed_beta) && error("fixed_beta is required when fit_mode=:fixed")
        isnothing(fixed_robust_nu) && error("fixed_robust_nu is required when fit_mode=:fixed")
        isnothing(fixed_consistent_nu) && error("fixed_consistent_nu is required when fit_mode=:fixed")
        fixed_beta > 0 || error("fixed_beta must be positive")
        fixed_robust_nu > 2 || error("fixed_robust_nu must exceed 2")
        fixed_consistent_nu > 2 || error("fixed_consistent_nu must exceed 2")
    end
    effective_beta_grid = fit_mode === :fixed ? [fixed_beta::Float64] : beta_grid
    effective_robust_nu_grid = fit_mode === :fixed ? [fixed_robust_nu::Float64] : nu_grid
    effective_consistent_nu_grid = fit_mode === :fixed ? [fixed_consistent_nu::Float64] : nu_grid
    battles = _load_battles(battles_path; parquet_dir=parquet_dir, dataset=dataset, download=download, min_games=min_games)
    # Arena battles are i.i.d. preference observations, not a temporal
    # forecasting problem. Shuffle every release before taking the contiguous
    # train/validation/test ranges. This also prevents newly added models from
    # being absent from the fitted posterior due to a chronological cutoff.
    battles = battles[randperm(MersenneTwister(split_seed), nrow(battles)), :]
    max_matches > 0 && (battles = first(battles, min(max_matches, nrow(battles))))
    names, index, matches, draws = _index_battles(battles)
    counts = Dict(name => count(==(name), battles.model_a) + count(==(name), battles.model_b) for name in names)
    train, validation, test = _split_battle_indices(battles, train_fraction, validation_fraction; seed=split_seed)
    trainval = vcat(train, validation)
    prior_s = TSG.Gaussian1DFromMeanVariance(mu, sigma^2)
    verbose && @printf("LLM Arena: %d battles, %d models; draws (tie + both_bad): %.1f%%\n", length(matches), length(names), 100mean(draws))

    # Learn vanilla's one global performance scale on the random validation set.
    validation_scores = DataFrame(beta=Float64[], validation_nll=Float64[])
    for beta in effective_beta_grid
        epsilon = _draw_margin(beta, mean(draws[train]))
        fit = _fit_vanilla(prior_s, matches[train], draws[train], beta, epsilon, outer_max_iters, false)
        push!(validation_scores, (beta=beta, validation_nll=_outcome_nll(fit, matches[validation], draws[validation], beta, epsilon)))
    end
    sort!(validation_scores, :validation_nll)
    learned_beta = validation_scores.beta[1]
    epsilon = _draw_margin(learned_beta, mean(draws[train]))
    vanilla_s = _fit_vanilla(prior_s, matches[trainval], draws[trainval], learned_beta, epsilon, outer_max_iters, verbose)

    consistency_validation = DataFrame(nu=Float64[], validation_nll=Float64[])
    for nu in effective_consistent_nu_grid
        prior_b = _precision_prior(nu)
        candidate_s, candidate_b = TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)),
            TSDC.DistributionBag(prior_b, length(names)), matches[train], epsilon;
            s2=learned_beta^2, draw_mask=draws[train], inner_max_iters=inner_max_iters,
            outer_max_iters=outer_max_iters, verbose=false)
        push!(consistency_validation, (nu=nu, validation_nll=_model_predictive_metrics(
            candidate_s, matches[validation], draws[validation], learned_beta, epsilon;
            model=:consistent, precisions=candidate_b).nll))
    end
    sort!(consistency_validation, :validation_nll)
    learned_consistency_nu = consistency_validation.nu[1]
    consistent_prior_b = _precision_prior(learned_consistency_nu)
    consistent_s, consistent_b = TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)),
        TSDC.DistributionBag(consistent_prior_b, length(names)), matches[trainval], epsilon;
        s2=learned_beta^2, draw_mask=draws[trainval], inner_max_iters=inner_max_iters,
        outer_max_iters=outer_max_iters, verbose=verbose)
    robust_validation = DataFrame(nu=Float64[], validation_nll=Float64[])
    for nu in effective_robust_nu_grid
        candidate_s, _ = _fit_robust(prior_s, matches[train], draws[train], learned_beta, epsilon, nu, inner_max_iters, outer_max_iters, false)
        push!(robust_validation, (nu=nu, validation_nll=_model_predictive_metrics(
            candidate_s, matches[validation], draws[validation], learned_beta, epsilon;
            model=:robust, nu=nu).nll))
    end
    sort!(robust_validation, :validation_nll)
    learned_robust_nu = robust_validation.nu[1]
    robust_s, robust_b = _fit_robust(prior_s, matches[trainval], draws[trainval], learned_beta, epsilon, learned_robust_nu, inner_max_iters, outer_max_iters, verbose)
    model_names = ["learned vanilla", "robust", "consistent"]
    robust_s2 = 2learned_beta^2 * (learned_robust_nu - 2) / learned_robust_nu
    metrics = [_model_predictive_metrics(vanilla_s, matches[test], draws[test], learned_beta, epsilon; model=:vanilla),
        _model_predictive_metrics(robust_s, matches[test], draws[test], learned_beta, epsilon;
            model=:robust, nu=learned_robust_nu, robust_s2=robust_s2),
        _model_predictive_metrics(consistent_s, matches[test], draws[test], learned_beta, epsilon;
            model=:consistent, precisions=consistent_b)]
    test_scores = DataFrame(model=model_names, test_nll=[m.nll for m in metrics],
        mean_log_probability=[m.mean_log_probability for m in metrics],
        mean_correct_probability=[m.mean_correct_probability for m in metrics], accuracy=[m.accuracy for m in metrics])
    train_metrics = [_model_predictive_metrics(vanilla_s, matches[trainval], draws[trainval], learned_beta, epsilon; model=:vanilla),
        _model_predictive_metrics(robust_s, matches[trainval], draws[trainval], learned_beta, epsilon;
            model=:robust, nu=learned_robust_nu, robust_s2=robust_s2),
        _model_predictive_metrics(consistent_s, matches[trainval], draws[trainval], learned_beta, epsilon;
            model=:consistent, precisions=consistent_b)]
    train_scores = DataFrame(model=model_names, n_matches=length(trainval),
        train_nll=[m.nll for m in train_metrics],
        mean_log_probability=[m.mean_log_probability for m in train_metrics],
        total_log_probability=[m.mean_log_probability * length(trainval) for m in train_metrics],
        mean_correct_probability=[m.mean_correct_probability for m in train_metrics], accuracy=[m.accuracy for m in train_metrics])
    test_predictive_scores = DataFrame(model=model_names, n_matches=length(test),
        nll=[m.nll for m in metrics], mean_log_probability=[m.mean_log_probability for m in metrics],
        mean_correct_probability=[m.mean_correct_probability for m in metrics], accuracy=[m.accuracy for m in metrics])
    train_predictive_scores = DataFrame(model=model_names, n_matches=length(trainval),
        nll=[m.nll for m in train_metrics], mean_log_probability=[m.mean_log_probability for m in train_metrics],
        mean_correct_probability=[m.mean_correct_probability for m in train_metrics], accuracy=[m.accuracy for m in train_metrics])
    report = DataFrame(model=names, n_battles=[counts[name] for name in names],
        vanilla_skill=[TSG.mean(vanilla_s[index[name]]) for name in names],
        vanilla_skill_sd=[sqrt(TSG.variance(vanilla_s[index[name]])) for name in names],
        robust_skill=[TSG.mean(robust_s[index[name]]) for name in names],
        robust_skill_sd=[sqrt(TSG.variance(robust_s[index[name]])) for name in names],
        consistent_skill=[TSG.mean(consistent_s[index[name]]) for name in names],
        consistent_skill_sd=[sqrt(TSG.variance(consistent_s[index[name]])) for name in names],
        performance_sd=[learned_beta * sqrt(TSGamma.expected_inverse_precision(consistent_b[index[name]])) for name in names],
        consistency_precision=[TSGamma.mean(consistent_b[index[name]]) for name in names])
    sort!(report, :consistent_skill, rev=true)
    report[!, :consistent_rank] = collect(1:nrow(report))
    report[!, :vanilla_rank] = sortperm(sortperm(-report.vanilla_skill))
    report[!, :robust_rank] = sortperm(sortperm(-report.robust_skill))
    noisiest_battles = _noisiest_robust_battles(battles, robust_b, robust_s2; n=10)
    split_summary = DataFrame(
        split=["train", "validation", "test", "train_validation"],
        n_battles=[length(train), length(validation), length(test), length(trainval)],
        n_draws=[count(draws[train]), count(draws[validation]), count(draws[test]), count(draws[trainval])],
        draw_rate=[mean(draws[train]), mean(draws[validation]), mean(draws[test]), mean(draws[trainval])],
    )
    metadata = DataFrame([(
        dataset=String(dataset), battles_path=isnothing(battles_path) ? "" : String(battles_path), parquet_dir=parquet_dir,
        output_dir=output_dir, fit_mode=String(fit_mode), split_seed=split_seed, min_games=min_games, max_matches=max_matches,
        n_models=length(names), n_battles=length(matches), n_draws=count(draws), mu=mu, sigma=sigma,
        learned_beta=learned_beta, learned_robust_nu=learned_robust_nu, learned_consistent_nu=learned_consistency_nu,
        draw_margin=epsilon, robust_s2=robust_s2, consistent_s2=learned_beta^2,
        train_fraction=train_fraction, validation_fraction=validation_fraction,
        outer_max_iters=outer_max_iters, inner_max_iters=inner_max_iters,
    )])
    data_dir = joinpath(output_dir, "data")
    logs_dir = joinpath(output_dir, "logs")
    foreach(mkpath, (data_dir, logs_dir))
    foreach(model -> mkpath(joinpath(output_dir, "models", model)), ("vanilla", "robust", "consistent"))
    CSV.write(joinpath(output_dir, "vanilla_beta_validation.csv"), validation_scores)
    CSV.write(joinpath(output_dir, "consistent_nu_validation.csv"), consistency_validation)
    CSV.write(joinpath(output_dir, "robust_nu_validation.csv"), robust_validation)
    CSV.write(joinpath(output_dir, "test_outcome_nll.csv"), test_scores)
    CSV.write(joinpath(output_dir, "train_outcome_metrics.csv"), train_scores)
    CSV.write(joinpath(output_dir, "test_predictive_metrics.csv"), test_predictive_scores)
    CSV.write(joinpath(output_dir, "train_predictive_metrics.csv"), train_predictive_scores)
    CSV.write(joinpath(output_dir, "split_summary.csv"), split_summary)
    CSV.write(joinpath(output_dir, "robust_noisiest_battles.csv"), noisiest_battles)
    CSV.write(joinpath(output_dir, "metadata.csv"), metadata)
    CSV.write(joinpath(output_dir, "model_report.csv"), report)
    CSV.write(joinpath(output_dir, "final_player_statistics.csv"), report)
    _plot_top_models(names, index, counts, vanilla_s, consistent_s, consistent_b, learned_beta, top_n, joinpath(output_dir, "top_models.png"))
    _plot_top_models(names, index, counts, vanilla_s, consistent_s, consistent_b, learned_beta, length(names), joinpath(output_dir, "all_models.png"))
    _write_probability_matrix(output_dir, "vanilla", names, _pairwise_win_probability_matrix(names, index, vanilla_s, learned_beta, epsilon); ranking=_skill_ranking(names, index, vanilla_s))
    _write_probability_matrix(output_dir, "robust", names, _pairwise_win_probability_matrix(names, index, robust_s, learned_beta, epsilon); ranking=_skill_ranking(names, index, robust_s))
    _write_probability_matrix(output_dir, "consistent", names, _pairwise_win_probability_matrix(names, index, consistent_s, learned_beta, epsilon; precisions=consistent_b); ranking=_skill_ranking(names, index, consistent_s))
    open(joinpath(output_dir, "run.log"), "w") do io
        println(io, "LLM Arena TrueSkill experiment")
        println(io, "dataset=$dataset output_dir=$output_dir fit_mode=$fit_mode")
        println(io, "n_battles=$(length(matches)) n_models=$(length(names)) n_draws=$(count(draws)) split_seed=$split_seed")
        println(io, "selected: beta=$learned_beta robust_nu=$learned_robust_nu consistent_nu=$learned_consistency_nu draw_margin=$epsilon")
        println(io, "validation grids: beta=$(effective_beta_grid) robust_nu=$(effective_robust_nu_grid) consistent_nu=$(effective_consistent_nu_grid)")
        println(io, "The selected setting was refit on train+validation before all test, predictive, plot, matrix, report, and noisiest-battle artifacts were written.")
        println(io, "artifacts: metadata.csv; split_summary.csv; robust_noisiest_battles.csv; train/test outcome and predictive metrics; model_report.csv")
    end
    return (report=report, validation_scores=validation_scores, test_scores=test_scores,
            train_scores=train_scores, test_predictive_scores=test_predictive_scores, train_predictive_scores=train_predictive_scores,
            noisiest_battles=noisiest_battles, metadata=metadata, learned_beta=learned_beta,
            learned_consistency_nu=learned_consistency_nu, learned_robust_nu=learned_robust_nu)
end

"""Full calibrated run for the human-preference dataset (long-running)."""
function run_llm_arena_calibrated_experiment()
    return run_llm_arena_experiment(
        dataset=:human_preference_140k,
        min_games=100,
        outer_max_iters=60,
        inner_max_iters=60,
        top_n=15,
    )
end

"""Full calibrated run for the smaller expert-vote dataset."""
function run_llm_arena_expert_calibrated_experiment()
    return run_llm_arena_experiment(
        dataset=:expert_5k,
        min_games=20,
        outer_max_iters=60,
        inner_max_iters=60,
        top_n=15,
    )
end

"""Coarse, reproducible real-data hyperparameter grid for LLM Arena.

This phase searches the statistical parameters on a capped but real training
prefix/subsample, records both train and validation likelihoods, and writes
every evaluated configuration.  Solver controls are intentionally fixed here;
they are numerical convergence settings and are swept separately around the
validation-selected statistical configuration.
"""
function run_llm_arena_real_grid_search(; dataset::Symbol=:expert_5k,
    min_games::Int=(dataset === :expert_5k ? 20 : 100), max_fit_matches::Int=0,
    max_metric_matches::Int=0, beta_grid::Vector{Float64}=[8.0, 12.0, 16.0, 24.0, 32.0],
    nu_grid::Vector{Float64}=[2.05, 2.25, 2.5, 3.0, 4.0, 6.0, 8.0, 16.0, 64.0, 256.0],
    outer_tol::Float64=1e-4, outer_max_iters::Int=60, inner_tol::Float64=1e-4,
    inner_max_iters::Int=60, schedule::Symbol=:interleaved, local_tol::Float64=1e-4,
    local_max_iters::Int=1, quadrature_points::Int=16, split_seed::Int=42,
    output_dir::AbstractString=joinpath(DEFAULT_OUTPUT_DIR, String(dataset), "hyperparameter_search", "coarse_current"))
    data_dir = joinpath(output_dir, "data")
    logs_dir = joinpath(output_dir, "logs")
    foreach(mkpath, (data_dir, logs_dir))
    foreach(model -> mkpath(joinpath(output_dir, "models", model)), ("vanilla", "robust", "consistent"))
    battles = _load_battles(nothing; dataset=dataset, min_games=min_games)
    # Timestamped releases are static preference-ranking data here, not
    # temporal forecasts. Only Human Preference has a session grouping; it is
    # randomized inside `_split_battle_indices` instead of shuffling rows.
    battles = _prepare_battle_split(battles; seed=split_seed)
    names, _, matches, draws = _index_battles(battles)
    train, validation, _ = _split_battle_indices(battles, 0.70, 0.15; seed=split_seed)
    # The draw margin is fitted from outcomes, so validation labels must not
    # enter this calibration.  The final refit can use train+validation only
    # after all statistical hyperparameters have been selected.
    calibration_indices = train
    empirical_draw_rate = mean(draws[calibration_indices])
    function sample_indices(indices, limit, seed)
        (limit <= 0 || length(indices) <= limit) && return collect(indices)
        rng = MersenneTwister(seed)
        values = collect(indices)
        return sort(values[randperm(rng, length(values))[1:limit]])
    end
    fit_indices = sample_indices(train, max_fit_matches, split_seed)
    train_metric_indices = sample_indices(train, max_metric_matches, split_seed + 1)
    validation_metric_indices = sample_indices(validation, max_metric_matches, split_seed + 2)
    prior_s = TSG.Gaussian1DFromMeanVariance(25.0, (25.0 / 3)^2)
    rows = DataFrame(model=String[], beta=Float64[], empirical_draw_rate=Float64[], epsilon=Float64[], nu=Float64[],
        schedule=String[], local_max_iters=Int[], local_tol=Float64[], inner_tol=Float64[], inner_max_iters=Int[],
        outer_tol=Float64[], outer_max_iters=Int[], n_fit=Int[], n_train_metric=Int[], n_validation_metric=Int[],
        runtime_seconds=Float64[], train_nll=Float64[], validation_nll=Float64[], quadrature_points=Int[])
    progress_path = joinpath(logs_dir, "progress.log")
    open(progress_path, "w") do io; println(io, "$(Dates.now()) LLM Arena grid started: $(dataset)"); end
    function log_progress(message)
        line = "$(Dates.now()) $(message)"
        open(progress_path, "a") do io; println(io, line); flush(io); end
        println(line); flush(stdout)
    end
    function record!(model, beta, nu, posteriors, elapsed; precisions=nothing)
        epsilon = _draw_margin(beta, empirical_draw_rate)
        predictive_kwargs = model == "vanilla" ? (; model=:vanilla) :
            model == "robust" ? (; model=:robust, nu=nu, quadrature_points=quadrature_points) :
            (; model=:consistent, precisions=precisions, quadrature_points=quadrature_points)
        train_metrics = _model_predictive_metrics(posteriors, matches[train_metric_indices], draws[train_metric_indices], beta, epsilon; predictive_kwargs...)
        validation_metrics = _model_predictive_metrics(posteriors, matches[validation_metric_indices], draws[validation_metric_indices], beta, epsilon; predictive_kwargs...)
        push!(rows, (model=model, beta=beta, empirical_draw_rate=empirical_draw_rate, epsilon=epsilon, nu=nu,
            schedule=String(schedule), local_max_iters=local_max_iters, local_tol=local_tol,
            inner_tol=inner_tol, inner_max_iters=inner_max_iters, outer_tol=outer_tol,
            outer_max_iters=outer_max_iters, n_fit=length(fit_indices), n_train_metric=length(train_metric_indices),
            n_validation_metric=length(validation_metric_indices), runtime_seconds=elapsed,
            train_nll=train_metrics.nll, validation_nll=validation_metrics.nll,
            quadrature_points=quadrature_points))
    end
    for beta in beta_grid
        epsilon = _draw_margin(beta, empirical_draw_rate)
        vanilla_timed = @timed _fit_vanilla(prior_s, matches[fit_indices], draws[fit_indices], beta, epsilon, outer_max_iters, false)
        record!("vanilla", beta, NaN, vanilla_timed.value, vanilla_timed.time)
        for nu in nu_grid
            robust_timed = @timed _fit_robust(prior_s, matches[fit_indices], draws[fit_indices], beta, epsilon, nu,
                inner_max_iters, outer_max_iters, false; schedule=schedule, local_max_iters=local_max_iters, local_tol=local_tol)
            robust_s, _ = robust_timed.value
            record!("robust", beta, nu, robust_s, robust_timed.time)
            prior_b = _precision_prior(nu)
            consistent_timed = @timed TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)),
                TSDC.DistributionBag(prior_b, length(names)), matches[fit_indices], epsilon; s2=beta^2,
                draw_mask=draws[fit_indices], inner_tol=inner_tol, inner_max_iters=inner_max_iters,
                outer_tol=outer_tol, outer_max_iters=outer_max_iters, schedule=schedule,
                local_max_iters=local_max_iters, local_tol=local_tol, verbose=false)
            consistent_s, consistent_b = consistent_timed.value
            record!("consistent", beta, nu, consistent_s, consistent_timed.time; precisions=consistent_b)
            log_progress("beta=$(beta), empirical_draw_rate=$(empirical_draw_rate), nu=$(nu) complete")
        end
        CSV.write(joinpath(data_dir, "grid_results.checkpoint.csv"), rows)
    end
    CSV.write(joinpath(data_dir, "grid_results.csv"), rows)
    best_rows = DataFrame()
    for criterion in (:train_nll, :validation_nll), model in unique(rows.model)
        subset = filter(row -> row.model == model, rows)
        best = subset[argmin(subset[!, criterion]), :]
        item = DataFrame(best)
        item[!, :selection_metric] = [String(criterion)]
        append!(best_rows, item; cols=:union)
    end
    CSV.write(joinpath(data_dir, "best_hyperparameters_by_metric.csv"), best_rows)
    for metric in (:train_nll, :validation_nll)
        for model in ("robust", "consistent")
            subset = filter(row -> row.model == model, rows)
            values = [only(filter(row -> row.beta == beta && row.nu == nu, subset)[!, metric]) for beta in beta_grid, nu in nu_grid]
            figure = heatmap(1:length(nu_grid), 1:length(beta_grid), values; xlabel="nu", ylabel="beta",
                xticks=(1:length(nu_grid), string.(nu_grid)), yticks=(1:length(beta_grid), string.(beta_grid)),
                xrotation=45, yflip=true, title="$(dataset): $(model), empirical draw rate=$(round(empirical_draw_rate; digits=3)), $(metric)",
                colorbar_title=String(metric), size=(1200, 750), bottom_margin=25Plots.mm, left_margin=18Plots.mm)
            savefig(figure, joinpath(output_dir, "models", model, "$(metric)_heatmap.png"))
        end
        figure = plot(xlabel="beta", ylabel=String(metric), title="$(dataset): vanilla $(metric)", legend=:best)
        subset = sort(filter(row -> row.model == "vanilla", rows), :beta)
        plot!(figure, subset.beta, subset[!, metric]; marker=:circle, label="empirical draw rate=$(round(empirical_draw_rate; digits=3))")
        savefig(figure, joinpath(output_dir, "models", "vanilla", "$(metric)_lines.png"))
    end
    return rows
end

function run_llm_arena_expert_real_grid_search()
    return run_llm_arena_real_grid_search(
        dataset=:expert_5k, min_games=20,
        beta_grid=[8.0, 12.0, 16.0, 20.0, 24.0, 32.0],
        nu_grid=[2.05, 2.25, 2.5, 3.0, 4.0, 6.0, 8.0, 16.0, 64.0, 256.0],
    )
end

function run_llm_arena_human_real_grid_search()
    return run_llm_arena_real_grid_search(
        dataset=:human_preference_140k, min_games=100,
        # A real but bounded training subset makes the coarse grid feasible;
        # final selected configurations are subsequently refit on all battles.
        max_fit_matches=25_000, max_metric_matches=15_000,
        beta_grid=[8.0, 12.0, 16.0, 24.0, 32.0],
        nu_grid=[2.05, 2.25, 2.5, 3.0, 4.0, 6.0, 8.0, 16.0, 64.0],
    )
end

"""Refine the full-data Arena grids around the coarse validation optima.

The coarse Human Preference-140k search used capped fits to locate the useful
region cheaply.  This phase uses *all* training and validation battles for the
same random/chronological split, a fixed draw margin calibrated from
train-plus-validation outcomes, and the posterior-predictive likelihood of
each respective model. Solver controls are held fixed rather than treated as
statistical hyperparameters.
"""
function run_llm_arena_refined_full_grid_search(; dataset::Symbol)
    dataset in (:expert_5k, :human_preference_140k) || error("Unsupported dataset: $dataset")
    beta_grid = dataset === :expert_5k ? [16.0, 20.0, 24.0, 28.0, 32.0, 36.0] :
        [24.0, 28.0, 32.0, 36.0, 40.0]
    nu_grid = dataset === :expert_5k ? [2.5, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0, 64.0, 128.0] :
        [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    return run_llm_arena_real_grid_search(
        dataset=dataset,
        min_games=dataset === :expert_5k ? 20 : 100,
        max_fit_matches=0,
        max_metric_matches=0,
        beta_grid=beta_grid,
        nu_grid=nu_grid,
        output_dir=joinpath(DEFAULT_OUTPUT_DIR, String(dataset), "hyperparameter_search", "refined_full"),
    )
end

"""Sweep valid log-spaced Gamma degrees of freedom on Arena Expert.

The requested grid is nu=2^n for n=-10:10.  Values nu<=2 are recorded as
invalid because neither the variance-matched robust model nor the consistency
predictive variance is finite there; fitting uses n=2:10.
"""
function run_llm_arena_expert_nu_sweep()
    output_dir = joinpath(DEFAULT_OUTPUT_DIR, "expert_5k", "nu_sweep")
    mkpath(output_dir)
    battles = _load_battles(nothing; dataset=:expert_5k, min_games=20)
    Random.seed!(42); battles = battles[randperm(nrow(battles)), :]
    names, index, matches, draws = _index_battles(battles)
    counts = Dict(name => count(==(name), battles.model_a) + count(==(name), battles.model_b) for name in names)
    train, validation, test = _split_indices(length(matches), 0.70, 0.15)
    trainval = 1:last(validation)
    beta, epsilon = 12.0, _draw_margin(12.0, mean(draws[train]))
    prior_s = TSG.Gaussian1DFromMeanVariance(25.0, (25.0 / 3)^2)
    nus = 2.0 .^ collect(2:10)
    vanilla_s = _fit_vanilla(prior_s, matches[train], draws[train], beta, epsilon, 60, false)
    rows = DataFrame(model=String[], nu=Float64[], split=String[], nll=Float64[], mean_log_probability=Float64[], accuracy=Float64[])
    function add_metrics!(model, nu, split_name, split, posteriors; precisions=nothing)
        metric = _outcome_metrics(posteriors, matches[split], draws[split], beta, epsilon;
            model=Symbol(model), precisions=precisions, nu=nu)
        push!(rows, (model=model, nu=nu, split=split_name, nll=metric.nll, mean_log_probability=metric.mean_log_probability, accuracy=metric.accuracy))
    end
    for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test)); add_metrics!("vanilla", NaN, split_name, split, vanilla_s); end
    for nu in nus
        prior_b = _precision_prior(nu)
        cs, cb = TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)), TSDC.DistributionBag(prior_b, length(names)), matches[train], epsilon; s2=beta^2, draw_mask=draws[train], inner_max_iters=60, outer_max_iters=60, verbose=false)
        for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test)); add_metrics!("consistent", nu, split_name, split, cs; precisions=cb); end
        rs, _ = _fit_robust(prior_s, matches[train], draws[train], beta, epsilon, nu, 60, 60, false)
        for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test)); add_metrics!("robust", nu, split_name, split, rs); end
    end
    invalid = DataFrame(n=collect(-10:1), nu=2.0 .^ collect(-10:1), reason=fill("nu <= 2 gives infinite predictive variance", 12))
    CSV.write(joinpath(output_dir, "nu_sweep_metrics.csv"), rows)
    CSV.write(joinpath(output_dir, "invalid_nu_values.csv"), invalid)
    panels = Any[]
    for split in ("train", "validation", "test")
        panel = plot(xscale=:log2, xlabel="nu", ylabel="NLL", title=split, legend=:topright)
        for model in ("robust", "consistent")
            subset = filter(row -> row.model == model && row.split == split, rows)
            plot!(panel, subset.nu, subset.nll; marker=:circle, label=model)
        end
        vanilla = filter(row -> row.model == "vanilla" && row.split == split, rows).nll[1]
        hline!(panel, [vanilla]; label="vanilla")
        push!(panels, panel)
    end
    nu_figure = plot(panels...; layout=(1, 3), size=(1800, 450))
    savefig(nu_figure, joinpath(output_dir, "nll_by_nu.png"))
    savefig(nu_figure, joinpath(output_dir, "nll_by_nu.pdf"))
    for model in ("robust", "consistent")
        best = sort(filter(row -> row.model == model && row.split == "validation", rows), :nll).nu[1]
        if model == "robust"
            s, _ = _fit_robust(prior_s, matches[trainval], draws[trainval], beta, epsilon, best, 60, 60, false)
            _write_probability_matrix(output_dir, model, names, _pairwise_win_probability_matrix(names, index, s, beta, epsilon); ranking=_skill_ranking(names, index, s))
        else
            b = _precision_prior(best)
            s, p = TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)), TSDC.DistributionBag(b, length(names)), matches[trainval], epsilon; s2=beta^2, draw_mask=draws[trainval], inner_max_iters=60, outer_max_iters=60, verbose=false)
            _write_probability_matrix(output_dir, model, names, _pairwise_win_probability_matrix(names, index, s, beta, epsilon; precisions=p); ranking=_skill_ranking(names, index, s))
        end
    end
    vs = _fit_vanilla(prior_s, matches[trainval], draws[trainval], beta, epsilon, 60, false)
    _write_probability_matrix(output_dir, "vanilla", names, _pairwise_win_probability_matrix(names, index, vs, beta, epsilon); ranking=_skill_ranking(names, index, vs))
    return rows
end

"""Joint beta/nu calibration on Arena Expert, with a held-out test split.

The nu-only sweep fixes beta to the value selected by vanilla TrueSkill.  This
run checks whether robust or player-consistency improvements were masked by
that shared choice.  Beta controls the baseline performance standard deviation
and receives its own draw margin estimated from the training split.
"""
function run_llm_arena_expert_beta_nu_sweep()
    output_dir = joinpath(DEFAULT_OUTPUT_DIR, "expert_5k", "beta_nu_sweep_extended_beta32")
    mkpath(output_dir)
    battles = _load_battles(nothing; dataset=:expert_5k, min_games=20)
    Random.seed!(42); battles = battles[randperm(nrow(battles)), :]
    names, index, matches, draws = _index_battles(battles)
    train, validation, test = _split_indices(length(matches), 0.70, 0.15)
    trainval = 1:last(validation)
    prior_s = TSG.Gaussian1DFromMeanVariance(25.0, (25.0 / 3)^2)
    # The dense nu-only sweep already covers 4,...,1024.  Here we add the
    # low-variance and near-heavy-tail-boundary regimes while retaining
    # enough high-nu values to detect the vanilla limit.
    # The first sweep selected beta=16 at the grid boundary, so extend the
    # scale upward before treating it as an optimum.
    betas = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 32.0]
    nus = [2.1, 2.5, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 1024.0]
    rows = DataFrame(model=String[], beta=Float64[], nu=Float64[], split=String[], nll=Float64[], mean_log_probability=Float64[], accuracy=Float64[])
    function add_metrics!(model, beta, nu, split_name, split, posteriors; precisions=nothing)
        epsilon = _draw_margin(beta, mean(draws[train]))
        metric = _outcome_metrics(posteriors, matches[split], draws[split], beta, epsilon;
            model=Symbol(model), precisions=precisions, nu=nu)
        push!(rows, (model=model, beta=beta, nu=nu, split=split_name, nll=metric.nll, mean_log_probability=metric.mean_log_probability, accuracy=metric.accuracy))
    end
    for beta in betas
        epsilon = _draw_margin(beta, mean(draws[train]))
        vs = _fit_vanilla(prior_s, matches[train], draws[train], beta, epsilon, 60, false)
        for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test)); add_metrics!("vanilla", beta, NaN, split_name, split, vs); end
        for nu in nus
            prior_b = _precision_prior(nu)
            cs, cb = TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)), TSDC.DistributionBag(prior_b, length(names)), matches[train], epsilon; s2=beta^2, draw_mask=draws[train], inner_max_iters=60, outer_max_iters=60, verbose=false)
            for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test)); add_metrics!("consistent", beta, nu, split_name, split, cs; precisions=cb); end
            rs, _ = _fit_robust(prior_s, matches[train], draws[train], beta, epsilon, nu, 60, 60, false)
            for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test)); add_metrics!("robust", beta, nu, split_name, split, rs); end
        end
    end
    CSV.write(joinpath(output_dir, "beta_nu_sweep_metrics.csv"), rows)
    for model in ("vanilla", "robust", "consistent")
        subset = filter(row -> row.model == model && row.split == "validation", rows)
        best = sort(subset, :nll)[1, :]
        CSV.write(joinpath(output_dir, "$(model)_best_validation_hyperparameters.csv"), DataFrame(best))
    end
    _write_beta_nu_sweep_plots(rows, output_dir)
    return rows
end

"""Re-render sweep visualizations from an existing CSV without refitting."""
function rerender_llm_arena_expert_beta_nu_sweep_plots()
    output_dir = joinpath(DEFAULT_OUTPUT_DIR, "expert_5k", "beta_nu_sweep_extended_beta32")
    rows = CSV.read(joinpath(output_dir, "beta_nu_sweep_metrics.csv"), DataFrame)
    _write_beta_nu_sweep_plots(rows, output_dir)
    return nothing
end

"""Fine validation sweep with posterior-predictive NLL scores.

The likelihood integrates the Gaussian--Gamma posterior-predictive distribution
by deterministic quadrature.  No test-set result is used here.
"""
function run_llm_arena_expert_fine_sweep()
    output_dir = joinpath(DEFAULT_OUTPUT_DIR, "expert_5k", "fine_beta_nu_sweep")
    mkpath(output_dir)
    battles = _load_battles(nothing; dataset=:expert_5k, min_games=20)
    Random.seed!(42); battles = battles[randperm(nrow(battles)), :]
    names, index, matches, draws = _index_battles(battles)
    train, validation, _ = _split_indices(length(matches), 0.70, 0.15)
    prior_s = TSG.Gaussian1DFromMeanVariance(25.0, (25.0 / 3)^2)
    betas = [16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 48.0, 56.0, 64.0]
    nus = [2.05, 2.1, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0,
        16.0, 24.0, 32.0, 48.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    quadrature_points = 16
    rows = DataFrame(model=String[], beta=Float64[], nu=Float64[], nll=Float64[],
        mean_log_probability=Float64[], quadrature_points=Int[])
    progress_path = joinpath(output_dir, "progress.log")
    open(progress_path, "w") do io
        println(io, "$(Dates.now()) fine sweep started; Gamma(shape=nu/2, rate=nu/2), Gauss-Laguerre nodes=$(quadrature_points)")
    end
    function log_progress(message)
        line = "$(Dates.now()) $(message)"
        open(progress_path, "a") do io
            println(io, line)
            flush(io)
        end
        println(line)
        flush(stdout)
    end
    function add_row!(model, beta, nu, posteriors; precisions=nothing)
        epsilon = _draw_margin(beta, mean(draws[train]))
        metrics = model == "vanilla" ?
            _model_predictive_metrics(posteriors, matches[validation], draws[validation], beta, epsilon; model=:vanilla) :
            model == "robust" ?
                _model_predictive_metrics(posteriors, matches[validation], draws[validation], beta, epsilon;
                    model=:robust, nu=nu, quadrature_points=quadrature_points) :
                _model_predictive_metrics(posteriors, matches[validation], draws[validation], beta, epsilon;
                    model=:consistent, precisions=precisions, quadrature_points=quadrature_points)
        push!(rows, (model=model, beta=beta, nu=nu, nll=metrics.nll,
            mean_log_probability=metrics.mean_log_probability,
            quadrature_points=quadrature_points))
    end
    for (beta_index, beta) in enumerate(betas)
        log_progress("beta $(beta) ($(beta_index)/$(length(betas))) started")
        epsilon = _draw_margin(beta, mean(draws[train]))
        vanilla = _fit_vanilla(prior_s, matches[train], draws[train], beta, epsilon, 60, false)
        add_row!("vanilla", beta, NaN, vanilla)
        for (nu_index, nu) in enumerate(nus)
            robust, _ = _fit_robust(prior_s, matches[train], draws[train], beta, epsilon, nu, 60, 60, false)
            add_row!("robust", beta, nu, robust)
            prior_b = _precision_prior(nu)
            consistent, precisions = TS.consistent_trueskill(TSDC.DistributionBag(prior_s, length(names)),
                TSDC.DistributionBag(prior_b, length(names)), matches[train], epsilon; s2=beta^2,
                draw_mask=draws[train], inner_max_iters=60, outer_max_iters=60, verbose=false)
            add_row!("consistent", beta, nu, consistent; precisions=precisions)
            log_progress("beta $(beta) ($(beta_index)/$(length(betas))), nu $(nu) ($(nu_index)/$(length(nus))) complete")
        end
        CSV.write(joinpath(output_dir, "fine_sweep_validation.checkpoint.csv"), rows)
        log_progress("beta $(beta) checkpoint written ($(nrow(rows)) rows)")
    end
    CSV.write(joinpath(output_dir, "fine_sweep_validation.csv"), rows)
    for metric in (:nll,)
        for model in ("robust", "consistent")
            subset = filter(row -> row.model == model, rows)
            values = [only(filter(row -> row.beta == beta && row.nu == nu, subset)[!, metric]) for beta in betas, nu in nus]
            figure = heatmap(collect(eachindex(nus)), collect(eachindex(betas)), values;
                xlabel="nu", ylabel="beta", title="$(model) validation $(metric)",
                xticks=(collect(eachindex(nus)), string.(nus)), yticks=(collect(eachindex(betas)), string.(betas)),
                xrotation=45, tickfont=font(8), colorbar_title=String(metric), yflip=true,
                size=(1250, 750), left_margin=18Plots.mm, bottom_margin=26Plots.mm)
            savefig(figure, joinpath(output_dir, "$(model)_$(metric)_heatmap.png"))
            savefig(figure, joinpath(output_dir, "$(model)_$(metric)_heatmap.pdf"))
        end
    end
    for metric in (:nll,)
        vanilla = sort(filter(row -> row.model == "vanilla", rows), :beta)
        figure = plot(vanilla.beta, vanilla[!, metric]; marker=:circle, xlabel="beta", ylabel=String(metric),
            title="vanilla validation $(metric)", legend=false, size=(900, 620), left_margin=14Plots.mm, bottom_margin=14Plots.mm)
        savefig(figure, joinpath(output_dir, "vanilla_$(metric)_by_beta.png"))
        savefig(figure, joinpath(output_dir, "vanilla_$(metric)_by_beta.pdf"))
    end
    return rows
end

"""Compare the original interleaved EP/VI schedule with local VI blocks.

Hyperparameters default to the validation choices from the joint Expert
beta/nu sweep (beta=16, robust nu=2.5, consistent nu=256).  Both schedules are trained only
on the training split and evaluated on train/validation/test, so this isolates
message scheduling rather than hyperparameter selection.
"""
function run_llm_arena_expert_schedule_comparison(; beta::Float64=16.0,
    robust_nu::Float64=2.5, consistent_nu::Float64=256.0,
    local_max_iters::Int=10, local_tol::Float64=1e-4,
    inner_max_iters::Int=60, outer_max_iters::Int=60)
    output_dir = joinpath(DEFAULT_OUTPUT_DIR, "expert_5k", "schedule_comparison_beta$(Int(beta))")
    mkpath(output_dir)
    battles = _load_battles(nothing; dataset=:expert_5k, min_games=20)
    Random.seed!(42); battles = battles[randperm(nrow(battles)), :]
    names, index, matches, draws = _index_battles(battles)
    train, validation, test = _split_indices(length(matches), 0.70, 0.15)
    epsilon = _draw_margin(beta, mean(draws[train]))
    prior_s = TSG.Gaussian1DFromMeanVariance(25.0, (25.0 / 3)^2)
    rows = DataFrame(model=String[], schedule=String[], local_max_iters=Int[], runtime_seconds=Float64[],
        split=String[], nll=Float64[], mean_log_probability=Float64[], accuracy=Float64[])
    function record!(model, schedule, epochs, elapsed, posteriors; precisions=nothing)
        for (split_name, split) in zip(("train", "validation", "test"), (train, validation, test))
            nu = model == "robust" ? robust_nu : model == "consistent" ? consistent_nu : nothing
            metric = _outcome_metrics(posteriors, matches[split], draws[split], beta, epsilon;
                model=Symbol(model), precisions=precisions, nu=nu)
            push!(rows, (model=model, schedule=String(schedule), local_max_iters=epochs,
                runtime_seconds=elapsed, split=split_name, nll=metric.nll,
                mean_log_probability=metric.mean_log_probability, accuracy=metric.accuracy))
        end
        stats = DataFrame(model=names,
            skill=[TSG.mean(posteriors[index[name]]) for name in names],
            skill_sd=[sqrt(TSG.variance(posteriors[index[name]])) for name in names])
        if precisions !== nothing
            stats[!, :precision] = [TSGamma.mean(precisions[index[name]]) for name in names]
            stats[!, :performance_sd] = [beta * sqrt(TSGamma.expected_inverse_precision(precisions[index[name]])) for name in names]
        end
        sort!(stats, :skill, rev=true)
        prefix = "$(model)_$(schedule)_local$(epochs)"
        CSV.write(joinpath(output_dir, "$(prefix)_final_player_statistics.csv"), stats)
        _write_probability_matrix(output_dir, prefix, names,
            _pairwise_win_probability_matrix(names, index, posteriors, beta, epsilon; precisions=precisions);
            ranking=_skill_ranking(names, index, posteriors))
    end
    for (schedule, epochs) in ((:interleaved, 1), (:local_vi, local_max_iters))
        robust_timed = @timed _fit_robust(prior_s, matches[train], draws[train], beta, epsilon,
            robust_nu, inner_max_iters, outer_max_iters, false; schedule=schedule,
            local_max_iters=epochs, local_tol=local_tol)
        robust_s, robust_b = robust_timed.value
        # Robust precisions are match-specific, so they do not enter a
        # player-level consistency table or the moment-matched predictor.
        record!("robust", schedule, epochs, robust_timed.time, robust_s)

        prior_b = _precision_prior(consistent_nu)
        consistent_timed = @timed TS.consistent_trueskill(
            TSDC.DistributionBag(prior_s, length(names)), TSDC.DistributionBag(prior_b, length(names)),
            matches[train], epsilon; s2=beta^2, draw_mask=draws[train], inner_max_iters=inner_max_iters,
            outer_max_iters=outer_max_iters, schedule=schedule, local_max_iters=epochs,
            local_tol=local_tol, verbose=false)
        consistent_s, consistent_b = consistent_timed.value
        record!("consistent", schedule, epochs, consistent_timed.time, consistent_s; precisions=consistent_b)
    end
    CSV.write(joinpath(output_dir, "schedule_comparison_metrics.csv"), rows)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_llm_arena_experiment()
end
