#=Synthetic corruption benchmark shared by the library and downstream experiments.=#

"""Generate the synthetic ranking/corruption dataset used in the paper experiments.

Players have ground-truth skills `1:k`. Base outcomes follow that order. In
`:match` mode, a fraction `corruption` of individual outcomes is reversed. In
`:player` mode, a persistent subset of unreliable players causes reversals.
"""
function generate_corruption_dataset(; k::Int=40, n::Int=10_000, seed::Int=42,
    corruption::Float64=0.1, mode::Symbol=:match)
    rng = MersenneTwister(seed)
    true_skills = Float64.(1:k)
    clean_matches = Vector{Tuple{Int64, Int64}}(undef, n)
    for i in eachindex(clean_matches)
        pair = randperm(rng, k)[1:2]
        clean_matches[i] = pair[1] > pair[2] ? (pair[1], pair[2]) : (pair[2], pair[1])
    end
    matches = copy(clean_matches)
    corrupted = falses(n)
    noisy_players = falses(k)
    if mode === :match
        count_corrupt = round(Int, corruption * n)
        for i in randperm(rng, n)[1:count_corrupt]
            matches[i] = reverse(matches[i])
            corrupted[i] = true
        end
    elseif mode === :player
        n_noisy = round(Int, sqrt(corruption) * k / 2)
        flip_probability = sqrt(corruption)
        if n_noisy > 0
            noisy_players[randperm(rng, k)[1:n_noisy]] .= true
            for i in eachindex(matches)
                winner, loser = matches[i]
                if (noisy_players[winner] || noisy_players[loser]) && rand(rng) < flip_probability
                    matches[i] = (loser, winner)
                    corrupted[i] = true
                end
            end
        end
    else
        error("mode must be :match or :player")
    end
    return (
        name=:synthetic,
        player_names=["player_$(i)" for i in 1:k],
        true_skills=true_skills,
        clean_matches=clean_matches,
        matches=matches,
        draws=falses(n),
        corrupted=corrupted,
        noisy_players=noisy_players,
        beta=k / 12,
        epsilon=0.0,
        prior_mu=k / 2,
        prior_sigma=k / 6,
        corruption=corruption,
        corruption_mode=mode,
        seed=seed,
    )
end

"""Probability that a Gamma precision variable lies below `threshold`."""
function precision_flag_probability(g::Gamma1D, threshold::Float64=1.0)
    return cdf(Distributions.Gamma(Gamma.shape(g), 1 / Gamma.rate(g)), threshold)
end

function _corruption_auroc(labels::AbstractVector{Bool}, scores::AbstractVector)
    positives, negatives = findall(labels), findall(.!labels)
    (isempty(positives) || isempty(negatives)) && return NaN
    return Statistics.mean(scores[positive] < scores[negative] ? 1.0 :
        scores[positive] == scores[negative] ? 0.5 : 0.0
        for positive in positives, negative in negatives)
end

"""Default location for the original synthetic corruption benchmark."""
function corruption_output_directory(k::Int, n::Int, seed::Int)
    return joinpath(@__DIR__, "..", "..", "output", "synthetic", "corruption_sweep", "k$(k)_n$(n)_seed$(seed)")
end

"""Run the original synthetic corruption benchmark and save its plots.

The benchmark jointly visualizes the match-corruption and player-corruption
regimes, so it is stored under the generic synthetic `corruption_sweep` rather
than either single-regime downstream directory.
"""
function experiment_corruption_robustness(; k::Int=40, n::Int=10_000, seed::Int=42,
    deltas=collect(0.0:0.02:0.2), output_dir::Union{Nothing, AbstractString}=nothing)
    output_dir = isnothing(output_dir) ? corruption_output_directory(k, n, seed) : String(output_dir)
    data_dir = joinpath(output_dir, "data")
    overview_dir = joinpath(output_dir, "overview")
    foreach(mkpath, (data_dir, overview_dir))
    prior_s = Gaussian1DFromMeanVariance(k / 2, (k / 6)^2)
    prior_b = Gamma1D(1.0, 2.0) # Gamma(shape=2, rate=2), equivalent to nu=4.
    beta2, epsilon = (k / 12)^2, 0.0
    variance_match = 0.5 # (nu - 2) / nu for nu=4.
    rows = DataFrame(mode=String[], corruption=Float64[], model=String[],
        kendall_tau=Float64[], detection_auroc=Float64[])
    for mode in (:match, :player), delta in deltas
        data = generate_corruption_dataset(; k=k, n=n, seed=seed, corruption=delta, mode=mode)
        vanilla = vanilla_trueskill(DistributionBag(prior_s, k), beta2, data.matches, epsilon)
        robust, robust_b = robust_trueskill(DistributionBag(prior_s, k),
            DistributionBag(prior_b, n), data.matches, epsilon;
            s2=2beta2 * variance_match)
        consistent, consistent_b = consistent_trueskill(DistributionBag(prior_s, k),
            DistributionBag(prior_b, k), data.matches, epsilon;
            s2=beta2 * variance_match)
        detection = Dict(
            "vanilla" => NaN,
            "robust" => _corruption_auroc(data.corrupted, Gamma.mean.(robust_b)),
            "consistent" => _corruption_auroc(data.noisy_players, Gamma.mean.(consistent_b)),
        )
        for (name, posterior) in (("vanilla", vanilla), ("robust", robust), ("consistent", consistent))
            push!(rows, (String(mode), delta, name,
                kendall_tau(data.true_skills, Gaussian.mean.(posterior)), detection[name]))
        end
    end
    CSV.write(joinpath(data_dir, "corruption_metrics.csv"), rows)
    ranking_plot = plot(; xlabel="% corrupted comparisons", ylabel="Kendall's tau",
        title="Synthetic corruption robustness", size=(900, 600))
    detection_plot = plot(; xlabel="% corrupted comparisons", ylabel="AUROC",
        title="Precision-based anomaly detection", size=(900, 600), ylims=(0.4, 1.05))
    colors = Dict("vanilla" => :steelblue, "robust" => :darkorange, "consistent" => :forestgreen)
    for mode in ("match", "player"), model in ("vanilla", "robust", "consistent")
        subset = filter(row -> row.mode == mode && row.model == model, rows)
        plot!(ranking_plot, 100 .* subset.corruption, subset.kendall_tau; label="$model / $mode",
            color=colors[model], linestyle=mode == "match" ? :solid : :dash, linewidth=2)
        finite = filter(row -> isfinite(row.detection_auroc), subset)
        isempty(finite) || plot!(detection_plot, 100 .* finite.corruption, finite.detection_auroc;
            label="$model / $mode", color=colors[model],
            linestyle=mode == "match" ? :solid : :dash, linewidth=2)
    end
    hline!(detection_plot, [0.5]; color=:gray, linestyle=:dot, label="chance")
    figure = plot(ranking_plot, detection_plot; layout=(1, 2), size=(1600, 600))
    savefig(figure, joinpath(overview_dir, "corruption.png"))
    savefig(figure, joinpath(overview_dir, "corruption.pdf"))
    return rows
end

demo6(; k=40, n=1_000, seed=123, output_dir=nothing) =
    experiment_corruption_robustness(; k=k, n=n, seed=seed, output_dir=output_dir)

export generate_corruption_dataset, precision_flag_probability, corruption_output_directory,
    experiment_corruption_robustness, demo6
