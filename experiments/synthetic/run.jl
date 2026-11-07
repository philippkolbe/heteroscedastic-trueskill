#=Synthetic ranking and noise-detection experiment.
Uses Gamma(1,1) precision priors and exports metrics and five comparison panels.
=#
module PaperSynthetic
include(joinpath(@__DIR__, "..", "..", "src", "trueskill.jl"))
using .TrueSkill
using .TrueSkill.Gaussian: Gaussian1DFromMeanVariance
using .TrueSkill.Gamma: Gamma1D
using .TrueSkill.DistributionCollections: DistributionBag
using .TrueSkill.MyMath: kendall_tau
import .TrueSkill.Gaussian
import .TrueSkill.Gamma
const GaussianBag = TrueSkill.GaussianBag
const vanilla_trueskill = TrueSkill.vanilla_trueskill
const robust_trueskill = TrueSkill.robust_trueskill
const consistent_trueskill = TrueSkill.consistent_trueskill
using Random, Statistics, Printf, CSV, DataFrames, Plots
using StatsBase: sample
function _auroc_lower_is_positive(labels::Vector{Float64}, scores::Vector{Float64})::Float64
    n_pos = sum(labels .== 1.0)
    n_neg = length(labels) - n_pos
    (n_pos == 0 || n_neg == 0) && return NaN
    order    = sortperm(scores)          # ascending: lower score gets lower rank
    rank_sum = sum(i for (i, idx) in enumerate(order) if labels[idx] == 1.0)
    U        = rank_sum - n_pos * (n_pos + 1) / 2
    return 1.0 - U / (n_pos * n_neg)   # flip: we want P(b̂_corrupted < b̂_clean)
end

function experiment_corruption_robustness(;
    k::Int = 40,
    n::Int = 10_000,
    seed::Int = 123,
    deltas = 0.0:0.02:0.2,
    output_dir::AbstractString = joinpath(@__DIR__, "..", "..", "output", "synthetic"),
)
    mkpath(output_dir)
    Random.seed!(seed)

    # True skills: object i has skill i (ground truth order is 1 < 2 < ... < k)
    true_skills = float.(1:k)

    # Priors: N(k/2, (k/6)^2) for skills; Gamma(1,1) = ν=2 Student-t precision prior
    prior_s = Gaussian1DFromMeanVariance(k / 2.0, (k / 6.0)^2)
    prior_b = Gamma1D(0.0, 1.0)
    β²      = (k / 12.0)^2
    ϵ       = 0.0

    # Draw n base pairs (a, b) with a > b — i.e. a beats b under the true ranking
    base_pairs = Vector{Tuple{Int64,Int64}}(undef, n)
    for i in 1:n
        a, b = sample(1:k, 2, replace=false)
        base_pairs[i] = a > b ? (a, b) : (b, a)
    end


    # Results per corruption mode
    results = Dict{Symbol, NamedTuple}()
    scatter_delta = last(deltas)
    ranking_snapshots = Dict{Symbol, NamedTuple}()

    for corruption_mode in (:robust, :consistent)
        vanilla_taus    = Float64[]
        robust_taus     = Float64[]
        consistent_taus = Float64[]
        robust_b_aurocs    = Float64[]
        consistent_b_aurocs = Float64[]

        for δ in deltas
            matches         = copy(base_pairs)
            is_corrupted    = zeros(Float64, n)
            is_noisy_player = zeros(Float64, k)

            if corruption_mode == :robust
                n_corrupt = round(Int, δ * n)
                if n_corrupt > 0
                    for i in randperm(n)[1:n_corrupt]
                        matches[i]      = (matches[i][2], matches[i][1])
                        is_corrupted[i] = 1.0
                    end
                end
            else  # :consistent
                # sqrt(δ)/2 noisy players, each match flipped w.p. sqrt(δ)
                # Expected flipped fraction: 2·(sqrt(δ)/2)·sqrt(δ) = δ  ✓
                n_noisy   = round(Int, sqrt(δ) * k / 2)
                flip_prob = sqrt(δ)
                if n_noisy > 0
                    noisy_set = Set(randperm(k)[1:n_noisy])
                    for p in noisy_set; is_noisy_player[p] = 1.0; end
                    for i in 1:n
                        if (matches[i][1] ∈ noisy_set || matches[i][2] ∈ noisy_set) && rand() < flip_prob
                            matches[i]      = (matches[i][2], matches[i][1])
                            is_corrupted[i] = 1.0
                        end
                    end
                end
            end

            priors_s_vanilla = DistributionBag(prior_s, k)
            vanilla_s        = vanilla_trueskill(priors_s_vanilla, β², matches, ϵ)
            vanilla_means    = Gaussian.mean.(vanilla_s)

            priors_s_robust  = DistributionBag(prior_s, k)
            priors_b_robust  = DistributionBag(prior_b, n)
            robust_s, robust_b = robust_trueskill(priors_s_robust, priors_b_robust, matches, ϵ)
            robust_means     = Gaussian.mean.(robust_s)
            b_match_means    = Gamma.mean.(robust_b)

            priors_s_cons    = DistributionBag(prior_s, k)
            priors_b_cons    = DistributionBag(prior_b, k)
            consistent_s, consistent_b = consistent_trueskill(priors_s_cons, priors_b_cons, matches, ϵ)
            consistent_means  = Gaussian.mean.(consistent_s)
            b_player_means    = Gamma.mean.(consistent_b)

            if δ == scatter_delta
                player_match_counts = zeros(Float64, k)
                player_corrupt_counts = zeros(Float64, k)
                for i in 1:n
                    winner, loser = matches[i]
                    player_match_counts[winner] += 1
                    player_match_counts[loser] += 1
                    if is_corrupted[i] == 1.0
                        player_corrupt_counts[winner] += 1
                        player_corrupt_counts[loser] += 1
                    end
                end

                ranking_snapshots[corruption_mode] = (
                    vanilla_s=vanilla_s,
                    robust_s=robust_s,
                    consistent_s=consistent_s,
                    corruption_mode=corruption_mode,
                    δ=δ,
                    player_corruption_rates=[
                        player_match_counts[i] > 0 ? player_corrupt_counts[i] / player_match_counts[i] : 0.0
                        for i in 1:k
                    ],
                )
            end

            push!(vanilla_taus,    kendall_tau(true_skills, vanilla_means))
            push!(robust_taus,     kendall_tau(true_skills, robust_means))
            push!(consistent_taus, kendall_tau(true_skills, consistent_means))

            auc_r = _auroc_lower_is_positive(is_corrupted, b_match_means)
            auc_c = _auroc_lower_is_positive(is_noisy_player, b_player_means)
            push!(robust_b_aurocs,    auc_r)
            push!(consistent_b_aurocs, auc_c)

            @printf("[%s] δ=%5.1f%%  vanilla τ=%.3f  robust τ=%.3f  consistent τ=%.3f  AUROC(rob b̂)=%.3f  AUROC(con b̂)=%.3f\n",
                    corruption_mode, 100δ,
                    vanilla_taus[end], robust_taus[end], consistent_taus[end],
                    isnan(auc_r) ? 0.5 : auc_r, isnan(auc_c) ? 0.5 : auc_c)
        end

        results[corruption_mode] = (
            vanilla_taus=vanilla_taus, robust_taus=robust_taus, consistent_taus=consistent_taus,
            robust_b_aurocs=robust_b_aurocs, consistent_b_aurocs=consistent_b_aurocs,
        )
    end

    δ_pct = collect(deltas) .* 100
    δ_pct_tail = δ_pct[2:end]

    # Single row: [tau robust noise | tau consistent noise | combined AUROC]
    plt_tau_r = plot(δ_pct, [results[:robust].vanilla_taus results[:robust].robust_taus results[:robust].consistent_taus];
               label=["Vanilla" "Robust" "Consistent"], color=[:red :green :blue],
               ls=[:dot :dash :solid], lw=2.2,
               xlabel="% of corrupted matches", ylabel="Kendall's τ", title="(a)  match noise",
               tickfontsize=9, legendfontsize=10, legend=:bottomleft, foreground_color_legend = nothing, legend_marker_size = 20)

    plt_tau_c = plot(δ_pct, [results[:consistent].vanilla_taus results[:consistent].robust_taus results[:consistent].consistent_taus];
               label=:none, color=[:red :green :blue],
               ls=[:dot :dash :solid], lw=2.2,
               xlabel="% of corrupted matches", title="(b)  player noise",
               tickfontsize=9, legend=false)

    # Combined AUROC: green = robust model detecting matches under robust noise
    #                 blue  = consistent model detecting players under consistent noise
    plt_auc = plot(δ_pct_tail, [results[:robust].robust_b_aurocs[2:end] results[:consistent].consistent_b_aurocs[2:end]];
               label=:none, color=[:green :blue], ls=[:dash :solid], lw=2.2,
               xlabel="% of corrupted matches", ylabel="AUROC", title="(c)  outlier detection",
               ylims=(0.4, 1.05), tickfontsize=9, legend=false)
    hline!(plt_auc, [0.5]; color=:gray, ls=:solid, lw=1, label="")
    annotate!(plt_auc, δ_pct_tail[end], 0.52, text("chance", 11, :gray, :right))

    @assert haskey(ranking_snapshots, :robust) && haskey(ranking_snapshots, :consistent) "No ranking snapshots were captured for the scatter plots"
    true_ranks = collect(1:k)

    function ranking_scatter(snapshot::NamedTuple, posteriors_s::GaussianBag, model_name::String, color, ylabel)
        estimated_skills = Gaussian.mean.(posteriors_s)
        skill_stds = sqrt.(Gaussian.variance.(posteriors_s))
        marker_sizes = 3 .+ 6 .* snapshot.player_corruption_rates
        p = scatter(true_ranks, estimated_skills;
                    yerr=3 .* skill_stds,
                    lc=:black,
                    markerstrokecolor=:black,
                    label="",
                    markercolor=color,
                    markersize=marker_sizes,
                    markerstrokewidth=1,
                    alpha=1,
                    xlabel="true skill",
                    ylabel=ylabel,
                    title="$model_name",
                    tickfontsize=9)
        # plot!(p, [1, k], [1, k]; label="", color=:gray, ls=:dash, lw=1)
        return p
    end

    player_snapshot = ranking_snapshots[:consistent]
    plt_rank_player_v = ranking_scatter(player_snapshot, player_snapshot.vanilla_s, "(d)  vanilla", :red, "estimated skill")
    plt_rank_player_c = ranking_scatter(player_snapshot, player_snapshot.consistent_s, "(e)  consistent", :blue, nothing)

    plt = plot(plt_tau_r, plt_tau_c, plt_auc,
               plt_rank_player_v, plt_rank_player_c;
               layout=(1, 5), size=(1200, 300),
               left_margin=8Plots.mm, bottom_margin=10Plots.mm, top_margin=4Plots.mm)
    savefig(plt, joinpath(output_dir, "corruption.png"))
    savefig(plt, joinpath(output_dir, "corruption.pdf"))

    rows = DataFrame(mode=String[], corruption=Float64[], model=String[], kendall_tau=Float64[], detection_auroc=Float64[])
    for mode in (:robust, :consistent), (i, delta) in enumerate(deltas)
        r = results[mode]
        for (name, taus, aucs) in (("vanilla", r.vanilla_taus, fill(NaN, length(deltas))),
            ("robust", r.robust_taus, r.robust_b_aurocs), ("consistent", r.consistent_taus, r.consistent_b_aurocs))
            push!(rows, (String(mode), delta, name, taus[i], aucs[i]))
        end
    end
    CSV.write(joinpath(output_dir, "metrics.csv"), rows)
    return rows
end


end
if abspath(PROGRAM_FILE) == @__FILE__
    PaperSynthetic.experiment_corruption_robustness(n=1_000)
end
