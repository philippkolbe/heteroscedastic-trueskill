isdefined(@__MODULE__, :TrueSkill) || include("../trueskill.jl")

using CSV
using DataFrames
using Distributions
using Plots
using Printf
using Statistics
using SpecialFunctions: loggamma

const TS = TrueSkill
const TSG = TrueSkill.Gaussian
const TSGamma = TrueSkill.Gamma
const TSDC = TrueSkill.DistributionCollections

const DEFAULT_MATCHES_PATH = joinpath(@__DIR__, "..", "..", "data", "chess", "blitz", "chess_matches.csv")
const DEFAULT_PLAYERS_PATH = joinpath(@__DIR__, "..", "..", "data", "chess", "blitz", "chess_players.csv")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "output", "chess")

function _truthy(x)::Bool
    x isa Bool && return x
    lowered = lowercase(strip(string(coalesce(x, ""))))
    return lowered in ("true", "t", "1", "yes", "y")
end

function _load_matches(matches_path::AbstractString, years)::DataFrame
    keep_years = Set(Int.(collect(years)))
    columns = [:match_id, :year, :date, :event, :white_name, :black_name, :white_elo, :black_elo,
               :result, :winner_name, :is_draw, :eco]
    available = propertynames(CSV.File(matches_path; limit=0))
    select_columns = [column for column in columns if column in available]
    required = [:year, :white_name, :black_name, :winner_name, :is_draw]
    missing_required = setdiff(required, select_columns)
    isempty(missing_required) || error("Missing required columns in $matches_path: $missing_required")

    df = DataFrame(CSV.File(
        matches_path;
        select=select_columns,
        missingstring=["", "NA", "N/A"],
        types=Dict(:year => Int),
    ))

    filter!(row -> row.year in keep_years, df)
    filter!(df) do row
        white = coalesce(row.white_name, "")
        black = coalesce(row.black_name, "")
        white != "" || return false
        black != "" || return false
        white != black || return false
        _truthy(row.is_draw) && return true
        winner = coalesce(row.winner_name, "")
        return winner == white || winner == black
    end

    sort_columns = [:year]
    :date in propertynames(df) && push!(sort_columns, :date)
    :match_id in propertynames(df) && push!(sort_columns, :match_id)
    sort!(df, sort_columns)
    return df
end

function _load_player_elos(players_path::AbstractString)::Dict{String, Float64}
    isfile(players_path) || error("Could not find players CSV: $players_path")
    players = DataFrame(CSV.File(players_path; select=[:name, :max_elo], missingstring=["", "NA", "N/A"]))
    elos = Dict{String, Float64}()
    for row in eachrow(players)
        ismissing(row.name) && continue
        ismissing(row.max_elo) && continue
        elos[String(row.name)] = Float64(row.max_elo)
    end
    return elos
end

function _player_counts(df::DataFrame)::Dict{String, Int}
    counts = Dict{String, Int}()
    for row in eachrow(df)
        white = String(row.white_name)
        black = String(row.black_name)
        counts[white] = get(counts, white, 0) + 1
        counts[black] = get(counts, black, 0) + 1
    end
    return counts
end

function _maybe_float(x)
    ismissing(x) && return nothing
    parsed = tryparse(Float64, string(x))
    return parsed
end

function _pct(numerator::Int, denominator::Int)::Float64
    denominator == 0 && return NaN
    return 100.0 * numerator / denominator
end

function _player_period_stats(df::DataFrame)
    values = Dict{String, Vector{Float64}}()
    has_white_elo = :white_elo in propertynames(df)
    has_black_elo = :black_elo in propertynames(df)
    upset_games = Dict{String, Int}()
    comparable_games = Dict{String, Int}()
    underdog_games = Dict{String, Int}()
    underdog_wins = Dict{String, Int}()

    for row in eachrow(df)
        white = String(row.white_name)
        black = String(row.black_name)
        haskey(values, white) || (values[white] = Float64[])
        haskey(values, black) || (values[black] = Float64[])
        haskey(upset_games, white) || (upset_games[white] = 0)
        haskey(upset_games, black) || (upset_games[black] = 0)
        haskey(comparable_games, white) || (comparable_games[white] = 0)
        haskey(comparable_games, black) || (comparable_games[black] = 0)

        white_elo = nothing
        black_elo = nothing
        if has_white_elo
            white_elo = _maybe_float(row.white_elo)
            white_elo === nothing || push!(values[white], white_elo)
        end
        if has_black_elo
            black_elo = _maybe_float(row.black_elo)
            black_elo === nothing || push!(values[black], black_elo)
        end

        if white_elo !== nothing && black_elo !== nothing && white_elo != black_elo && !_truthy(row.is_draw)
            winner = String(coalesce(row.winner_name, ""))
            winner == white || winner == black || continue
            comparable_games[white] += 1
            comparable_games[black] += 1
            underdog = white_elo < black_elo ? white : black
            underdog_games[underdog] = get(underdog_games, underdog, 0) + 1
            if winner == underdog
                upset_games[white] += 1
                upset_games[black] += 1
                underdog_wins[underdog] = get(underdog_wins, underdog, 0) + 1
            end
        end
    end

    stats = Dict{String, NamedTuple{(:mean_elo, :pct_upsets, :underdog_wins, :favorite_losses,
                                     :underdog_games, :comparable_games),
                                    Tuple{Float64, Float64, Int, Int, Int, Int}}}()
    for (player, elos) in values
        wins = get(underdog_wins, player, 0)
        stats[player] = (
            mean_elo=isempty(elos) ? NaN : mean(elos),
            pct_upsets=_pct(get(upset_games, player, 0), get(comparable_games, player, 0)),
            underdog_wins=wins,
            favorite_losses=upset_games[player] - wins,
            underdog_games=get(underdog_games, player, 0),
            comparable_games=get(comparable_games, player, 0),
        )
    end
    return stats
end

function _build_player_index(df::DataFrame)::Tuple{Vector{String}, Dict{String, Int}}
    names = sort!(collect(keys(_player_counts(df))))
    index = Dict(name => i for (i, name) in enumerate(names))
    return names, index
end

function _matches_and_draw_mask(df::DataFrame, player_to_index::Dict{String, Int})
    matches = Tuple{Int64, Int64}[]
    draw_mask = BitVector()
    player_match_indices = [Int[] for _ in 1:length(player_to_index)]

    for row in eachrow(df)
        white = String(row.white_name)
        black = String(row.black_name)
        white_id = player_to_index[white]
        black_id = player_to_index[black]

        if _truthy(row.is_draw)
            push!(matches, (white_id, black_id))
            push!(draw_mask, true)
        else
            winner = String(row.winner_name)
            push!(matches, winner == white ? (white_id, black_id) : (black_id, white_id))
            push!(draw_mask, false)
        end

        match_index = length(matches)
        push!(player_match_indices[white_id], match_index)
        push!(player_match_indices[black_id], match_index)
    end

    return matches, draw_mask, player_match_indices
end

function _student_t_precision_prior(beta::Float64, nu::Float64, target_variance::Float64)
    beta_param = max(nu / 2 - 1, 1e-6)
    return TSGamma.Gamma1D(beta_param, target_variance * beta_param)
end

function _draw_margin(beta::Float64, draw_rate::Float64)::Float64
    clipped = clamp(draw_rate, 1e-6, 1 - 1e-6)
    return sqrt(2.0) * beta * quantile(Normal(), (clipped + 1.0) / 2.0)
end

function _top_consistent_skill_players(
    player_names::Vector{String},
    player_to_index::Dict{String, Int},
    consistent_s,
    n::Int,
)::Vector{String}
    candidates = copy(player_names)
    sort!(candidates; by=name -> (-TSG.mean(consistent_s[player_to_index[name]]), name))
    length(candidates) > n && return candidates[1:n]
    return candidates
end

function _top_consistent_skill_report(
    top_players::Vector{String},
    player_to_index::Dict{String, Int},
    player_counts::Dict{String, Int},
    player_elo_stats,
    player_match_indices::Vector{Vector{Int}},
    max_elos::Dict{String, Float64},
    vanilla_s,
    robust_s,
    robust_b,
    consistent_s,
    consistent_b,
    beta::Float64,
)::DataFrame
    robust_precision_means = TSGamma.mean.(robust_b)
    rows = DataFrame(
        consistent_skill_rank=Int[],
        player=String[],
        max_elo=Float64[],
        n_games=Int[],
        period_mean_elo=Float64[],
        var"%upsets"=Float64[],
        vanilla_skill=Float64[],
        vanilla_mean_precision=Float64[],
        robust_skill=Float64[],
        robust_mean_precision=Float64[],
        consistent_skill=Float64[],
        consistent_mean_precision=Float64[],
        consistent_expected_inverse_precision=Float64[],
        consistent_inverse_precision_sd=Float64[],
    )

    for (rank, player) in enumerate(top_players)
        player_id = player_to_index[player]
        robust_indices = player_match_indices[player_id]
        robust_mean_precision = isempty(robust_indices) ? NaN : mean(robust_precision_means[robust_indices])
        elo_stats = get(player_elo_stats, player, (mean_elo=NaN, pct_upsets=NaN))
        push!(rows, (
            consistent_skill_rank=rank,
            player=player,
            max_elo=get(max_elos, player, NaN),
            n_games=get(player_counts, player, 0),
            period_mean_elo=elo_stats.mean_elo,
            var"%upsets"=elo_stats.pct_upsets,
            vanilla_skill=TSG.mean(vanilla_s[player_id]),
            vanilla_mean_precision=1.0 / beta^2,
            robust_skill=TSG.mean(robust_s[player_id]),
            robust_mean_precision=robust_mean_precision,
            consistent_skill=TSG.mean(consistent_s[player_id]),
            consistent_mean_precision=TSGamma.mean(consistent_b[player_id]),
            consistent_expected_inverse_precision=TSGamma.expected_inverse_precision(consistent_b[player_id]),
            consistent_inverse_precision_sd=sqrt(_inverse_precision_variance(consistent_b[player_id])),
        ))
    end

    return rows
end

function _player_axis_label(player::String, player_counts::Dict{String, Int}, player_elo_stats)::String
    elo_stats = get(player_elo_stats, player, (mean_elo=NaN, pct_upsets=NaN))
    elo_label = isfinite(elo_stats.mean_elo) ? @sprintf("%.0f", elo_stats.mean_elo) : "NA"
    upset_label = isfinite(elo_stats.pct_upsets) ? @sprintf("%.1f", elo_stats.pct_upsets) : "NA"
    return @sprintf("%-28s %6d %8s %9s", player, get(player_counts, player, 0), elo_label, upset_label)
end

function _player_axis_ticks(players::Vector{String}, y::Vector{Int}, player_counts::Dict{String, Int}, player_elo_stats)
    labels = [_player_axis_label(player, player_counts, player_elo_stats) for player in players]
    return ([y; maximum(y) + 1], [labels; @sprintf("%-28s %6s %8s %9s", "name", "#games", "avg elo", "%upsets")])
end

function _elo_summary_label(player::String, player_elo_stats)::String
    elo_stats = get(player_elo_stats, player, (mean_elo=NaN, pct_upsets=NaN))
    return isfinite(elo_stats.mean_elo) ? @sprintf("%.0f", elo_stats.mean_elo) : "NA"
end

function _pct_label(x::Float64)::String
    return isfinite(x) ? @sprintf("%.1f", x) : "NA"
end

function _player_label_panel(players::Vector{String}, y::Vector{Int}, player_counts::Dict{String, Int}, player_elo_stats)
    p = plot(
        xlims=(0.0, 1.0),
        ylims=(0.5, length(players) + 1.5),
        axis=false,
        grid=false,
        legend=false,
        title="",
    )

    header_y = maximum(y) + 1
    annotate!(p, 0.00, header_y, text("name", 8, :left))
    annotate!(p, 0.6, header_y, text("#games", 8, :right))
    annotate!(p, 0.75, header_y, text("avg elo", 8, :right))
    annotate!(p, 0.9, header_y, text("%upsets", 8, :right))

    for (yi, player) in zip(y, players)
        elo_stats = get(player_elo_stats, player, (mean_elo=NaN, pct_upsets=NaN))
        annotate!(p, 0.00, yi, text(player, 8, :left))
        annotate!(p, 0.6, yi, text(string(get(player_counts, player, 0)), 8, :right))
        annotate!(p, 0.75, yi, text(_elo_summary_label(player, player_elo_stats), 8, :right))
        annotate!(p, 0.9, yi, text(_pct_label(elo_stats.pct_upsets), 8, :right))
    end
    return p
end

function _inverse_precision_variance(g::TSGamma.Gamma1D)::Float64
    if !isfinite(g.β) || !isfinite(g.λ) || g.λ <= 0.0 || g.β <= 1.0
        return NaN
    end
    return g.λ^2 / (g.β^2 * (g.β - 1.0))
end

function _top_consistent_skill_model_comparison_plot(
    top_players::Vector{String},
    player_to_index::Dict{String, Int},
    player_counts::Dict{String, Int},
    player_elo_stats,
    vanilla_s,
    robust_s,
    consistent_s,
    consistent_b,
    plot_path::AbstractString,
)
    player_ids = [player_to_index[player] for player in top_players]

    function ordered_skill_panel(posteriors_s, title)
        ordered = sort(
            [(player=top_players[i], id=player_ids[i], skill=TSG.mean(posteriors_s[player_ids[i]]))
             for i in eachindex(top_players)];
            by=row -> row.skill,
        )
        y = collect(1:length(ordered))
        ordered_players = [row.player for row in ordered]
        ordered_ids = [row.id for row in ordered]
        skill_means = [TSG.mean(posteriors_s[i]) for i in ordered_ids]
        skill_error = [3.0 * sqrt(TSG.variance(posteriors_s[i])) for i in ordered_ids]
        return scatter(
            skill_means,
            y;
            xerr=skill_error,
            yticks=_player_axis_ticks(ordered_players, y, player_counts, player_elo_stats),
            ylims=(0.5, length(ordered) + 1.5),
            xlabel="Skill",
            ylabel="Player",
            title=title,
            label="",
            markersize=5,
        )
    end

    vanilla_plot = ordered_skill_panel(vanilla_s, "Vanilla skill")
    robust_plot = ordered_skill_panel(robust_s, "Robust skill")

    consistent_order = sort(
        [(player=top_players[i], id=player_ids[i], skill=TSG.mean(consistent_s[player_ids[i]]))
         for i in eachindex(top_players)];
        by=row -> row.skill,
    )
    y = collect(1:length(consistent_order))
    consistent_players = [row.player for row in consistent_order]
    consistent_ids = [row.id for row in consistent_order]
    consistent_skill_means = [TSG.mean(consistent_s[i]) for i in consistent_ids]
    consistent_skill_error = [3.0 * sqrt(TSG.variance(consistent_s[i])) for i in consistent_ids]
    consistent_plot = scatter(
        consistent_skill_means,
        y;
        xerr=consistent_skill_error,
        yticks=_player_axis_ticks(consistent_players, y, player_counts, player_elo_stats),
        ylims=(0.5, length(consistent_order) + 1.5),
        xlabel="estimated skill +/- 3σ",
        ylabel="Player",
        title="consistent model skill",
        label="",
        markersize=5,
    )

    inv_precision_means = [sqrt(TSGamma.expected_inverse_precision(consistent_b[i])) for i in consistent_ids]
    inv_precision_error = [3.0 * sqrt(_inverse_precision_variance(consistent_b[i])) for i in consistent_ids]

    inverse_precision_plot = scatter(
        inv_precision_means,
        y;
        xerr=inv_precision_error,
        yticks=_player_axis_ticks(consistent_players, y, player_counts, player_elo_stats),
        ylims=(0.5, length(consistent_order) + 1.5),
        xlabel="estimated performance s.d. on rating scale",
        ylabel="player",
        title="consistent model performance",
        label="",
        markersize=5,
    )

    plt = plot(
        vanilla_plot,
        robust_plot,
        consistent_plot,
        inverse_precision_plot;
        layout=(1, 4),
        size=(2400, 300),
        margin=8 * Plots.mm,
        left_margin=18 * Plots.mm,
    )
    return plt
end

# If B ~ Gamma(shape, rate), Var(B^(-1/2)) = E[B^-1] - E[B^(-1/2)]^2.
# The point estimate remains sqrt(E[B^-1]), as in the paper.
function _performance_sd_uncertainty(g::TSGamma.Gamma1D)
    a, rate = TSGamma.shape(g), TSGamma.rate(g)
    a > 1 || return Inf
    mean_sd = sqrt(rate) * exp(loggamma(a-0.5) - loggamma(a))
    return sqrt(max(0.0, TSGamma.expected_inverse_precision(g) - mean_sd^2))
end

include("../plots/chess.jl")

function _value_or_empty(x)::String
    ismissing(x) && return ""
    return string(x)
end

function _value_or_nan(x)::Float64
    ismissing(x) && return NaN
    parsed = tryparse(Float64, string(x))
    parsed === nothing && return NaN
    return parsed
end

function _noisiest_robust_matches(matches_df::DataFrame, robust_b; n::Int=10)::DataFrame
    rows = DataFrame(
        match_index=Int[],
        match_id=String[],
        year=Int[],
        date=String[],
        event=String[],
        white_name=String[],
        black_name=String[],
        white_elo=Float64[],
        black_elo=Float64[],
        result=String[],
        winner_name=String[],
        is_draw=Bool[],
        eco=String[],
        robust_mean_precision=Float64[],
        robust_expected_noise_variance=Float64[],
    )

    for i in 1:min(nrow(matches_df), length(robust_b))
        row = matches_df[i, :]
        push!(rows, (
            match_index=i,
            match_id=:match_id in propertynames(matches_df) ? _value_or_empty(row.match_id) : "",
            year=Int(row.year),
            date=:date in propertynames(matches_df) ? _value_or_empty(row.date) : "",
            event=:event in propertynames(matches_df) ? _value_or_empty(row.event) : "",
            white_name=_value_or_empty(row.white_name),
            black_name=_value_or_empty(row.black_name),
            white_elo=:white_elo in propertynames(matches_df) ? _value_or_nan(row.white_elo) : NaN,
            black_elo=:black_elo in propertynames(matches_df) ? _value_or_nan(row.black_elo) : NaN,
            result=:result in propertynames(matches_df) ? _value_or_empty(row.result) : "",
            winner_name=:winner_name in propertynames(matches_df) ? _value_or_empty(row.winner_name) : "",
            is_draw=_truthy(row.is_draw),
            eco=:eco in propertynames(matches_df) ? _value_or_empty(row.eco) : "",
            robust_mean_precision=TSGamma.mean(robust_b[i]),
            robust_expected_noise_variance=TSGamma.expected_inverse_precision(robust_b[i]),
        ))
    end

    sort!(rows, :robust_mean_precision)
    return first(rows, min(n, nrow(rows)))
end

_mib(bytes::Real)::Float64 = bytes / 1024.0^2
_maxrss_bytes()::Int64 = Int64(Sys.maxrss())

function run_chess_experiment(;
    matches_path::AbstractString=DEFAULT_MATCHES_PATH,
    players_path::AbstractString=DEFAULT_PLAYERS_PATH,
    output_dir::AbstractString=DEFAULT_OUTPUT_DIR,
    years=2022:2024,
    mu::Float64=1500.0,
    sigma::Float64=350.0,
    beta::Float64=220.0,
    nu::Float64=4.0,
    top_n::Int=10,
    min_games_for_display::Int=20,
    draw_margin::Union{Nothing, Float64}=nothing,
    outer_max_iters::Int=100,
    inner_max_iters::Int=100,
    verbose::Bool=true,
)::NamedTuple
    isfile(matches_path) || error("Could not find matches CSV: $matches_path")
    mkpath(output_dir)

    matches_df = _load_matches(matches_path, years)
    isempty(matches_df) && error("No valid blitz chess matches found for years $(collect(years))")

    max_elos = _load_player_elos(players_path)
    player_names, player_to_index = _build_player_index(matches_df)
    player_counts = _player_counts(matches_df)
    player_elo_stats = _player_period_stats(matches_df)
    matches, draw_mask, player_match_indices = _matches_and_draw_mask(matches_df, player_to_index)

    n_players = length(player_names)
    n_matches = length(matches)
    empirical_draw_rate = mean(draw_mask)
    epsilon = draw_margin === nothing ? _draw_margin(beta, empirical_draw_rate) : draw_margin

    prior_s = TSG.Gaussian1DFromMeanVariance(mu, sigma^2)
    robust_prior_b = _student_t_precision_prior(beta, nu, 2.0 * beta^2)
    consistent_prior_b = _student_t_precision_prior(beta, nu, beta^2)

    verbose && begin
        println("Blitz chess TrueSkill experiment")
        println("  matches: $matches_path")
        println("  players: $players_path")
        println("  years: $(join(sort!(collect(Set(Int.(collect(years))))), ","))")
        println("  observed players: $n_players")
        println("  matches: $n_matches")
        @printf("  mu=%.1f sigma=%.1f beta=%.1f nu=%.1f draw_margin=%.3f draw_rate=%.3f\n",
                mu, sigma, beta, nu, epsilon, empirical_draw_rate)
    end

    verbose && println("\nRunning vanilla TrueSkill")
    total_start = time()
    peak_rss_at_start_bytes = _maxrss_bytes()
    vanilla_timed = @timed TS.vanilla_trueskill(
        TSDC.DistributionBag(prior_s, n_players),
        beta^2,
        matches,
        epsilon;
        draw_mask=draw_mask,
        outer_max_iters=outer_max_iters,
        verbose=verbose,
    )
    vanilla_s = vanilla_timed.value
    vanilla_runtime_seconds = vanilla_timed.time
    vanilla_allocated_bytes = vanilla_timed.bytes
    vanilla_peak_rss_bytes = _maxrss_bytes()

    verbose && println("\nRunning robust TrueSkill")
    robust_timed = @timed TS.robust_trueskill(
        TSDC.DistributionBag(prior_s, n_players),
        TSDC.DistributionBag(robust_prior_b, n_matches),
        matches,
        epsilon;
        draw_mask=draw_mask,
        inner_max_iters=inner_max_iters,
        outer_max_iters=outer_max_iters,
        verbose=verbose,
    )
    robust_s, robust_b = robust_timed.value
    robust_runtime_seconds = robust_timed.time
    robust_allocated_bytes = robust_timed.bytes
    robust_peak_rss_bytes = _maxrss_bytes()

    verbose && println("\nRunning consistent TrueSkill")
    consistent_timed = @timed TS.consistent_trueskill(
        TSDC.DistributionBag(prior_s, n_players),
        TSDC.DistributionBag(consistent_prior_b, n_players),
        matches,
        epsilon;
        draw_mask=draw_mask,
        inner_max_iters=inner_max_iters,
        outer_max_iters=outer_max_iters,
        verbose=verbose,
    )
    consistent_s, consistent_b = consistent_timed.value
    consistent_runtime_seconds = consistent_timed.time
    consistent_allocated_bytes = consistent_timed.bytes
    consistent_peak_rss_bytes = _maxrss_bytes()

    eligible_players = filter(name -> get(player_counts,name,0) >= min_games_for_display, player_names)
    top_players = _top_consistent_skill_players(eligible_players, player_to_index, consistent_s, top_n)
    report = _top_consistent_skill_report(
        top_players,
        player_to_index,
        player_counts,
        player_elo_stats,
        player_match_indices,
        max_elos,
        vanilla_s,
        robust_s,
        robust_b,
        consistent_s,
        consistent_b,
        beta,
    )
    noisiest_matches = _noisiest_robust_matches(matches_df, robust_b; n=10)
    total_runtime_seconds = time() - total_start

    metadata = DataFrame([
        (
            matches_path=matches_path,
            players_path=players_path,
            years=join(sort!(collect(Set(Int.(collect(years))))), ","),
            n_players=n_players,
            n_matches=n_matches,
            n_draws=count(draw_mask),
            mu=mu,
            sigma=sigma,
            beta=beta,
            nu=nu,
            draw_margin=epsilon,
            empirical_draw_rate=empirical_draw_rate,
            robust_prior_mean_precision=TSGamma.mean(robust_prior_b),
            consistent_prior_mean_precision=TSGamma.mean(consistent_prior_b),
            vanilla_runtime_seconds=vanilla_runtime_seconds,
            robust_runtime_seconds=robust_runtime_seconds,
            consistent_runtime_seconds=consistent_runtime_seconds,
            total_runtime_seconds=total_runtime_seconds,
            peak_rss_at_start_bytes=peak_rss_at_start_bytes,
            vanilla_allocated_bytes=vanilla_allocated_bytes,
            robust_allocated_bytes=robust_allocated_bytes,
            consistent_allocated_bytes=consistent_allocated_bytes,
            vanilla_peak_rss_bytes=vanilla_peak_rss_bytes,
            robust_peak_rss_bytes=robust_peak_rss_bytes,
            consistent_peak_rss_bytes=consistent_peak_rss_bytes,
            model_fit_peak_rss_bytes=maximum((vanilla_peak_rss_bytes, robust_peak_rss_bytes, consistent_peak_rss_bytes)),
        )
    ])

    report_path = joinpath(output_dir, "top$(top_n)_consistent_skill_player_report.csv")
    noisiest_matches_path = joinpath(output_dir, "robust_noisiest_matches.csv")
    metadata_path = joinpath(output_dir, "metadata.csv")
    plot_path = joinpath(output_dir, "top$(top_n)_consistent_skill_model_comparison.png")
    consistent_plot_path = joinpath(output_dir, "top$(top_n)_consistent_skill_and_noise.png")
    CSV.write(report_path, report)
    CSV.write(noisiest_matches_path, noisiest_matches)
    CSV.write(metadata_path, metadata)
    model_comparison_plot = _top_consistent_skill_model_comparison_plot(top_players, player_to_index, player_counts, player_elo_stats, vanilla_s, robust_s, consistent_s, consistent_b, plot_path)
    consistent_plot = _consistent_skill_and_noise_plot(top_players, player_to_index, player_counts, player_elo_stats, consistent_s, consistent_b, consistent_plot_path)

    verbose && begin
        println("\nPlayers with top $(length(top_players)) consistent skill")
        show(report; allrows=true, allcols=true)
        println("\n\n10 matches with lowest robust inferred precision")
        show(noisiest_matches; allrows=true, allcols=true)
        @printf("\n\nRuntime: vanilla %.2fs, robust %.2fs, consistent %.2fs, total %.2fs\n",
                vanilla_runtime_seconds, robust_runtime_seconds, consistent_runtime_seconds, total_runtime_seconds)
        @printf("Allocated: vanilla %.1f MiB, robust %.1f MiB, consistent %.1f MiB\n",
                _mib(vanilla_allocated_bytes), _mib(robust_allocated_bytes), _mib(consistent_allocated_bytes))
        @printf("Peak RSS after fits: vanilla %.1f MiB, robust %.1f MiB, consistent %.1f MiB\n",
                _mib(vanilla_peak_rss_bytes), _mib(robust_peak_rss_bytes), _mib(consistent_peak_rss_bytes))
        println("\n\nWrote:")
        println("  $report_path")
        println("  $noisiest_matches_path")
        println("  $metadata_path")
        println("  $plot_path")
        println("  $consistent_plot_path")
    end

    return (
        report=report,
        noisiest_matches=noisiest_matches,
        metadata=metadata,
        runtimes=(
            vanilla_seconds=vanilla_runtime_seconds,
            robust_seconds=robust_runtime_seconds,
            consistent_seconds=consistent_runtime_seconds,
            total_seconds=total_runtime_seconds,
        ),
        memory=(
            peak_rss_at_start_bytes=peak_rss_at_start_bytes,
            vanilla_allocated_bytes=vanilla_allocated_bytes,
            robust_allocated_bytes=robust_allocated_bytes,
            consistent_allocated_bytes=consistent_allocated_bytes,
            vanilla_peak_rss_bytes=vanilla_peak_rss_bytes,
            robust_peak_rss_bytes=robust_peak_rss_bytes,
            consistent_peak_rss_bytes=consistent_peak_rss_bytes,
            model_fit_peak_rss_bytes=maximum((vanilla_peak_rss_bytes, robust_peak_rss_bytes, consistent_peak_rss_bytes)),
        ),
        model_comparison_plot=model_comparison_plot,
        consistent_plot=consistent_plot,
        vanilla_s=vanilla_s,
        robust_s=robust_s,
        robust_b=robust_b,
        consistent_s=consistent_s,
        consistent_b=consistent_b,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_chess_experiment()
end
