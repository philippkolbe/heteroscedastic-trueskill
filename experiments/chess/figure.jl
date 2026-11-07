# Regenerate the print figure without unrelated model fits.
include(joinpath(@__DIR__, "..", "..", "src", "datasets", "chess.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plots", "capped_estimates.jl"))

function chess_figure_rows(players, index, counts, stats, skills, precisions)
    rows = map(players) do name
        id = index[name]
        skill, precision = skills[id], precisions[id]
        posterior = Distributions.Gamma(TSGamma.shape(precision), 1 / TSGamma.rate(precision))
        descriptive = stats[name]
        (player=name, games=counts[name], mean_elo=descriptive.mean_elo,
         underdog_wins=descriptive.underdog_wins, favorite_losses=descriptive.favorite_losses,
         underdog_games=descriptive.underdog_games,
         comparable_games=descriptive.comparable_games,
         upset_win_pct=_pct(descriptive.underdog_wins, descriptive.comparable_games),
         upset_loss_pct=_pct(descriptive.favorite_losses, descriptive.comparable_games),
         legacy_upset_pct=descriptive.pct_upsets,
         skill_mean=TSG.mean(skill), skill_sd=sqrt(TSG.variance(skill)),
         performance_sd=sqrt(TSGamma.expected_inverse_precision(precision)),
         performance_lo=1 / sqrt(quantile(posterior, 0.975)),
         performance_hi=1 / sqrt(quantile(posterior, 0.025)))
    end
    return DataFrame(rows)
end

function chess_panel_axis(lower, upper)
    padding = 0.04 * (upper - lower)
    limits = (lower - padding, upper + padding)
    ticks, _, _ = Plots.optimize_ticks(lower, upper; k_min=3, k_max=4, k_ideal=4)
    return (xlims=limits, xticks=filter(tick -> limits[1] <= tick <= limits[2], ticks))
end

function print_chess_figure(rows, output_dir)
    y = 1:nrow(rows)
    common = (; ylims=(0.4, nrow(rows) + 1.08), yflip=true, legend=false,
        tickfontsize=9, guidefontsize=11, titlefontsize=11, titlelocation=:center,
        top_margin=3Plots.mm, bottom_margin=7Plots.mm)
    metadata = plot(; xlims=(0, 1), axis=false, grid=false,
        title="(a)  Blitz players", left_margin=2Plots.mm, right_margin=1Plots.mm, common...)
    columns = (0.01, 0.57, 0.75, 0.99)
    alignments = (:left, :right, :right, :right)
    for (x, label, alignment) in zip(columns, ("PLAYER", "GAMES", "ELO", "UPSET %"), alignments)
        annotate!(metadata, x, 0.62, Plots.text(label, 9, alignment, "Helvetica Bold", :black))
    end
    plot!(metadata, [0.01, 0.99], [1.04, 1.04]; color=:lightgray, linewidth=0.7)
    for (i, row) in enumerate(eachrow(rows))
        label = first(split(row.player, ','; limit=2))
        elo = isfinite(row.mean_elo) ? @sprintf("%.0f", row.mean_elo) : "NA"
        upset = isfinite(row.legacy_upset_pct) ? @sprintf("%.1f", row.legacy_upset_pct) : "NA"
        for (x, value, alignment) in zip(columns, (label, string(row.games), elo, upset), alignments)
            annotate!(metadata, x, i + 0.44, Plots.text(value, 10, alignment))
        end
    end

    style = (; yticks=false, grid=:x, gridalpha=0.14, framestyle=:axes,
        left_margin=1Plots.mm, right_margin=3Plots.mm)
    skill_errors = 3 .* rows.skill_sd
    skill_lower = rows.skill_mean .- skill_errors
    skill_upper = rows.skill_mean .+ skill_errors
    skill_axis = chess_panel_axis(minimum(skill_lower), maximum(skill_upper))
    skill = plot(;
        title="(b)  Estimated skill", xlabel="Rating points", skill_axis..., style..., common...)
    plot_capped_estimates!(skill, rows.skill_mean, y .+ 0.44, skill_lower, skill_upper)
    variability_axis = chess_panel_axis(min(minimum(rows.performance_lo), minimum(rows.performance_sd)),
                                        max(maximum(rows.performance_hi), maximum(rows.performance_sd)))
    variability = plot(; title="(c)  Estimated inconsistency", xlabel="Rating points",
        variability_axis..., style..., common...)
    plot_capped_estimates!(variability, rows.performance_sd, y .+ 0.44,
                           rows.performance_lo, rows.performance_hi)
    figure = plot(metadata, skill, variability; layout=grid(1, 3; widths=[0.45, 0.26, 0.29]),
                  size=(920, 335), background_color=:white, fontfamily="sans-serif")
    savefig(figure, joinpath(output_dir, "blitz.pdf"))
    savefig(figure, joinpath(output_dir, "blitz.png"))
    return figure
end

function regenerate_chess_figure(matches_path;years=2022:2024,
    output_dir=joinpath(@__DIR__, "..", "..", "output", "chess", "figure"))
    table=_load_matches(matches_path,years)
    isempty(table) && error("No valid chess matches in the requested years")
    names,index=_build_player_index(table)
    matches,draws,_=_matches_and_draw_mask(table,index)
    prior=TSG.Gaussian1DFromMeanVariance(1500.0,350.0^2)
    epsilon=_draw_margin(220.0,mean(draws))
    precision=_student_t_precision_prior(220.0,4.0,220.0^2)
    println("Fitting consistent model on ",length(matches)," games and ",length(names)," players")
    measured=@timed TS.consistent_trueskill(TSDC.DistributionBag(prior,length(names)),
        TSDC.DistributionBag(precision,length(names)),matches,epsilon;
        draw_mask=draws,inner_max_iters=100,outer_max_iters=100,diagnostics=true,verbose=true)
    skills,precisions,diagnostic=measured.value
    diagnostic.converged || error("Chess figure fit did not meet the outer residual tolerance")
    counts=_player_counts(table)
    stats=_player_period_stats(table)
    eligible = filter(names) do name
        counts[name] >= 20 && sqrt(TSG.variance(skills[index[name]])) <= 15
    end
    isempty(eligible) && error("No player meets the display criteria (at least 20 games and skill s.d. at most 15)")
    top = _top_consistent_skill_players(eligible, index, skills, 10)
    rows = chess_figure_rows(top, index, counts, stats, skills, precisions)
    mkpath(output_dir)
    figure = print_chess_figure(rows, output_dir)
    CSV.write(joinpath(output_dir, "figure_values.csv"), rows)
    CSV.write(joinpath(output_dir,"fit_diagnostics.csv"),DataFrame(n_games=[length(matches)],n_players=[length(names)],
        draw_margin=[epsilon],seconds=[measured.time],converged=[diagnostic.converged],
        iterations=[diagnostic.outer_iterations],residual=[diagnostic.final_residual]))
    return figure
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) == 1 || error("Usage: figure.jl MATCHES.csv")
    regenerate_chess_figure(only(ARGS))
end
