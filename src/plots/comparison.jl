#=Shared comparison plots for real-world TrueSkill experiments.=#
module ComparisonPlots

using Plots
using Statistics

"""Render skills, consistency, and optional rank-baseline checks on one ordering.

`skill_series` and `performance` contain only numeric means and error bars, so
this plotting helper is independent of the particular posterior type used by a
dataset. The caller supplies the common ordering and any dataset-specific row
labels.
"""
function model_comparison_plot(y, y_ticks, skill_series, performance, plot_path;
    skill_title::AbstractString="skills", performance_title::AbstractString="player performance",
    skill_xlabel::AbstractString="skill mean +/- 3 s.d.",
    performance_xlabel::AbstractString="performance s.d.",
    ylabel::AbstractString="player", baseline_series=NamedTuple[], row_height::Int=32)
    isempty(skill_series) && error("At least one skill series is required.")
    ylims = (0.5, length(y) + 1.5)
    first_series = first(skill_series)
    skill = scatter(first_series.means, y; xerr=first_series.errors, color=first_series.color,
        linecolor=first_series.color, markercolor=first_series.color, markerstrokecolor=first_series.color,
        label="", yticks=y_ticks, ylims=ylims, xlabel=skill_xlabel,
        ylabel=ylabel, title=skill_title, markersize=3, left_margin=95Plots.mm)
    for series in Iterators.drop(skill_series, 1)
        scatter!(skill, series.means, y; xerr=series.errors, color=series.color,
            linecolor=series.color, markercolor=series.color, markerstrokecolor=series.color,
            label="", markersize=3)
    end
    # GR scales scatter symbols disproportionately inside legends.  Separate
    # tiny line-and-marker handles keep the legend legible without shrinking
    # the data markers or their colored confidence intervals.
    for series in skill_series
        plot!(skill, [NaN], [NaN]; color=series.color, marker=:circle,
            markersize=1, markerstrokecolor=series.color, label=series.label, linewidth=1.5)
    end
    consistency = scatter(performance.means, y; xerr=performance.errors,
        color=performance.color, linecolor=performance.color, markercolor=performance.color,
        markerstrokecolor=performance.color, label="", yticks=(y, fill("", length(y))),
        ylims=ylims, xlabel=performance_xlabel, ylabel="", title=performance_title, markersize=3)
    plot!(consistency, [NaN], [NaN]; color=performance.color, marker=:circle,
        markersize=1, markerstrokecolor=performance.color, label=performance.label, linewidth=1.5)
    panels = Any[skill, consistency]
    for series in baseline_series
        values = Float64.(series.values)
        scale = std(values)
        standardized = scale > 0 ? (values .- mean(values)) ./ scale : zeros(length(values))
        push!(panels, scatter(standardized, y; yticks=(y, fill("", length(y))), ylims=ylims,
            xlabel="standardized score", ylabel="", title=series.title, label="", markersize=3))
    end
    figure = plot(panels...; layout=(1, length(panels)),
        size=(950 * length(panels), max(650, row_height * length(y) + 120)),
        margin=8Plots.mm,
        thickness_scaling=row_height <= 24 ? 0.65 : 1.0)
    savefig(figure, plot_path)
    savefig(figure, replace(plot_path, r"\.[^.]+$" => ".pdf"))
    return figure
end

end
