# Plot interval caps independently of the black whiskers and marker outlines.
function plot_capped_estimates!(panel, estimates, y, lower, upper)
    midpoints = (lower .+ upper) ./ 2
    halfwidths = (upper .- lower) ./ 2
    plot!(panel, midpoints, y; seriestype=:xerror, xerr=halfwidths,
          linecolor=:black, linewidth=1, markercolor=:blue,
          markerstrokecolor=:blue, markerstrokewidth=1, markersize=4)
    scatter!(panel, estimates, y; markercolor=:blue,
             markerstrokecolor=:black, markerstrokewidth=1, markersize=4)
    return panel
end
