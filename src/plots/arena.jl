include("capped_estimates.jl")

const PREFERENCE_LABELS=Dict(
    "gemini-2.5-pro"=>"Gemini 2.5 Pro", "gemini-2.5-flash"=>"Gemini 2.5 Flash",
    "o3-2025-04-16"=>"o3 (2025-04-16)", "chatgpt-4o-latest-20250326"=>"ChatGPT-4o latest (03-26)",
    "gpt-4.1-2025-04-14"=>"GPT-4.1 (04-14)", "claude-3-5-sonnet-20241022"=>"Claude Sonnet 3.5 (10-22)",
    "qwen3-235b-a22b-no-thinking"=>"Qwen3-235B (no thinking)", "mistral-medium-2505"=>"Mistral Medium (2505)",
    "gemma-3-27b-it"=>"Gemma 3 27B", "command-a-03-2025"=>"Command A (03-2025)",
    "claude-opus-4-20250514"=>"Claude Opus 4", "claude-sonnet-4-20250514"=>"Claude Sonnet 4",
    "claude-3-7-sonnet-20250219"=>"Claude Sonnet 3.7",
    "claude-3-7-sonnet-20250219-thinking-32k"=>"Claude Sonnet 3.7 (32k)")

function preference_panel_axis(lower, upper)
    padding = 0.04 * (upper - lower)
    limits = (lower - padding, upper + padding)
    ticks, _, _ = Plots.optimize_ticks(lower, upper; k_min=3, k_max=4, k_ideal=4)
    return (xlims=limits, xticks=filter(tick -> limits[1] <= tick <= limits[2], ticks))
end

function preference_panels(rows, path)
    t = sort(rows, :skill; rev=true)
    counts = hasproperty(t, :comparisons) ? t.comparisons : t.training_comparisons
    y = (1:nrow(t)) .+ 0.44
    labels = [get(PREFERENCE_LABELS, name, name) for name in t.model]
    @assert allunique(labels) && all(label -> !occursin('\n', label), labels)
    common = (; ylims=(0.4, nrow(t) + 1.08), yflip=true, legend=false,
        tickfontsize=9, guidefontsize=11, titlefontsize=11, titlelocation=:center,
        top_margin=3Plots.mm, bottom_margin=7Plots.mm)
    metadata = plot(; xlims=(0, 1), axis=false, grid=false, title="(a)  Language models",
        left_margin=2Plots.mm, right_margin=1Plots.mm, common...)
    annotate!(metadata, 0.01, 0.62, Plots.text("MODEL", 9, :left, "Helvetica Bold", :black))
    annotate!(metadata, 0.99, 0.62, Plots.text("VOTES", 9, :right, "Helvetica Bold", :black))
    plot!(metadata, [0.01, 0.99], [1.04, 1.04]; color=:lightgray, linewidth=0.7)
    for (i, yi) in enumerate(y)
        annotate!(metadata, 0.01, yi, Plots.text(labels[i], 10, :left))
        annotate!(metadata, 0.99, yi, Plots.text(string(counts[i]), 10, :right))
    end

    style = (; yticks=false, grid=:x, gridalpha=0.14, framestyle=:axes,
        left_margin=1Plots.mm, right_margin=3Plots.mm)
    skill_axis = preference_panel_axis(minimum(t.skill_low_3sd), maximum(t.skill_high_3sd))
    skill = plot(; title="(b)  Estimated skill", xlabel="Rating points", skill_axis..., style..., common...)
    noise_axis = preference_panel_axis(minimum(t.performance_lo), maximum(t.performance_hi))
    noise = plot(; title="(c)  Estimated inconsistency", xlabel="Rating points", noise_axis..., style..., common...)
    plot_capped_estimates!(skill, t.skill, y, t.skill_low_3sd, t.skill_high_3sd)
    plot_capped_estimates!(noise, t.performance_sd, y, t.performance_lo, t.performance_hi)
    p = plot(metadata, skill, noise; layout=grid(1, 3; widths=[0.45, 0.26, 0.29]),
             size=(920, 22nrow(t) + 95), background_color=:white, fontfamily="sans-serif")
    savefig(p, path * ".png")
    savefig(p, path * ".pdf")
    return p
end
