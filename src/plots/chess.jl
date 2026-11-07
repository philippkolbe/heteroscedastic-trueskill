# Diagnostic plot for the general chess fit.
function _consistent_skill_and_noise_plot(top_players::Vector{String},
    player_to_index::Dict{String,Int}, player_counts::Dict{String,Int}, player_elo_stats,
    consistent_s, consistent_b, plot_path::AbstractString)
    players=sort(top_players;by=name->TSG.mean(consistent_s[player_to_index[name]]),rev=true)
    ids=[player_to_index[name] for name in players]
    y=collect(1:length(players))
    labels=["$(replace(name,","=>", "))  (n=$(get(player_counts,name,0)))" for name in players]
    means=[TSG.mean(consistent_s[i]) for i in ids]
    skill_errors=[3sqrt(TSG.variance(consistent_s[i])) for i in ids]
    performance=[sqrt(TSGamma.expected_inverse_precision(consistent_b[i])) for i in ids]
    lower=Float64[];upper=Float64[]
    for id in ids
        b=consistent_b[id]
        distribution=Distributions.Gamma(TSGamma.shape(b),1/TSGamma.rate(b))
        push!(lower,sqrt(1/quantile(distribution,.975)))
        push!(upper,sqrt(1/quantile(distribution,.025)))
    end
    common=(;ylims=(.4,length(players)+.6),yflip=true,legend=false,
        grid=:x,gridalpha=.15,framestyle=:box,markersize=4,markerstrokewidth=0,
        color=:steelblue,linewidth=1.5,tickfontsize=10,guidefontsize=11,titlefontsize=12,
        top_margin=5Plots.mm,bottom_margin=9Plots.mm)
    skill=scatter(means,y;xerr=skill_errors,yticks=(y,labels),
        title="Estimated skill",xlabel="Mean ± 3 posterior standard deviations",
        left_margin=3Plots.mm,right_margin=4Plots.mm,common...)
    variability=scatter(performance,y;xerr=(max.(performance.-lower,0.0),max.(upper.-performance,0.0)),
        yticks=(y,fill("",length(y))),title="Performance variability",
        xlabel="Performance s.d.\n95% approximate posterior interval",
        xlims=(0.0,1.07maximum(upper)),left_margin=2Plots.mm,right_margin=5Plots.mm,common...)
    figure=plot(skill,variability;layout=grid(1,2;widths=[.62,.38]),
        size=(1400,max(440,36length(players)+100)))
    mkpath(dirname(plot_path))
    savefig(figure,plot_path)
    savefig(figure,replace(plot_path,r"\.[^.]+$"=>".pdf"))
    return figure
end
