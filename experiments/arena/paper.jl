# Shared preference fitting and archived LMArena evaluation.
include(joinpath(@__DIR__, "..", "..", "src", "datasets", "arena.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plots", "arena.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "evaluation", "baselines.jl"))
using SHA

const PAPER_SETTINGS = (
    human_preference_140k=(vanilla=(24.0, NaN), robust=(24.0, 10.0), consistent=(40.0, 2.5)),
    human_preference_55k=(vanilla=(40.0, NaN), robust=(40.0, 6.0), consistent=(40.0, 32.0)),
)

function paper_55k(path)
    isfile(path) || error("Supply the pinned Arena 55k train.csv")
    db = DuckDB.DB()
    table = try
        escaped = replace(path, '\\'=>'/', "'"=>"''")
        first_prompt = raw"json_extract_string(prompt, '$[0]')"
        DataFrame(DuckDB.DBInterface.execute(db,
            "SELECT id, model_a, model_b, winner_model_a, winner_model_b, winner_tie,
             $first_prompt AS first_prompt FROM read_csv_auto('$escaped')"))
    finally
        DuckDB.close(db)
    end
    @assert allunique(table.id)
    for row in eachrow(table)
        labels = (row.winner_model_a, row.winner_model_b, row.winner_tie)
        @assert all(x -> !ismissing(x) && x in (0,1), labels) && sum(labels) == 1
    end
    filter!(row -> !ismissing(row.model_a) && !ismissing(row.model_b) && row.model_a != row.model_b, table)
    sort!(table, :id)
    battles = DataFrame(id=table.id, model_a=String.(table.model_a), model_b=String.(table.model_b),
        winner=[row.winner_model_a == 1 ? "model_a" : row.winner_model_b == 1 ? "model_b" : "tie"
                for row in eachrow(table)],
        group_id=[ismissing(row.first_prompt) ? "missing_prompt:$(row.id)" :
                  bytes2hex(sha256(row.first_prompt)) for row in eachrow(table)])
    train, validation, test = _split_battle_indices(battles, .70, .15; seed=42)
    battles.phase = zeros(Int, nrow(battles))
    battles.phase[train] .= 1
    battles.phase[validation] .= 2
    battles.phase[test] .= 3
    counts = Dict{String,Int}()
    for row in train, name in (battles.model_a[row], battles.model_b[row])
        counts[name] = get(counts, name, 0) + 1
    end
    filter!(row -> get(counts, row.model_a, 0) >= 100 && get(counts, row.model_b, 0) >= 100, battles)
    return battles, (findall(==(phase), battles.phase) for phase in 1:3)
end

function paper_inputs(dataset, input)
    if dataset === :human_preference_140k
        battles = isfile(input) ? _load_battles(input; dataset, min_games=100) :
                  _load_battles(nothing; parquet_dir=input, dataset, min_games=100)
        battles.group_id = String.(battles.group_id)
        train, validation, test = _split_battle_indices(battles, .70, .15; seed=42)
        lines = ["$(row.model_a)|$(row.model_b)|$(row.winner)|$(row.group_id)" for row in eachrow(battles)]
        expected = "7aefd2b0cf8accf454a654e067503030225c7391a2f1d282eb698caf7b96a5ec"
    else
        battles, (train, validation, test) = paper_55k(input)
        lines = ["$(row.id)|$(row.model_a)|$(row.model_b)|$(row.winner)|$(row.group_id)" for row in eachrow(battles)]
        expected = "e590d2bfbfadfe737826e2efc7bd1f3b11b4e994308ae61961d38be466808a27"
    end
    signature = bytes2hex(sha256(join(lines, "\n")))
    signature == expected || error("Input rows or ordering differ from the pinned paper protocol: $signature")
    for (a,b) in ((train,validation), (train,test), (validation,test))
        @assert isempty(intersect(Set(battles.group_id[a]), Set(battles.group_id[b])))
    end
    names, _, matches, draws = _index_battles(battles)
    pairs = [minmax(i,j) for (i,j) in matches]
    labels = [draw ? 2 : i < j ? 1 : 3 for ((i,j),draw) in zip(matches,draws)]
    return (; names, matches, draws, pairs, labels, groups=battles.group_id,
             train, validation, test, signature)
end

function paper_fit(data, model, setting)
    beta, nu = setting
    train = data.train
    matches, draws = data.matches[train], data.draws[train]
    epsilon = _draw_margin(beta, mean(draws))
    n = length(data.names)
    skills_prior = TSDC.DistributionBag(TSG.Gaussian1DFromMeanVariance(25.0, (25/3)^2), n)
    println("Fitting $model on $(length(train)) votes")
    if model === :vanilla
        skills = TS.vanilla_trueskill(skills_prior, beta^2, matches, epsilon;
                                     draw_mask=draws, outer_max_iters=100)
        return (; predictor=PL.VanillaPredictive(beta), skills, precisions=nothing, epsilon)
    end
    robust = model === :robust
    scale = robust ? 2beta^2 * (nu - 2) / nu : beta^2
    prior = _precision_prior(nu)
    algorithm = robust ? TS.robust_trueskill : TS.consistent_trueskill
    skills, precisions, diagnostics = algorithm(skills_prior,
        TSDC.DistributionBag(prior, robust ? length(train) : n), matches, epsilon;
        s2=scale, draw_mask=draws, inner_max_iters=100, outer_max_iters=100, diagnostics=true)
    diagnostics.converged || error("$model did not converge: $diagnostics")
    predictor = robust ? PL.RobustPredictive(prior, scale, 64) : PL.ConsistentPredictive(scale, 64)
    return (; predictor, skills, precisions, epsilon, scale)
end

function paper_predictions(data, model, fit, rows)
    cache = Dict{Tuple{Int,Int},NTuple{3,Float64}}()
    predictions = Matrix{Float64}(undef, length(rows), 3)
    for (position, row) in enumerate(rows)
        i,j = data.pairs[row]
        predictions[position,:] .= get!(cache, (i,j)) do
            p = model === :consistent ?
                PL.predict_outcomes(fit.predictor, fit.skills, fit.precisions, i, j, fit.epsilon) :
                PL.predict_outcomes(fit.predictor, fit.skills, i, j, fit.epsilon)
            (p.first, p.draw, p.second)
        end
    end
    return log.(max.(predictions, 1e-12))
end

function paper_probabilities(logits, parameters)
    values = parameters[1] .* logits
    values[:,2] .+= parameters[2]
    values .-= maximum(values; dims=2)
    probabilities = exp.(values)
    return probabilities ./ sum(probabilities; dims=2)
end

paper_losses(p, labels) = [-log(max(p[i, labels[i]], 1e-12)) for i in eachindex(labels)]

function paper_calibration(logits, labels)
    parameters = [1.0, 0.0]
    observed = [logits[i, labels[i]] for i in eachindex(labels)]
    for _ in 1:60
        p = paper_probabilities(logits, parameters)
        loss = mean(paper_losses(p, labels))
        expected = vec(sum(p .* logits; dims=2))
        g = [mean(expected .- observed), mean(p[:,2] .- (labels .== 2))]
        norm(g) < 1e-8 && break
        h11 = mean(vec(sum(p .* (logits .- expected).^2; dims=2)))
        h12 = mean(p[:,2] .* (logits[:,2] .- expected))
        h22 = mean(p[:,2] .* (1 .- p[:,2]))
        step = ([h11 h12; h12 h22] + 1e-8I) \ g
        accepted = false
        for alpha in 0.5 .^ (0:25)
            candidate = parameters .- alpha .* step
            (0.01 <= candidate[1] <= 100 && abs(candidate[2]) <= 10) || continue
            if mean(paper_losses(paper_probabilities(logits, candidate), labels)) < loss - 1e-12
                parameters = candidate
                accepted = true
                break
            end
        end
        accepted || break
    end
    return parameters
end

function paper_paired_interval(groups, baseline, candidate)
    ids = unique(groups)
    index = Dict(group => i for (i,group) in enumerate(ids))
    counts = zeros(Int, length(ids))
    totals = zeros(length(ids), 2)
    for (i, group) in enumerate(groups)
        j = index[group]
        counts[j] += 1
        totals[j,1] += baseline[i]
        totals[j,2] += candidate[i]
    end
    rng = MersenneTwister(42)
    differences = map(1:5000) do _
        sample = rand(rng, eachindex(ids), length(ids))
        n = sum(counts[sample])
        sum(totals[sample,2])/n - sum(totals[sample,1])/n
    end
    return (groups=length(ids), nll_delta=mean(candidate)-mean(baseline),
            ci95_low=quantile(differences, .025), ci95_high=quantile(differences, .975))
end

function paper_upsets(elo, matches, draws)
    eligible = zeros(Int, length(elo))
    upsets = zeros(Int, length(elo))
    for ((winner, loser), draw) in zip(matches, draws)
        (draw || elo[winner] == elo[loser]) && continue
        upset = elo[winner] < elo[loser]
        for model in (winner, loser)
            eligible[model] += 1
            upsets[model] += upset
        end
    end
    rate = [n == 0 ? missing : 100 * u / n for (u, n) in zip(upsets, eligible)]
    return (; heldout_upsets=upsets, heldout_eligible_votes=eligible, heldout_upset_pct=rate)
end

function paper_estimate_rows(data, fit)
    exposure = zeros(Int, length(data.names))
    for (i,j) in data.matches[data.train]
        exposure[i] += 1
        exposure[j] += 1
    end
    skill = TSG.mean.(fit.skills)
    sd = sqrt.(TSG.variance.(fit.skills))
    q = [Distributions.Gamma(TSGamma.shape(b), 1/TSGamma.rate(b)) for b in fit.precisions]
    return DataFrame(model=data.names, training_comparisons=exposure, skill=skill, skill_sd=sd,
        skill_low_3sd=skill .- 3 .* sd, skill_high_3sd=skill .+ 3 .* sd,
        performance_sd=sqrt.(fit.scale .* TSGamma.expected_inverse_precision.(fit.precisions)),
        performance_lo=sqrt.(fit.scale ./ quantile.(q,.975)),
        performance_hi=sqrt.(fit.scale ./ quantile.(q,.025)))
end

function paper_model_rows(data, fit)
    rows = paper_estimate_rows(data, fit)
    elo = RatingBaselines.fit_bradley_terry(data.matches[data.train], data.draws[data.train],
                                           length(data.names))
    insertcols!(rows, 3, :elo => elo)
    return hcat(rows, DataFrame(paper_upsets(elo, data.matches[data.test], data.draws[data.test])))
end

function write_paper_arena_figure(rows, output; rank_by=:training_comparisons)
    ranked = sort(rows, [rank_by, :model]; rev=[true,false])
    shown = ranked[1:15, :]
    rows.in_main_figure = in(Set(shown.model)).(rows.model)
    rows.display_name = [get(PREFERENCE_LABELS, name, name) for name in rows.model]
    CSV.write(joinpath(output, "model_values.csv"), sort(rows, :skill; rev=true))
    preference_panels(shown, joinpath(output, "human_preference_consistency"))
end

function run_paper_arena(dataset, input, output)
    settings = getproperty(PAPER_SETTINGS, dataset)
    data = paper_inputs(dataset, input)
    mkpath(output)
    CSV.write(joinpath(output, "protocol.csv"), [(; dataset, data.signature, models=length(data.names),
        train=length(data.train), validation=length(data.validation), test=length(data.test))])
    losses = Dict{Symbol,Vector{Float64}}()
    summary = NamedTuple[]
    labels = data.labels[data.test]
    priority = rand(MersenneTwister(42), length(labels))
    for model in keys(settings)
        fit = paper_fit(data, model, settings[model])
        validation = paper_predictions(data, model, fit, data.validation)
        parameters = paper_calibration(validation, data.labels[data.validation])
        test = paper_predictions(data, model, fit, data.test)
        probabilities = paper_probabilities(test, parameters)
        losses[model] = paper_losses(probabilities, labels)
        correct = [argmax(row) == label for (row, label) in zip(eachrow(probabilities), labels)]
        confidence = vec(maximum(probabilities; dims=2))
        accepted = sortperm(eachindex(labels); by=i -> (-confidence[i], priority[i]))
        half = accepted[1:round(Int, length(labels) / 2)]
        beta, nu = settings[model]
        push!(summary, (; model, beta, nu, inverse_temperature=parameters[1],
                        draw_intercept=parameters[2], nll=mean(losses[model]),
                        accuracy=mean(correct), risk50=mean(.!correct[half])))
        if dataset === :human_preference_140k && model === :consistent
            write_paper_arena_figure(paper_model_rows(data, fit), output)
        end
    end
    intervals = [(; model, paper_paired_interval(data.groups[data.test], losses[:vanilla], losses[model])...)
                 for model in (:robust,:consistent)]
    CSV.write(joinpath(output, "summary.csv"), summary)
    CSV.write(joinpath(output, "nll_bootstrap.csv"), intervals)
    display(DataFrame(summary))
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 || error("Usage: paper.jl DATASET [PARQUET_DIR_OR_CSV]")
    dataset = Symbol(ARGS[1])
    hasproperty(PAPER_SETTINGS, dataset) || error("Choose human_preference_140k or human_preference_55k")
    input = length(ARGS) == 2 ? ARGS[2] : dataset === :human_preference_140k ? DEFAULT_PARQUET_DIR : ""
    output = joinpath(@__DIR__, "..", "..", "output", "arena_paper", String(dataset))
    run_paper_arena(dataset, input, output)
end
