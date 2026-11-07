include("paper.jl")

function run_arena_figure(input, output)
    dataset = :human_preference_140k
    battles = isfile(input) ? _load_battles(input; dataset, min_games=100) :
              _load_battles(nothing; parquet_dir=input, dataset, min_games=100)
    records = join((string(row.model_a, '|', row.model_b, '|', row.winner, '|', row.group_id)
                    for row in eachrow(battles)), "\n")
    signature = bytes2hex(sha256(IOBuffer(records), 4 * 1024^2))
    signature == "7aefd2b0cf8accf454a654e067503030225c7391a2f1d282eb698caf7b96a5ec" ||
        error("LMArena input differs from the pinned snapshot")
    names, _, matches, draws = _index_battles(battles)
    data = (; names, matches, draws, train=eachindex(matches))
    setting = PAPER_SETTINGS.human_preference_140k.consistent
    fit = paper_fit(data, :consistent, setting)
    rows = paper_estimate_rows(data, fit)
    rename!(rows, :training_comparisons => :comparisons)
    mkpath(output)
    CSV.write(joinpath(output, "protocol.csv"), [(dataset,
        revision=_dataset_spec(dataset).revision, signature,
        first_date=Date(first(string(minimum(battles.timestamp)), 10)),
        last_date=Date(first(string(maximum(battles.timestamp)), 10)),
        votes=length(matches), models=length(names), beta=setting[1], nu=setting[2],
        epsilon=fit.epsilon)])
    write_paper_arena_figure(rows, output; rank_by=:skill)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: figure.jl [PARQUET_DIR_OR_CSV]")
    input = isempty(ARGS) ? DEFAULT_PARQUET_DIR : only(ARGS)
    output = joinpath(@__DIR__, "..", "..", "output", "arena_figure", "human_preference_140k")
    run_arena_figure(input, output)
end
