include("paper.jl")

const COMPARIA_REVISION = "fd92db9d3b2f7b18c6024367711a4cabb16e7a34"
const COMPARIA_SHA256 = "9a14ea67dcd8fbb3d07b74d8ca863773ab0ca36492b47d427d5051d22d8652d4"
# Fixed from the exploratory 2026 fit before changing the unit to one vote per conversation.
const COMPARIA_SETTING = (16.0, 32.0)

function comparia_turns(path)
    isfile(path) || error("Supply the pinned comparia-fr-arena.parquet as COMPARIA_INPUT")
    println("Verifying the pinned Compar:IA snapshot")
    open(io -> bytes2hex(sha256(io, 4 * 1024^2)), path) == COMPARIA_SHA256 ||
        error("Compar:IA input does not match the pinned snapshot $COMPARIA_REVISION")
    println("Reading vote fields")
    source = replace(abspath(path), '\\' => '/', "'" => "''")
    db = DuckDB.DB()
    votes = try
        DataFrame(DuckDB.DBInterface.execute(db, """
            SELECT comparison_id AS group_id, turn, model_a, model_b, choice,
                   CAST(timestamp AS VARCHAR) AS timestamp
            FROM read_parquet('$source')
            WHERE timestamp >= TIMESTAMP '2026-01-01' AND timestamp < TIMESTAMP '2027-01-01'
              AND choice IN ('a_better', 'b_better', 'both_good', 'both_bad')
              AND comparison_id IS NOT NULL AND turn IS NOT NULL
              AND model_a IS NOT NULL AND model_b IS NOT NULL AND model_a != model_b
            """))
    finally
        DuckDB.close(db)
    end
    return prepare_comparia_turns(votes)
end

function prepare_comparia_turns(votes)
    turns = unique(votes)
    # Exclude ambiguous turn keys before choosing the last eligible vote.
    unambiguous = unique(turns, [:group_id, :turn]; keep=:noduplicates)
    sort!(unambiguous, [:group_id, :turn])
    unambiguous.winner = [choice == "a_better" ? "model_a" :
                          choice == "b_better" ? "model_b" : "tie"
                          for choice in unambiguous.choice]
    return unambiguous
end

function comparia_exposure(battles, rows)
    counts = Dict{String,Int}()
    for row in rows, model in (battles.model_a[row], battles.model_b[row])
        counts[model] = get(counts, model, 0) + 1
    end
    return counts
end

function comparia_cohort(turns)
    battles = unique(turns, :group_id; keep=:last)
    while true
        counts = comparia_exposure(battles, axes(battles, 1))
        before = nrow(battles)
        filter!(row -> get(counts, row.model_a, 0) >= 100 &&
                       get(counts, row.model_b, 0) >= 100, battles)
        nrow(battles) == before && break
    end
    return battles
end

function fit_comparia(battles)
    names, _, matches, draws = _index_battles(battles)
    data = (; names, matches, draws, train=eachindex(matches))
    fit = paper_fit(data, :consistent, COMPARIA_SETTING)
    rows = paper_estimate_rows(data, fit)
    rename!(rows, :training_comparisons => :comparisons)
    return rows, fit.epsilon
end

function run_comparia(path, output)
    write_comparia_results(comparia_turns(path), output)
end

function write_comparia_results(turns, output)
    println("Preparing $(nrow(turns)) labeled turns")
    battles = comparia_cohort(turns)
    println("Retained $(nrow(battles)) conversations")
    rows, epsilon = fit_comparia(battles)
    mkpath(output)
    CSV.write(joinpath(output, "protocol.csv"), [(revision=COMPARIA_REVISION,
        source_sha256=COMPARIA_SHA256, first_date=Date(first(minimum(battles.timestamp), 10)),
        last_date=Date(first(maximum(battles.timestamp), 10)),
        labeled_turns=nrow(turns), labeled_conversations=length(unique(turns.group_id)),
        votes=nrow(battles), models=nrow(rows),
        beta=COMPARIA_SETTING[1], nu=COMPARIA_SETTING[2], epsilon)])
    write_paper_arena_figure(rows, output; rank_by=:skill)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: comparia.jl PARQUET")
    run_comparia(only(ARGS), joinpath(@__DIR__, "..", "..", "output", "comparia_paper"))
end
