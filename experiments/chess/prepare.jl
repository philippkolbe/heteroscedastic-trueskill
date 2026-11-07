#!/usr/bin/env julia
#
# Chess: parse ChessBase text → CSV, validate, deduplicate, analyze, plot, report.
# Can be run as a script OR included from a notebook:
#
#   julia experiments/chess/prepare.jl [chess.txt] [out_dir]   # script mode
#   include("experiments/chess/prepare.jl")                  # notebook mode
#
# In notebook mode every function is available to call individually.
# Plot functions return Plots.Plot objects for inline display.

using Dates
using Printf
using CSV
using DataFrames
using Statistics
using StatsBase
using Plots

# ─── config (isdefined guards allow notebook re-include without errors) ───────

@isdefined(MIN_YEAR)             || (global MIN_YEAR             = 2004)
@isdefined(MIN_GAMES_FOR_LEADER) || (global MIN_GAMES_FOR_LEADER = 5)
@isdefined(MATCH_CHUNK_SIZE)     || (global MATCH_CHUNK_SIZE     = 100_000)

# ─── structs ──────────────────────────────────────────────────────────────────

mutable struct PlayerInfo
    id::String
    name::String
    start_year::Int
    end_year::Int
    max_elo::Union{Int,Missing}
end

mutable struct MatchStats
    total::Int
    missing_date::Int
    exact_date::Int
    year_fallback_date::Int
    missing_white_elo::Int
    missing_black_elo::Int
    draws::Int
    white_wins::Int
    black_wins::Int
    unknown_results::Int
    events::Set{String}
    ecos::Set{String}
    years::Vector{Union{Int,Missing}}
    avg_elos::Vector{Union{Float64,Missing}}
    elo_diffs::Vector{Union{Int,Missing}}
end

MatchStats() = MatchStats(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Set{String}(), Set{String}(),
    Union{Int,Missing}[],
    Union{Float64,Missing}[],
    Union{Int,Missing}[],
)

function update_match_stats!(s::MatchStats, row::NamedTuple)
    s.total += 1
    if row._has_any_date
        row._has_exact_date ? (s.exact_date += 1) : (s.year_fallback_date += 1)
    else
        s.missing_date += 1
    end
    row.white_elo === missing && (s.missing_white_elo += 1)
    row.black_elo === missing && (s.missing_black_elo += 1)
    if     row.result == "½-½"; s.draws       += 1
    elseif row.result == "1-0"; s.white_wins  += 1
    elseif row.result == "0-1"; s.black_wins  += 1
    else                        s.unknown_results += 1
    end
    row.event !== missing && push!(s.events, String(row.event))
    row.eco   !== missing && push!(s.ecos,   String(row.eco))
    push!(s.years, row.year)
    avg = (row.white_elo === missing || row.black_elo === missing) ? missing :
          (row.white_elo + row.black_elo) / 2.0
    push!(s.avg_elos, avg)
    d = (row.white_elo === missing || row.black_elo === missing) ? missing :
        (row.white_elo - row.black_elo)
    push!(s.elo_diffs, d)
end

# ─── edge logger ──────────────────────────────────────────────────────────────

const EDGE_SNIPPET_MAX_LINES = 10
const EDGE_SNIPPET_MAX_CHARS = 2_000

mutable struct EdgeLogger
    io::IO
    max_per_category::Int
    counts::Dict{String,Int}
    written::Dict{String,Int}
end

function EdgeLogger(path::String; max_per_category::Int = 200)
    io = open(path, "w")
    println(io, "# Chess parsing edge cases")
    println(io, "generated_at=", Dates.now())
    println(io)
    EdgeLogger(io, max_per_category, Dict{String,Int}(), Dict{String,Int}())
end

function log_edge_case!(l::EdgeLogger, cat::String, msg::String,
                         snippet::Vector{Tuple{Int,String}})
    l.counts[cat] = get(l.counts, cat, 0) + 1
    already = get(l.written, cat, 0)
    already >= l.max_per_category && return
    l.written[cat] = already + 1
    println(l.io, "## category=", cat, " case=", l.written[cat])
    println(l.io, "message: ", msg)
    println(l.io, "snippet:")
    used = 0
    for (idx, (ln, text)) in enumerate(snippet)
        idx > EDGE_SNIPPET_MAX_LINES && break
        t = strip(text); used += ncodeunits(t)
        if used > EDGE_SNIPPET_MAX_CHARS
            println(l.io, "  ... [truncated]"); break
        end
        println(l.io, "  ", lpad(string(ln), 8), ": ", t)
    end
    println(l.io)
end

function close_edge_logger!(l::EdgeLogger)
    println(l.io, "# Summary")
    for cat in sort!(collect(keys(l.counts)))
        println(l.io, "- ", cat, ": total=", l.counts[cat],
                ", shown=", get(l.written, cat, 0))
    end
    close(l.io)
end

# ─── ChessBase text parsing helpers ──────────────────────────────────────────────────────

normalize_name(n::AbstractString) = lowercase(strip(replace(n, r"\s+" => " ")))
looks_like_header(l::AbstractString) = occursin(" - ", l) && !occursin(r"^\d+\.", l)
looks_like_moves(l::AbstractString)  = occursin(r"^\d+\.", l)

function extract_result(text::AbstractString)
    m = collect(eachmatch(r"(1-0|0-1|½-½|1/2-1/2|\*)", text))
    isempty(m) ? missing : m[end].match
end

normalize_result(r) = r === missing ? missing : (r == "1/2-1/2" ? "½-½" : String(r))

function parse_date_and_event(line::Union{AbstractString,Nothing})
    line === nothing && return (missing, "", false, false)
    s = strip(line)
    # full date: dd.mm.yyyy at end of line
    m_full = match(r"(\d{2}\.\d{2}\.\d{4})\s*$", s)
    if m_full !== nothing
        dt = try Date(m_full.captures[1], dateformat"dd.mm.yyyy") catch; missing end
        ev = strip(replace(s, r",?\s*\d{2}\.\d{2}\.\d{4}\s*$" => ""))
        return (dt, ev, true, dt !== missing)
    end
    # year only: 19xx or 20xx at end of line
    m_year = match(r"(?:^|[^0-9])((?:19|20)\d{2})\s*$", s)
    if m_year !== nothing
        y = try parse(Int, m_year.captures[1]) catch; -1 end
        if y > 0
            ev = strip(replace(s, r",?\s*(?:19|20)\d{2}\s*$" => ""))
            return (Date(y, 1, 1), ev, false, true)
        end
    end
    return (missing, s, false, false)
end

function parse_player(raw::AbstractString)
    s = strip(raw)
    m = match(r"^(.*?)\s*\((\d{3,4})\)\s*$", s)
    m === nothing && return (String(s), missing)
    name = strip(m.captures[1])
    elo  = try parse(Int, m.captures[2]) catch; missing end
    (String(name), elo)
end

function parse_header(line::AbstractString)
    s = strip(line)
    eco = missing
    em  = match(r"\[([A-E][0-9]{2})\]\s*$", s)
    if em !== nothing
        eco = em.captures[1]
        s   = strip(replace(s, r"\[[A-E][0-9]{2}\]\s*$" => ""))
    end
    si = findfirst(" - ", s); si === nothing && return nothing
    wn, we = parse_player(s[1:prevind(s, first(si))])
    bn, be = parse_player(s[nextind(s, last(si)):end])
    (white_name=wn, white_elo=we, black_name=bn, black_elo=be, eco=eco)
end

function get_or_create_player!(players::Dict{String,PlayerInfo},
                                next_idx::Ref{Int},
                                name::AbstractString, elo, yr)
    key = normalize_name(name)
    if !haskey(players, key)
        id = "chp_" * lpad(string(next_idx[]), 8, '0')
        next_idx[] += 1
        sy = yr === missing ? 0 : yr
        players[key] = PlayerInfo(id, String(name), sy, sy, elo)
    end
    p = players[key]
    if yr !== missing
        (p.start_year == 0 || yr < p.start_year) && (p.start_year = yr)
        yr > p.end_year && (p.end_year = yr)
    end
    if elo !== missing && (p.max_elo === missing || elo > p.max_elo)
        p.max_elo = elo
    end
    p.id
end

function outcome_fields(result, wn, bn)
    result == "1-0" ? (wn, bn) :
    result == "0-1" ? (bn, wn) : (missing, missing)
end

function build_match_row(players, pidx, midx, hline, eline, mlines)
    hline === nothing && return nothing
    isempty(mlines)   && return nothing
    h = parse_header(hline); h === nothing && return nothing
    moves_text = join(mlines, " ")
    result     = normalize_result(extract_result(moves_text))
    gdate, event, exact, hasany = parse_date_and_event(eline)
    yr = gdate === missing ? missing : year(gdate)
    get_or_create_player!(players, pidx, h.white_name, h.white_elo, yr)
    get_or_create_player!(players, pidx, h.black_name, h.black_elo, yr)
    wn, ln = outcome_fields(result, h.white_name, h.black_name)
    mid    = "chm_" * lpad(string(midx[]), 10, '0'); midx[] += 1
    (match_id=mid, year=yr, date=gdate,
     event=(isempty(event) ? missing : event),
     white_name=h.white_name, black_name=h.black_name,
     white_elo=h.white_elo, black_elo=h.black_elo,
     result=result, winner_name=wn, is_draw=(result == "½-½"), eco=h.eco,
     _has_any_date=hasany, _has_exact_date=exact)
end

function strip_internal(r::NamedTuple)
    (match_id=r.match_id, year=r.year, date=r.date, event=r.event,
     white_name=r.white_name, black_name=r.black_name,
     white_elo=r.white_elo, black_elo=r.black_elo,
     result=r.result, winner_name=r.winner_name, is_draw=r.is_draw, eco=r.eco)
end

function flush_rows!(path, rows, write_header)
    isempty(rows) && return write_header
    CSV.write(path, rows; append=!write_header, writeheader=write_header)
    empty!(rows)
    false
end

# ─── Step 1: parse ChessBase text → CSV ──────────────────────────────────────────────────

function parse_chess_file(input_path, matches_csv, players_csv, stats_path, edge_path)
    println("Parsing ChessBase text: $input_path")
    players  = Dict{String,PlayerInfo}()
    pidx     = Ref(1); midx = Ref(1)
    hline    = nothing; eline = nothing; expect_event = false
    mlines   = String[]; glines = Tuple{Int,String}[]
    parsed   = 0; skipped = 0; wrote_hdr = true
    rows     = NamedTuple[]; rcounts = Dict{String,Int}()
    mstats   = MatchStats(); elog = EdgeLogger(edge_path)
    lno      = 0

    try
        open(input_path) do io
            for raw in eachline(io)
                lno += 1
                line = strip(raw)
                isempty(line) && continue

                if looks_like_header(line) && isempty(mlines)
                    hline = line; eline = nothing; expect_event = true
                    empty!(glines); push!(glines, (lno, String(raw)))
                    continue
                end
                hline === nothing && continue

                if expect_event && !looks_like_moves(line)
                    eline = line; expect_event = false
                    push!(glines, (lno, String(raw))); continue
                end
                expect_event = false
                push!(mlines, line); push!(glines, (lno, String(raw)))

                r = extract_result(line)
                if r !== missing
                    row = build_match_row(players, pidx, midx, hline, eline, mlines)
                    if row === nothing
                        skipped += 1
                        log_edge_case!(elog, "unparsed_header_block",
                            "Could not parse header into a game", glines)
                    else
                        push!(rows, strip_internal(row)); parsed += 1
                        rk = row.result === missing ? "missing" : String(row.result)
                        rcounts[rk] = get(rcounts, rk, 0) + 1
                        update_match_stats!(mstats, row)
                    end
                    length(rows) >= MATCH_CHUNK_SIZE &&
                        (wrote_hdr = flush_rows!(matches_csv, rows, wrote_hdr))
                    hline = nothing; eline = nothing
                    empty!(mlines); empty!(glines); expect_event = false
                end
            end
        end
        if hline !== nothing
            skipped += 1
            log_edge_case!(elog, "incomplete_game_at_eof",
                "Incomplete game at EOF", glines)
        end
    finally
        close_edge_logger!(elog)
    end

    flush_rows!(matches_csv, rows, wrote_hdr)

    # players CSV
    plist = sort!(collect(values(players)), by = p -> p.id)
    prows = [(player_id=p.id, name=p.name,
              start_year=(p.start_year == 0 ? missing : p.start_year),
              end_year=(p.end_year == 0 ? missing : p.end_year),
              max_elo=p.max_elo) for p in plist]
    CSV.write(players_csv, prows)

    # stats report
    open(stats_path, "w") do io
        println(io, "# Chess parsing statistics")
        println(io, "generated_at=", Dates.now())
        println(io)
        println(io, "[global]")
        println(io, "parsed_games=", parsed)
        println(io, "skipped_games=", skipped)
        println(io, "unique_players=", length(players))
        println(io)
        println(io, "[matches.overview]")
        println(io, "total=",              mstats.total)
        println(io, "exact_date=",         mstats.exact_date)
        println(io, "year_fallback_date=", mstats.year_fallback_date)
        println(io, "missing_date=",       mstats.missing_date)
        println(io, "unique_tournaments=", length(mstats.events))
        println(io, "unique_eco=",         length(mstats.ecos))
        println(io, "draws=",              mstats.draws)
        println(io, "white_wins=",         mstats.white_wins)
        println(io, "black_wins=",         mstats.black_wins)
        println(io, "unknown_results=",    mstats.unknown_results)
        println(io)
        for (label, vals) in [("matches.year", mstats.years),
                               ("matches.avg_elo", mstats.avg_elos),
                               ("matches.elo_diff", mstats.elo_diffs)]
            present = [Float64(v) for v in vals if v !== missing]
            na = length(vals) - length(present)
            println(io, "[$label]")
            println(io, "na=", na)
            if !isempty(present)
                println(io, "min=",    minimum(present))
                println(io, "max=",    maximum(present))
                println(io, "mean=",   mean(present))
                println(io, "median=", median(present))
            end
            println(io)
        end
        println(io, "[result_counts]")
        for k in sort!(collect(keys(rcounts)))
            println(io, k, "=", rcounts[k])
        end
    end

    println("  Parsed:  $parsed")
    println("  Skipped: $skipped")
    println("  Players: $(length(players))")
    println("  Matches: $matches_csv")
    println("  Players: $players_csv")
    println("  Stats:   $stats_path")
    println("  Edges:   $edge_path")
end

# ─── Step 2: load CSV, validate, deduplicate, filter ─────────────────────────

function load_and_clean_chess(matches_csv::String; verbose::Bool=true)
    df    = CSV.read(matches_csv, DataFrame; types=Dict(:year => Union{Int,Missing}))
    n_raw = nrow(df)
    verbose && println("Loaded $n_raw rows from $matches_csv")

    # duplicate match_id check
    n_dup_id = n_raw - length(unique(df.match_id))
    if n_dup_id > 0
        verbose && println("WARNING: $n_dup_id duplicate match_id values — removing")
        unique!(df, :match_id)
    else
        verbose && println("No duplicate match_ids")
    end

    # exact-content duplicate check (ignoring match_id)
    content_cols = filter(c -> c != "match_id", names(df))
    n_dup_row    = nrow(df) - nrow(unique(df, content_cols))
    if n_dup_row > 0
        verbose && println("WARNING: $n_dup_row exact content-duplicate rows — removing")
        unique!(df, content_cols)
    else
        verbose && println("No content-duplicate rows")
    end

    # must have a valid year
    before = nrow(df); dropmissing!(df, :year)
    verbose && println("Dropped $(before - nrow(df)) rows with missing year  ($(nrow(df)) remain)")

    # must have a complete result (not missing, not "*")
    before = nrow(df)
    df = df[.!ismissing.(df.result) .& (df.result .!= "*"), :]
    verbose && println("Dropped $(before - nrow(df)) rows with missing/unfinished result  ($(nrow(df)) remain)")

    # apply year floor
    before = nrow(df)
    df = df[df.year .>= MIN_YEAR, :]
    verbose && println("Dropped $(before - nrow(df)) rows with year < $MIN_YEAR  ($(nrow(df)) remain)")

    # coerce elo columns to numeric
    for col in [:white_elo, :black_elo]
        df[!, col] = [v === missing ? missing :
                      (v isa Number ? Int(v) : tryparse(Int, string(v)))
                      for v in df[!, col]]
    end
    df
end

# ─── Step 3: yearly leader Elo with tournament provenance ────────────────────

"""Return (leaders, py_max) DataFrames. py_max has one row per (player, year)
with the peak Elo for that year and the game/event it came from."""
function compute_yearly_leaders(df::DataFrame)
    white = select(df, :year, :white_name => :player, :white_elo => :elo,
                       :event, :date, :match_id)
    black = select(df, :year, :black_name => :player, :black_elo => :elo,
                       :event, :date, :match_id)
    long = vcat(white, black)
    dropmissing!(long, [:year, :player, :elo])
    long.elo  = Float64.(long.elo)
    long.year = Int.(long.year)

    games_py = combine(groupby(long, [:player, :year]), nrow => :n_games)

    sort!(long, [:player, :year, order(:elo, rev=true)])
    py_max = combine(groupby(long, [:player, :year]),
        :elo      => first => :elo,
        :event    => first => :event,
        :date     => first => :date,
        :match_id => first => :match_id)

    py_max = innerjoin(py_max, games_py, on=[:player, :year])
    py_max = py_max[py_max.n_games .>= MIN_GAMES_FOR_LEADER, :]

    yr_max  = combine(groupby(py_max, :year), :elo => maximum => :year_max_elo)
    leaders = innerjoin(py_max, yr_max, on=:year)
    leaders = leaders[leaders.elo .== leaders.year_max_elo, :]
    sort!(leaders, :year), py_max
end

"""Unique players active per year (both white and black appearances)."""
function compute_player_activity(df::DataFrame)
    long = vcat(select(df, :year, :white_name => :player),
                select(df, :year, :black_name => :player))
    dropmissing!(long)
    py = combine(groupby(long, :year),
        :player => (x -> length(unique(x))) => :n_players)
    sort!(py, :year)
end

# ─── Step 4: individual plot functions (each returns a Plots.Plot) ───────────

"""Bar chart: number of games recorded per year."""
function plot_matches_per_year(df::DataFrame)
    gpy = combine(groupby(df, :year), nrow => :n) |> x -> sort(x, :year)
    bar(gpy.year, gpy.n;
        xlabel="Year", ylabel="Games", title="Chess games recorded per year",
        legend=false, linewidth=0, color=:steelblue, size=(800, 400))
end

"""Bar chart: unique players active (at least one game) per year."""
function plot_players_per_year(activity::DataFrame)
    bar(activity.year, activity.n_players;
        xlabel="Year", ylabel="Unique players",
        title="Unique players active per year",
        legend=false, linewidth=0, color=:darkorange, size=(800, 400))
end

"""Histogram of white and black Elo values across all games."""
function plot_elo_distribution(df::DataFrame)
    elos = Float64.(filter(!ismissing, vcat(df.white_elo, df.black_elo)))
    histogram(elos; bins=60,
        xlabel="Elo", ylabel="Count",
        title="Elo distribution — all games, both players",
        legend=false, color=:steelblue, alpha=0.8, size=(800, 400))
end

"""Median Elo per year with IQR ribbon — shows rating inflation / deflation."""
function plot_elo_trend(df::DataFrame)
    elos = vcat(select(df, :year, :white_elo => :elo),
                select(df, :year, :black_elo => :elo))
    dropmissing!(elos)
    elos.elo = Float64.(elos.elo)
    grp = combine(groupby(elos, :year),
        :elo => median                   => :med,
        :elo => (x -> quantile(x, 0.25)) => :q25,
        :elo => (x -> quantile(x, 0.75)) => :q75) |> x -> sort(x, :year)
    plot(grp.year, grp.med;
        ribbon=(grp.med .- grp.q25, grp.q75 .- grp.med),
        fillalpha=0.2, xlabel="Year", ylabel="Elo",
        title="Elo over time — median ± IQR band",
        label="Median", color=:steelblue, linewidth=2.5, size=(800, 400))
end

"""Histogram of per-game Elo differences (white_elo − black_elo)."""
function plot_elo_diff_distribution(df::DataFrame)
    diffs = Float64.(filter(!ismissing,
        [we === missing || be === missing ? missing : we - be
         for (we, be) in zip(df.white_elo, df.black_elo)]))
    histogram(diffs; bins=60,
        xlabel="White Elo − Black Elo", ylabel="Count",
        title="Elo difference per game",
        legend=false, color=:mediumpurple, alpha=0.8, size=(800, 400))
end

"""Bar chart of game results (white win / draw / black win)."""
function plot_result_share(df::DataFrame)
    counts = sort(combine(groupby(df, :result), nrow => :n), order(:n, rev=true))
    labels = map(r -> r === missing ? "missing" :
                      r == "½-½"  ? "Draw" :
                      r == "1-0"  ? "White win" :
                      r == "0-1"  ? "Black win" : String(r),
                 counts.result)
    bar(1:nrow(counts), counts.n;
        xticks=(1:nrow(counts), labels),
        xlabel="Result", ylabel="Count",
        title="Game result distribution",
        legend=false, color=[:steelblue, :coral, :forestgreen, :gray],
        size=(700, 400))
end

"""Line plot of peak Elo trajectories for all yearly #1 players."""
function plot_leader_elo(leaders::DataFrame, py_max::DataFrame)
    leader_names = unique(leaders.player)
    p = plot(; xlabel="Year", ylabel="Elo",
               title="Elo trajectories — yearly top players",
               legend=:outertopright, size=(1000, 500))
    for name in leader_names
        grp = sort(py_max[py_max.player .== name, :], :year)
        nrow(grp) < 1 && continue
        plot!(p, grp.year, grp.elo; label=name, linewidth=2)
        lrows = leaders[leaders.player .== name, :]
        scatter!(p, lrows.year, lrows.elo; label=false, markersize=7,
                 markershape=:star5)
    end
    p
end

"""Horizontal bar chart of the top N players by total games in the dataset."""
function plot_top_players(py_max::DataFrame; n::Int=20)
    totals = sort(combine(groupby(py_max, :player), :n_games => sum => :total),
                  order(:total, rev=true))
    top = first(totals, n)
    bar(top.total;
        xticks=(1:nrow(top), top.player), xrotation=45,
        xlabel="Player", ylabel="Total games in dataset",
        title="Top $n most active players",
        legend=false, color=:steelblue,
        size=(900, 420), bottom_margin=70Plots.px)
end

# ─── Step 4b: save all plots + CSV + report to disk ──────────────────────────

"""Run all plot functions, save PNGs, write leader CSV and text report.
   Returns the path to the written report file."""
function save_chess_outputs(df::DataFrame, leaders::DataFrame,
                             py_max::DataFrame, n_raw::Int, out_dir::String)
    mkpath(out_dir)
    activity = compute_player_activity(df)
    for (fname, p) in [
            ("chess_matches_per_year.png",       plot_matches_per_year(df)),
            ("chess_players_per_year.png",        plot_players_per_year(activity)),
            ("chess_elo_distribution.png",        plot_elo_distribution(df)),
            ("chess_elo_trend.png",               plot_elo_trend(df)),
            ("chess_elo_diff_distribution.png",   plot_elo_diff_distribution(df)),
            ("chess_result_share.png",            plot_result_share(df)),
            ("chess_leader_elo.png",              plot_leader_elo(leaders, py_max)),
            ("chess_top_players.png",             plot_top_players(py_max)),
        ]
        savefig(p, joinpath(out_dir, fname))
    end
    CSV.write(joinpath(out_dir, "chess_yearly_leader_elo.csv"), leaders)
    rpath = joinpath(out_dir, "chess_report.txt")
    write_analysis_report(rpath, df, leaders, n_raw)
    println("Outputs saved to $out_dir")
    rpath
end

# ─── Step 5: text report ──────────────────────────────────────────────────────

function write_analysis_report(path::String, df::DataFrame,
                                leaders::DataFrame, n_raw::Int)
    yrs  = sort(unique(df.year))
    gpy  = combine(groupby(df, :year), nrow => :n) |> x -> sort(x, :year)
    elos = Float64.(filter(!ismissing, vcat(df.white_elo, df.black_elo)))
    open(path, "w") do io
        println(io, "# Chess analysis report")
        println(io, "generated_at=", Dates.now())
        println(io)
        println(io, "[data]")
        println(io, "raw_rows=",               n_raw)
        println(io, "clean_rows=",             nrow(df))
        println(io, "min_year_filter=",        MIN_YEAR)
        println(io, "min_games_for_leader=",   MIN_GAMES_FOR_LEADER)
        println(io, "year_range=",             first(yrs), "–", last(yrs))
        println(io)
        println(io, "[elo_stats]")
        if !isempty(elos)
            println(io, "median=",  round(median(elos), digits=1))
            println(io, "mean=",    round(mean(elos),   digits=1))
            println(io, "q25=",     round(quantile(elos, 0.25), digits=1))
            println(io, "q75=",     round(quantile(elos, 0.75), digits=1))
            println(io, "min=",     minimum(elos))
            println(io, "max=",     maximum(elos))
        end
        println(io)
        println(io, "[yearly_leaders]")
        for r in eachrow(leaders)
            @printf(io, "%d: %-28s  elo=%.0f  games=%4d  event=%s  date=%s\n",
                r.year, r.player, r.elo, r.n_games, r.event, r.date)
        end
        println(io)
        println(io, "[games_per_year]")
        for r in eachrow(gpy)
            println(io, "$(r.year)=$(r.n)")
        end
    end
    println("Chess report written: $path")
end

"""Read a saved report file and print it to stdout (useful in notebooks)."""
function log_report(path::String)
    isfile(path) || (println("Report not found: $path"); return)
    println(read(path, String))
end

# ─── main ─────────────────────────────────────────────────────────────────────

function main()
    root      = dirname(dirname(abspath(@__FILE__)))
    data_dir  = joinpath(root, "data", "chess")
    out_dir   = joinpath(root, "output")
    mkpath(out_dir)

    chess_txt    = joinpath(data_dir, "chess.txt")
    matches_csv  = joinpath(data_dir, "chess_matches.csv")
    players_csv  = joinpath(data_dir, "chess_players.csv")

    # override via ARGS (must happen before computing output paths)
    length(ARGS) >= 1 && (chess_txt = ARGS[1])
    if length(ARGS) >= 2
        out_dir = ARGS[2]; mkpath(out_dir)
    end

    # Keep CSVs with the requested output, including when no data directory exists.
    matches_csv = joinpath(out_dir, "chess_matches.csv")
    players_csv = joinpath(out_dir, "chess_players.csv")

    stats_path = joinpath(out_dir, "chess_parse_stats.txt")
    edge_path  = joinpath(out_dir, "chess_edge_cases.log")

    # Step 1 – parse ChessBase text (skip if CSV already present and non-empty)
    if !isfile(matches_csv) || filesize(matches_csv) < 100
        println("\n=== Step 1: Parse ChessBase text → CSV ===")
        parse_chess_file(chess_txt, matches_csv, players_csv, stats_path, edge_path)
    else
        println("\n=== Step 1: Skipping ChessBase text parse (CSV exists: $matches_csv) ===")
    end

    # Step 2 – load, validate, clean
    println("\n=== Step 2: Load & clean ===")
    df    = load_and_clean_chess(matches_csv)
    n_raw = CSV.File(matches_csv) |> length

    # Step 3 – yearly leaders
    println("\n=== Step 3: Yearly leader Elo ===")
    leaders, py_max = compute_yearly_leaders(df)

    # Step 4 + 5 – save plots / CSV / report
    println("\n=== Step 4: Save outputs ===")
    rpath = save_chess_outputs(df, leaders, py_max, n_raw, out_dir)

    println("\n=== Done ===")
    println("Output: $out_dir")
    log_report(rpath)
end

# Only run main() when executed as a script, not when included from a notebook.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
