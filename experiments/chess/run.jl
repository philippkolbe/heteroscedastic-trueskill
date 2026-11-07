#=Descriptive chess fit used by the submitted paper. Data must be supplied locally.=#
include(joinpath(@__DIR__, "..", "..", "src", "datasets", "chess.jl"))
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (2, 4) || error("Usage: chess.jl MATCHES.csv PLAYERS.csv [FIRST_YEAR LAST_YEAR]")
    years = length(ARGS) == 4 ? (parse(Int, ARGS[3]):parse(Int, ARGS[4])) : (2022:2024)
    run_chess_experiment(matches_path=abspath(ARGS[1]), players_path=abspath(ARGS[2]),
        years=years, top_n=10, output_dir=joinpath(@__DIR__, "..", "..", "output", "chess", "$(first(years))_$(last(years))"))
end
