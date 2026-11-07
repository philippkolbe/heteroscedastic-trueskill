using Test, Distributions, Random, CSV, DataFrames
include(joinpath(@__DIR__, "..", "src", "trueskill.jl"))
include(joinpath(@__DIR__, "..", "src", "evaluation", "predictive.jl"))
const G = TrueSkill.Gaussian
const Ga = TrueSkill.Gamma
const DC = TrueSkill.DistributionCollections
const PL = PredictiveLikelihoods

@testset "Truncated Gaussian draw update" begin
    # Compare the implemented variance reduction with the distribution's moments.
    mu, sigma, lo, hi = 0.4, 1.3, -0.5, 0.5
    reference = truncated(Normal(mu, sigma), lo, hi)
    @test TrueSkill.MyMath.v_lu(mu, sigma, lo, hi) ≈ (mean(reference)-mu)/sigma
    @test TrueSkill.MyMath.w_lu(mu, sigma, lo, hi) ≈ 1-var(reference)/sigma^2
end

@testset "Inference and predictive semantics" begin
    prior = G.Gaussian1DFromMeanVariance(0.0, 9.0)
    matches = repeat([(Int64(3), Int64(2)), (Int64(2), Int64(1)), (Int64(3), Int64(1))], 12)
    draws = falses(length(matches))
    skills = TrueSkill.vanilla_trueskill(DC.DistributionBag(prior, 3), 1.0, matches, 0.0)
    @test issorted(G.mean.(skills))
    @test all(x -> isfinite(G.variance(x)) && G.variance(x)>0, skills)
    precision = Ga.Gamma1D(1.0, 2.0)
    @test Ga.shape(precision) == 2.0
    @test Ga.mean(precision) == 1.0
    # A decisive record is winner-first. Exact predictive ties must not be
    # resolved in favor of that known winner during evaluation.
    unfitted=DC.DistributionBag(prior,2)
    tied_metrics=PL.model_metrics(PL.VanillaPredictive(1.0),unfitted,[(Int64(1),Int64(2))],falses(1),0.0)
    @test tied_metrics.accuracy == 0.5
    robust, _ = TrueSkill.robust_trueskill(DC.DistributionBag(prior, 3),
        DC.DistributionBag(precision, length(matches)), matches, 0.0; s2=1.0)
    consistent, precisions = TrueSkill.consistent_trueskill(DC.DistributionBag(prior, 3),
        DC.DistributionBag(precision, 3), matches, 0.0; s2=1.0)
    for (model, posterior, b) in ((PL.VanillaPredictive(1.0), skills, nothing),
        (PL.RobustPredictive(precision, 1.0, 16), robust, nothing),
        (PL.ConsistentPredictive(1.0, 16), consistent, precisions))
        p = b === nothing ? PL.predict_outcomes(model, posterior, 1, 3, 0.3) :
            PL.predict_outcomes(model, posterior, b, 1, 3, 0.3)
        @test all(x -> isfinite(x) && 0<=x<=1, (p.first, p.draw, p.second))
        @test p.first+p.draw+p.second ≈ 1.0
        @test p.second > p.first
        metrics = b === nothing ? PL.model_metrics(model, posterior, matches, draws, 0.3) :
            PL.model_metrics(model, posterior, b, matches, draws, 0.3)
        @test isfinite(metrics.nll) && metrics.nll >= 0
    end
end

include(joinpath(@__DIR__, "..", "src", "datasets", "arena.jl"))
@testset "Preference input and session separation" begin
    battles = DataFrame(model_a=["a", "a", "b", "b"], model_b=["b", "b", "a", "a"],
        winner=["model_a", "model_b", "tie", "both_bad"])
    _, _, matches, draws = _index_battles(battles)
    @test matches == [(1,2),(2,1),(2,1),(2,1)]
    @test draws == [false,false,true,true]
    grouped = DataFrame(group_id=repeat(string.(1:20), inner=3))
    @test _prepare_battle_split(grouped).group_id == grouped.group_id
    train, validation, test = _split_battle_indices(grouped, .7, .15)
    a,b,c = (Set(grouped.group_id[i]) for i in (train,validation,test))
    @test isempty(intersect(a,b)) && isempty(intersect(a,c)) && isempty(intersect(b,c))
    @test sort(vcat(train, validation, test)) == collect(1:nrow(grouped))
    mktempdir() do directory
        battles.evaluation_session_id = ["one","one","two","two"]
        path = joinpath(directory,"preferences.csv")
        CSV.write(path,battles)
        loaded = _load_battles(path;min_games=1,download=false)
        @test loaded.group_id == battles.evaluation_session_id
        @test_throws ErrorException _load_battles(joinpath(directory,"missing.csv");download=false)
    end
end

module ChessPreprocessing
include(joinpath(@__DIR__, "..", "experiments", "chess", "prepare.jl"))
end
@testset "ChessBase text preprocessing" begin
    mktempdir() do output
        input = joinpath(output,"fictional.txt")
        write(input, "Fictional A (1500) - Fictional B (1600) [A00]\nInvented Event 01.01.2023\n1. e4 e5 1-0\n\nFictional B (1600) - Fictional A (1500) [A00]\nInvented Event 02.01.2023\n1. d4 d5 1/2-1/2\n")
        matches,players = joinpath(output,"matches.csv"),joinpath(output,"players.csv")
        ChessPreprocessing.parse_chess_file(input,matches,players,joinpath(output,"stats.txt"),joinpath(output,"edges.log"))
        table = DataFrame(CSV.File(matches))
        @test nrow(table)==2
        @test table.is_draw == [false,true]
        @test nrow(DataFrame(CSV.File(players)))==2
    end
end

include(joinpath(@__DIR__, "..", "experiments", "synthetic", "run.jl"))
@testset "Paper synthetic entry point" begin
    mktempdir() do output
        rows = PaperSynthetic.experiment_corruption_robustness(k=8, n=120, seed=123,
            deltas=[0.0, 0.2], output_dir=output)
        @test nrow(rows) == 12
        @test all(isfinite, rows.kendall_tau)
        @test all(isfile, joinpath.(output, ["metrics.csv", "corruption.png", "corruption.pdf"]))
    end
end

module ChessFixture
include(joinpath(@__DIR__, "..", "src", "datasets", "chess.jl"))
end
@testset "Chess upset denominator" begin
    matches = DataFrame(white_name=fill("a", 5), black_name=fill("b", 5),
        white_elo=[1500, 1500, 1600, 1700, 1500], black_elo=fill(1600, 5),
        winner_name=["a", "b", "a", "a", ""], is_draw=[false, false, false, false, true])
    stats = ChessFixture._player_period_stats(matches)
    @test (stats["a"].underdog_wins, stats["a"].underdog_games) == (1, 2)
    @test (stats["b"].underdog_wins, stats["b"].underdog_games) == (0, 1)
    @test (stats["a"].favorite_losses, stats["b"].favorite_losses) == (0, 1)
    @test stats["a"].comparable_games == stats["b"].comparable_games == 3
    @test stats["a"].pct_upsets ≈ 100 / 3
    @test stats["b"].pct_upsets ≈ 100 / 3
end

@testset "User-supplied chess CSV end to end" begin
    mktempdir() do output
        # Invented players and games; no licensed records are used by this check.
        rows = DataFrame(year=Int[], date=String[], white_name=String[], black_name=String[],
            winner_name=Union{Missing,String}[], is_draw=Bool[], white_elo=Int[], black_elo=Int[])
        for repetition in 1:5, i in 1:8, j in i+1:8
            draw = (i+j+repetition)%7 == 0
            push!(rows, (2023,"2023.01.01","fictional_$i","fictional_$j",draw ? missing : "fictional_$j",draw,1500,1500))
        end
        matches, players = joinpath(output,"matches.csv"),joinpath(output,"players.csv")
        CSV.write(matches, rows)
        CSV.write(players, DataFrame(name=["fictional_$i" for i in 1:8],max_elo=fill(1500,8)))
        result = ChessFixture.run_chess_experiment(matches_path=matches,players_path=players,
            output_dir=joinpath(output,"fit"),top_n=5,outer_max_iters=20,inner_max_iters=20,verbose=false)
        @test length(result.consistent_s) == 8
        @test all(isfinite, ChessFixture.TSG.mean.(result.consistent_s))
        @test isfile(joinpath(output,"fit","top5_consistent_skill_and_noise.png"))
    end
end
