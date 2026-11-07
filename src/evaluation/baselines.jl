module RatingBaselines

using Statistics

export fit_elo, fit_bradley_terry, elo_probability

"""Sequential Elo ratings. Draws count as half a win for each player."""
function fit_elo(matches::Vector{Tuple{Int64, Int64}}, draws::BitVector, n_players::Int;
    initial_rating::Float64=1000.0, k_factor::Float64=32.0)
    ratings = fill(initial_rating, n_players)
    for ((winner, loser), draw) in zip(matches, draws)
        probability = elo_probability(ratings[winner], ratings[loser])
        delta = k_factor * ((draw ? 0.5 : 1.0) - probability)
        ratings[winner] += delta
        ratings[loser] -= delta
    end
    return ratings
end

elo_probability(first_rating::Real, second_rating::Real) =
    1.0 / (1.0 + 10.0^((second_rating - first_rating) / 400.0))

"""Bradley--Terry MLE via the standard minorization--maximization iteration.

The returned values use the conventional Elo display scale. Draws contribute
half a win to each player.
"""
function fit_bradley_terry(matches::Vector{Tuple{Int64, Int64}}, draws::BitVector, n_players::Int;
    anchor::Float64=1000.0, tol::Float64=1e-9, max_iters::Int=1_000)
    wins = zeros(Float64, n_players)
    games = zeros(Float64, n_players, n_players)
    for ((winner, loser), draw) in zip(matches, draws)
        wins[winner] += draw ? 0.5 : 1.0
        draw && (wins[loser] += 0.5)
        games[winner, loser] += 1.0
        games[loser, winner] += 1.0
    end
    ability = ones(Float64, n_players)
    for _ in 1:max_iters
        updated = similar(ability)
        for i in 1:n_players
            denominator = sum(games[i, j] == 0 ? 0.0 : games[i, j] / (ability[i] + ability[j])
                              for j in 1:n_players)
            updated[i] = denominator == 0.0 ? ability[i] : wins[i] / denominator
        end
        updated .= max.(updated, eps(Float64))
        updated ./= exp(mean(log.(updated)))
        maximum(abs.(log.(updated ./ ability))) < tol && (ability = updated; break)
        ability = updated
    end
    return anchor .+ 400.0 / log(10.0) .* log.(ability)
end

end
