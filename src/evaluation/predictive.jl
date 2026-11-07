#=Shared posterior-predictive likelihood utilities for TrueSkill models.=#

module PredictiveLikelihoods

using Distributions
using LinearAlgebra
using Random
using Statistics

const TSG = Main.TrueSkill.Gaussian
const TSGamma = Main.TrueSkill.Gamma

abstract type AbstractPredictiveModel end

"""Algorithm used to integrate the approximate posterior predictive."""
abstract type AbstractPredictiveMethod end

"""One Gaussian moment-matching step, using the EP/VI posterior messages."""
struct MessageApproximation <: AbstractPredictiveMethod end

"""Deterministic Gauss--Laguerre integration over Gamma precision messages."""
struct NumericalQuadrature <: AbstractPredictiveMethod
    n_nodes::Int
end

"""Rao--Blackwellized Monte Carlo integration over skills and precisions."""
struct MonteCarlo <: AbstractPredictiveMethod
    n_samples::Int
    rng::AbstractRNG
end
MonteCarlo(n_samples::Int; seed::Int=42) = MonteCarlo(n_samples, MersenneTwister(seed))

"""Fully ancestral sampling of skills, precisions, performance, and outcome."""
struct AncestralMonteCarlo <: AbstractPredictiveMethod
    n_samples::Int
    rng::AbstractRNG
end
AncestralMonteCarlo(n_samples::Int; seed::Int=42) = AncestralMonteCarlo(n_samples, MersenneTwister(seed))

"""Vanilla TrueSkill's fixed Gaussian comparison-noise model."""
struct VanillaPredictive <: AbstractPredictiveModel
    beta::Float64
end

"""Robust TrueSkill with a fresh match precision integrated under its prior."""
struct RobustPredictive{G} <: AbstractPredictiveModel
    precision_prior::G
    noise_scale::Float64
    quadrature_points::Int
end

"""Consistent TrueSkill with fitted player precision posteriors."""
struct ConsistentPredictive <: AbstractPredictiveModel
    noise_scale::Float64
    quadrature_points::Int
end

"""TrueSkill draw margin matching an empirical draw probability."""
function draw_margin(beta::Real, draw_rate::Real)
    probability = clamp(draw_rate, 1e-6, 1 - 1e-6)
    return sqrt(2.0) * beta * quantile(Normal(), (probability + 1) / 2)
end

"""Win/draw/loss probabilities for a Gaussian performance difference."""
function outcome_probabilities(mean_difference, variance, epsilon)
    distribution = Normal(mean_difference, sqrt(max(variance, eps(Float64))))
    second = cdf(distribution, -epsilon)
    draw = mean_difference <= 0 ? ccdf(distribution, -epsilon)-ccdf(distribution, epsilon) :
        cdf(distribution, epsilon)-second
    return (first=ccdf(distribution, epsilon), draw=draw, second=second)
end

"""Nodes and normalized weights for expectations under a Gamma distribution."""
function gamma_quadrature(gamma, n_nodes::Int=16)
    n_nodes >= 2 || error("quadrature_points must be at least two")
    shape, rate = TSGamma.shape(gamma), TSGamma.rate(gamma)
    shape > 0 && rate > 0 || error("Gamma distribution must be proper")
    alpha = shape - 1
    decomposition = eigen(SymTridiagonal([2k - 1 + alpha for k in 1:n_nodes],
        [sqrt(k * (k + alpha)) for k in 1:(n_nodes - 1)]))
    return decomposition.values ./ rate, vec(abs2.(decomposition.vectors[1, :]))
end

"""Model-specific posterior-predictive outcome metrics.

`robust_precision` is the Gamma prior for a new match precision.  In the
consistent model, `consistent_precisions` contains the fitted player Gamma
posteriors.  The Gaussian skill posterior is integrated analytically and Gamma
variables are integrated with deterministic Gauss--Laguerre quadrature.
"""
function _skill_moments(posteriors, i::Int, j::Int)
    return TSG.mean(posteriors[i]) - TSG.mean(posteriors[j]),
        TSG.variance(posteriors[i]) + TSG.variance(posteriors[j])
end

_expected_inverse_precision(gamma) = TSGamma.expected_inverse_precision(gamma)

function _sample_skill_difference(rng, posteriors, i, j)
    return rand(rng, Normal(TSG.mean(posteriors[i]), sqrt(TSG.variance(posteriors[i])))) -
        rand(rng, Normal(TSG.mean(posteriors[j]), sqrt(TSG.variance(posteriors[j]))))
end

function _sample_precision(rng, gamma)
    return rand(rng, Gamma(TSGamma.shape(gamma), 1 / TSGamma.rate(gamma)))
end

function _mean_probabilities(samples)
    first = mean(p.first for p in samples)
    draw = mean(p.draw for p in samples)
    second = mean(p.second for p in samples)
    return (first=first, draw=draw, second=second)
end

function _ancestral_probabilities(rng, n_samples, posteriors, i, j, epsilon, noise_variance)
    wins = draws = losses = 0
    for _ in 1:n_samples
        difference = _sample_skill_difference(rng, posteriors, i, j) + sqrt(noise_variance()) * randn(rng)
        if difference > epsilon
            wins += 1
        elseif difference < -epsilon
            losses += 1
        else
            draws += 1
        end
    end
    return (first=wins / n_samples, draw=draws / n_samples, second=losses / n_samples)
end

# Message/moment approximation.  It keeps the Gaussian skill messages and
# replaces each precision message by E_q[B^{-1}], so it is fast but discards
# the non-Gaussian tails that robust/consistent TrueSkill were introduced for.
function predict_outcomes(::MessageApproximation, model::VanillaPredictive, posteriors, i::Int, j::Int, epsilon)
    mean_difference, skill_variance = _skill_moments(posteriors, i, j)
    return outcome_probabilities(mean_difference, skill_variance + 2model.beta^2, epsilon)
end
function predict_outcomes(::MessageApproximation, model::RobustPredictive, posteriors, i::Int, j::Int, epsilon)
    mean_difference, skill_variance = _skill_moments(posteriors, i, j)
    return outcome_probabilities(mean_difference, skill_variance + model.noise_scale * _expected_inverse_precision(model.precision_prior), epsilon)
end
function predict_outcomes(::MessageApproximation, model::ConsistentPredictive, posteriors, precisions, i::Int, j::Int, epsilon)
    mean_difference, skill_variance = _skill_moments(posteriors, i, j)
    return outcome_probabilities(mean_difference, skill_variance + model.noise_scale *
        (_expected_inverse_precision(precisions[i]) + _expected_inverse_precision(precisions[j])), epsilon)
end

function predict_outcomes(method::NumericalQuadrature, model::VanillaPredictive, posteriors, i::Int, j::Int, epsilon)
    return predict_outcomes(MessageApproximation(), model, posteriors, i, j, epsilon)
end
function predict_outcomes(method::NumericalQuadrature, model::RobustPredictive, posteriors, i::Int, j::Int, epsilon)
    mean_difference, skill_variance = _skill_moments(posteriors, i, j)
    first = tied = second = 0.0
    for (node, weight) in zip(gamma_quadrature(model.precision_prior, method.n_nodes)...)
        p = outcome_probabilities(mean_difference, skill_variance + model.noise_scale / node, epsilon)
        first += weight * p.first; tied += weight * p.draw; second += weight * p.second
    end
    return (first=first, draw=tied, second=second)
end
function predict_outcomes(method::NumericalQuadrature, model::ConsistentPredictive, skill_posteriors, precision_posteriors, i::Int, j::Int, epsilon)
    mean_difference, skill_variance = _skill_moments(skill_posteriors, i, j)
    nodes_i, weights_i = gamma_quadrature(precision_posteriors[i], method.n_nodes)
    nodes_j, weights_j = gamma_quadrature(precision_posteriors[j], method.n_nodes)
    first = tied = second = 0.0
    for (node_i, weight_i) in zip(nodes_i, weights_i), (node_j, weight_j) in zip(nodes_j, weights_j)
        p = outcome_probabilities(mean_difference, skill_variance + model.noise_scale / node_i + model.noise_scale / node_j, epsilon)
        weight = weight_i * weight_j
        first += weight * p.first; tied += weight * p.draw; second += weight * p.second
    end
    return (first=first, draw=tied, second=second)
end

function predict_outcomes(method::MonteCarlo, model::VanillaPredictive, posteriors, i::Int, j::Int, epsilon)
    return _mean_probabilities(outcome_probabilities(_sample_skill_difference(method.rng, posteriors, i, j), 2model.beta^2, epsilon) for _ in 1:method.n_samples)
end
function predict_outcomes(method::MonteCarlo, model::RobustPredictive, posteriors, i::Int, j::Int, epsilon)
    return _mean_probabilities(outcome_probabilities(_sample_skill_difference(method.rng, posteriors, i, j), model.noise_scale / _sample_precision(method.rng, model.precision_prior), epsilon) for _ in 1:method.n_samples)
end
function predict_outcomes(method::MonteCarlo, model::ConsistentPredictive, posteriors, precisions, i::Int, j::Int, epsilon)
    return _mean_probabilities(outcome_probabilities(_sample_skill_difference(method.rng, posteriors, i, j), model.noise_scale / _sample_precision(method.rng, precisions[i]) + model.noise_scale / _sample_precision(method.rng, precisions[j]), epsilon) for _ in 1:method.n_samples)
end

# In contrast with `MonteCarlo`, ancestral sampling draws the final noisy
# performance difference and estimates event probabilities by frequencies.
# It is unbiased but has strictly higher variance than the Rao--Blackwellized
# version above, particularly for rare draws or upsets.
function predict_outcomes(method::AncestralMonteCarlo, model::VanillaPredictive, posteriors, i::Int, j::Int, epsilon)
    return _ancestral_probabilities(method.rng, method.n_samples, posteriors, i, j, epsilon, () -> 2model.beta^2)
end
function predict_outcomes(method::AncestralMonteCarlo, model::RobustPredictive, posteriors, i::Int, j::Int, epsilon)
    return _ancestral_probabilities(method.rng, method.n_samples, posteriors, i, j, epsilon,
        () -> model.noise_scale / _sample_precision(method.rng, model.precision_prior))
end
function predict_outcomes(method::AncestralMonteCarlo, model::ConsistentPredictive, posteriors, precisions, i::Int, j::Int, epsilon)
    return _ancestral_probabilities(method.rng, method.n_samples, posteriors, i, j, epsilon,
        () -> model.noise_scale / _sample_precision(method.rng, precisions[i]) + model.noise_scale / _sample_precision(method.rng, precisions[j]))
end

function predict_outcomes(model::VanillaPredictive, posteriors, i::Int, j::Int, epsilon)
    return predict_outcomes(NumericalQuadrature(model isa VanillaPredictive ? 2 : model.quadrature_points), model, posteriors, i, j, epsilon)
end

function predict_outcomes(model::RobustPredictive, posteriors, i::Int, j::Int, epsilon)
    return predict_outcomes(NumericalQuadrature(model.quadrature_points), model, posteriors, i, j, epsilon)
end

function predict_outcomes(model::ConsistentPredictive, skill_posteriors, precision_posteriors,
    i::Int, j::Int, epsilon)
    return predict_outcomes(NumericalQuadrature(model.quadrature_points), model, skill_posteriors, precision_posteriors, i, j, epsilon)
end

function _aggregate_metrics(matches, draws, predict)
    losses, log_probabilities, correct_probabilities, correct_predictions = Float64[], Float64[], Float64[], Float64[]
    draw_correct, decisive_correct = Float64[], Float64[]
    predicted_draws = Float64[]
    for ((i, j), draw) in zip(matches, draws)
        p = predict(i, j)
        observed = draw ? p.draw : p.first
        log_probability = log(clamp(observed, 1e-12, 1.0))
        push!(losses, -log_probability); push!(log_probabilities, log_probability)
        push!(correct_probabilities, observed)
        probabilities = (p.first, p.draw, p.second)
        tied_classes = findall(==(maximum(probabilities)), probabilities)
        # Decisive inputs are winner-first. A fixed first-index tie break would
        # therefore use the observed outcome to resolve an otherwise fair tie.
        correct = (draw ? 2 : 1) in tied_classes ? 1.0/length(tied_classes) : 0.0
        push!(correct_predictions, correct)
        push!(predicted_draws, 2 in tied_classes ? 1.0/length(tied_classes) : 0.0)
        draw ? push!(draw_correct, correct) : push!(decisive_correct, correct)
    end
    safe_mean(values) = isempty(values) ? NaN : mean(values)
    return (nll=mean(losses), mean_log_probability=mean(log_probabilities),
        mean_correct_probability=mean(correct_probabilities), accuracy=mean(correct_predictions),
        decisive_accuracy=safe_mean(decisive_correct), draw_accuracy=safe_mean(draw_correct),
        predicted_draw_rate=mean(predicted_draws))
end

function model_metrics(model::Union{VanillaPredictive, RobustPredictive}, skill_posteriors,
    matches, draws, epsilon)
    cache = Dict{Tuple{Int,Int},NamedTuple{(:first,:draw,:second),Tuple{Float64,Float64,Float64}}}()
    return _aggregate_metrics(matches, draws,
        (i, j) -> get!(() -> predict_outcomes(model, skill_posteriors, i, j, epsilon), cache, (i,j)))
end

function model_metrics(model::ConsistentPredictive, skill_posteriors, precision_posteriors,
    matches, draws, epsilon)
    cache = Dict{Tuple{Int,Int},NamedTuple{(:first,:draw,:second),Tuple{Float64,Float64,Float64}}}()
    return _aggregate_metrics(matches, draws,
        (i, j) -> get!(() -> predict_outcomes(model, skill_posteriors, precision_posteriors, i, j, epsilon), cache, (i,j)))
end

end # module
