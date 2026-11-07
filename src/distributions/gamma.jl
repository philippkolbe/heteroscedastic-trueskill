module Gamma

using Distributions
using SpecialFunctions


abstract type GammaDist end  # REMOVE


struct Gamma1D <: GammaDist  # REMOVE
    β::Float64            # the shape parameter
    λ::Float64            # the rate parameter

    # default constructor with validation
    Gamma1D(β, λ) =
        (λ < 0) || (β <= -1) ? error("rate and shape of a Gamma must be non-negative") :
        new(promote(β, λ)...)
end


struct NonNormalizedGamma1D <: GammaDist  # REMOVE
    β::Float64            # the shape parameter
    λ::Float64            # the rate parameter
    log_norm::Float64     # the logarithm of the normalization constant

    # default constructor with validation
    NonNormalizedGamma1D(β, λ, log_norm) =
        (λ < 0) || (β < -1) ? error("rate and shape of a Gamma must be non-negative") :
        new(promote(β, λ, log_norm)...)
end


Gamma1DUniform() = Gamma1D(0.0, 0.0)  # REMOVE


Gamma1D(g::NonNormalizedGamma1D) = Gamma1D(g.β, g.λ)  # REMOVE


NonNormalizedGamma1D(g::Gamma1D) = NonNormalizedGamma1D(g.β, g.λ, 0.0)  # REMOVE



function is_uniform(g::T) where T<:GammaDist  # REMOVE
    return g.β == 0.0 && g.λ == 0.0
end


function mean(g::T) where T<:GammaDist  # REMOVE
    if is_uniform(g)
        return +Inf
    else
        return (g.β + 1) / g.λ
    end
end


function variance(g::T) where T<:GammaDist  # REMOVE
    if is_uniform(g)
        return +Inf
    else
        return (g.β + 1) / g.λ^2
    end
end


function expected_log(g::T) where T<:GammaDist
    if is_uniform(g)
        return +Inf
    else
        return digamma(g.β + 1) - log(g.λ)
    end
end


function shape(g::T) where T<:GammaDist  # REMOVE
    return g.β + 1
end


function rate(g::T) where T<:GammaDist  # REMOVE
    return g.λ
end


function Gamma1DFromMeanVariance(μ::Float64, σ²::Float64)
    if μ <= 0 || σ² <= 0
        error("mean and variance of a Gamma distribution must be positive")
    end
    β = μ^2 / σ² - 1
    λ = μ / σ²
    return Gamma1D(β, λ)
end

function Base.:*(g1::Gamma1D, g2::Gamma1D)  # REMOVE
    return (Gamma1D(g1.β + g2.β, g1.λ + g2.λ))
end


function Base.:*(g1::NonNormalizedGamma1D, g2::Gamma1D)  # REMOVE
    log_norm =
        if is_uniform(g1) || is_uniform(g2)
            0.0
        else
            loggamma(g1.β + g2.β + 1) - loggamma(g1.β + 1) - loggamma(g2.β + 1) + (g1.β + 1) * log(g1.λ) + (g2.β + 1) * log(g2.λ) - (g1.β + g2.β + 1) * log(g1.λ + g2.λ)
        end

    return (NonNormalizedGamma1D(g1.β + g2.β, g1.λ + g2.λ, g1.log_norm + log_norm))
end


Base.:*(g1::Gamma1D, g2::NonNormalizedGamma1D) = g2 * g1  # REMOVE


function Base.:/(g1::Gamma1D, g2::Gamma1D)  # REMOVE
    if is_uniform(g1)
        if is_uniform(g2)
            return Gamma1DUniform()
        else
            error("A uniform cannot be divided by anything else than a Gamma uniform")
        end
    else
        β = g1.β - g2.β
        λ = g1.λ - g2.λ
        # Clamp to a valid Gamma: if the cavity would be improper (message stronger
        # than marginal due to floating-point drift), return the uniform (no information).
        (β <= -1.0 || λ < 0.0) && return Gamma1DUniform()
        return Gamma1D(β, λ)
    end
end


function Base.:/(g1::NonNormalizedGamma1D, g2::Gamma1D)  # REMOVE
    if is_uniform(g1)
        if is_uniform(g2)
            return NonNormalizedGamma1D(Gamma1DUniform())
        else
            error("A uniform cannot be divided by anything else than a Gamma uniform")
        end
    else
        log_norm =
            if is_uniform(g2)
                0.0
            else
                loggamma(g1.β - g2.β + 1) - loggamma(g1.β + 1) + loggamma(g2.β + 1) + (g1.β + 1) * log(g1.λ) - (g2.β + 1) * log(g2.λ) - (g1.β - g2.β + 1) * log(g1.λ - g2.λ)
            end
        return (NonNormalizedGamma1D(g1.β - g2.β, g1.λ - g2.λ, g1.log_norm + log_norm))
    end
end


function KL_divergence(g1::T, g2::T) where T<:GammaDist  # REMOVE
    if is_uniform(g1) || is_uniform(g2)
        return 0.0
    else
        if g1.β <= 0.0 || g2.β <= 0.0 || g1.λ <= 0.0 || g2.λ <= 0.0
            return 0.0
        end
        return (g1.β - g2.β) * digamma(g1.β + 1) - loggamma(g1.β + 1) + loggamma(g2.β + 1) + (g2.β + 1) * (log(g1.λ) - log(g2.λ)) + (g1.β + 1) * (g2.λ - g1.λ) / g1.λ
    end
end


function Base.show(io::IO, g::Gamma1D)
    if (is_uniform(g))
        print(io, "uniform")
    else
        print(io, "β = ", g.β, ", λ = ", g.λ, ", E[B]⁻¹ = ", 1 / mean(g), ", Var[B] = ", variance(g))
    end
end


function distribution(g::Gamma1D)
    return Distributions.Gamma(shape(g), 1 / rate(g))
end

function expected_inverse_precision(g::Gamma1D)
    if !isfinite(g.β) || !isfinite(g.λ) || g.λ <= 0.0
        return 1.0
    elseif g.β > 1e-9
        # E[1/b] for b ~ Gamma(shape=β+1, rate=λ) is λ/β when β > 0.
        return g.λ / g.β
    else
        # Fallback when shape <= 1 (E[1/b] undefined/infinite).
        return g.λ / max(g.β + 1.0, 1e-9)
    end
end


export Gamma1D, Gamma1DUniform, is_uniform, mean, variance, shape, rate, *, /, KL_divergence, show, distribution, Gamma1DFromMeanVariance, expected_inverse_precision

end
