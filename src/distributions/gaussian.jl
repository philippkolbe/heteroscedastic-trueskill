module Gaussian

using Distributions
import Distributions: Normal

abstract type GaussianDist end

struct Gaussian1D <: GaussianDist
    τ::Float64
    ρ::Float64

    # default constructor with validation
    function Gaussian1D(τ, ρ)
        τp, ρp = promote(τ, ρ)

        # Keep an explicit representation for the true uniform prior.
        if τp == 0.0 && ρp == 0.0
            return new(0.0, 0.0)
        end

        # EP updates subtract natural parameters and may produce tiny negative
        # precision from floating-point cancellation.
        tol = 1e-12 * max(1.0, abs(τp), abs(ρp))

        if ρp < -tol
            error("precision of a Gaussian must be non-negative")
        elseif ρp <= tol
            # Project to the nearest proper Gaussian by enforcing minimal
            # positive precision rather than collapsing to uniform.
            return new(τp, tol)
        else
            return new(τp, ρp)
        end
    end
end

Gaussian1DFromMeanVariance(μ, σ2) =
    (σ2 == 0.0) ? error("Dirac delta cannot be initialized with GaussianFromMeanVariance") :
    Gaussian1D(μ / σ2, 1 / σ2)


Gaussian1DUniform() = Gaussian1D(0.0, 0.0)


function mean(g::T) where T<:GaussianDist
    if g.ρ == 0.0   # Check if uniform
        return 0.0
    else
        return g.τ / g.ρ
    end
end


function variance(g::T) where T<:GaussianDist  # REMOVE FUNCTION
    if is_uniform(g)
        return +Inf
    else
        return 1.0 / g.ρ
    end
end


function Base.:*(g1::Gaussian1D, g2::Gaussian1D)
    return (Gaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ))
end

function Base.:/(g1::Gaussian1D, g2::Gaussian1D)
    τ = g1.τ - g2.τ
    ρ = g1.ρ - g2.ρ
    min_ρ = 1e-12 * max(1.0, abs(g1.τ), abs(g1.ρ), abs(g2.τ), abs(g2.ρ))

    # EP cavity/message subtraction can yield invalid precision when a message
    # temporarily overpowers the marginal. Project to the nearest proper Gaussian.
    if ρ <= min_ρ
        return Gaussian1D(τ, min_ρ)
    end
    return Gaussian1D(τ, ρ)
end


function KL_divergence(g1::T, g2::T) where T<:GaussianDist  # REMOVE and adjust case where both are uniform
    if is_uniform(g1) && is_uniform(g2)
        return 0.0
    elseif is_uniform(g1) || is_uniform(g2)
        return Inf
    else
        μ_diff = mean(g1) - mean(g2)
        σ1_2 = variance(g1)
        σ2_2 = variance(g2)
        if σ1_2 <= 0.0 || σ2_2 <= 0.0 || !isfinite(σ1_2) || !isfinite(σ2_2)
            return 0.0
        end
        return 0.5 * (log(σ2_2 / σ1_2) + (σ1_2 + μ_diff * μ_diff) / σ2_2 - 1)
    end
end


function is_uniform(g::T) where T<:GaussianDist  #REMOVE
    return g.ρ == 0.0
end



function Base.show(io::IO, g::Gaussian1D)
    if (is_uniform(g))
        print(io, "Uniform()")
    else
        print(io, "N(μ = ", mean(g), ", σ² = ", variance(g), ") resp. G(τ = ", g.τ, ", ρ = ", g.ρ, ")")
    end
end



function Base.show(io::IO, g::Gaussian1D)
    if (is_uniform(g))
        print(io, "Uniform()")
    else
        print(io, "N(μ = ", mean(g), ", σ² = ", variance(g),") resp. G(τ = ", g.τ, ", ρ = ", g.ρ, ")")
    end
end


function Normal(g::Gaussian1D)
    if is_uniform(g)
        return Distributions.Uniform(-Inf, Inf)
    else
        return Distributions.Normal(mean(g), sqrt(variance(g)))
    end
end


function win_prob(g1::Gaussian1D, g2::Gaussian1D)::Float64
    μ_diff = mean(g1) - mean(g2)
    σ2_diff = variance(g1) + variance(g2)
    if !isfinite(σ2_diff) || σ2_diff <= 0.0
        return 0.5
    end
    return Distributions.cdf(Distributions.Normal(0, 1), μ_diff / sqrt(σ2_diff))
end


export Gaussian1D, Gaussian1DFromMeanVariance, Gaussian1DUniform, is_uniform, Normal
export mean, variance, *, /, KL_divergence, show, distribution, win_prob

end
