module Factors

import Distributions
using SpecialFunctions: digamma, trigamma

using ..DistributionCollections
using ..Gaussian
using ..Gamma
using ..MyMath
using LinearAlgebra: dot

export Factor, GaussianFactor, GaussianMeanFactor, GammaFactor, GaussianMeanFactorWithNoise, DifferenceFactor
export TruncationFactorWin, TruncationFactorDraw, TruncationFactorLoss
export GaussianObservationFactor
export update_msg_to_s!, update_msg_to_p!, update_msg_to_x!, update_msg_to_y!, update_msg_to_z!, update_msg_to_b!, update_msg_to_m!, update_msg_to_d!


abstract type Factor end

const GaussianBag = DistributionBag{Gaussian1D}
const GammaBag = DistributionBag{Gamma1D}

struct GaussianFactor <: Factor
    db_marginal::GaussianBag
    db_msg::GaussianBag
    s::Int64
    prior::Gaussian1D
    msg_to_s::Int64
end

GaussianFactor(db_marginal::GaussianBag, db_msg::GaussianBag, s::Int64, prior::Gaussian1D) = GaussianFactor(db_marginal, db_msg, s, prior, add!(db_msg))
GaussianFactor(db::GaussianBag, s::Int64, prior::Gaussian1D) = GaussianFactor(db, db, s, prior, add!(db))

function update_msg_to_s!(f::GaussianFactor)
    old_marginal = f.db_marginal[f.s]

    msg_s_to_f_dist = old_marginal / f.db_msg[f.msg_to_s]
    new_marginal = msg_s_to_f_dist * f.prior

    f.db_msg[f.msg_to_s] = f.prior
    f.db_marginal[f.s] = new_marginal
    return Gaussian.KL_divergence(old_marginal, new_marginal)
end


struct GammaFactor <: Factor
    db_marginal::GammaBag
    db_msg::GammaBag
    β::Int64
    prior::Gamma1D
    msg_to_β::Int64
end

GammaFactor(db_marginal::GammaBag, db_msg::GammaBag, β::Int64, prior::Gamma1D) = GammaFactor(db_marginal, db_msg, β, prior, add!(db_msg))
GammaFactor(db::GammaBag, β::Int64, prior::Gamma1D) = GammaFactor(db, db, β, prior, add!(db))

function update_msg_to_b!(f::GammaFactor)
    old_marginal = f.db_marginal[f.β]

    msg_β_to_f_dist = old_marginal / f.db_msg[f.msg_to_β]
    new_marginal = msg_β_to_f_dist * f.prior

    f.db_msg[f.msg_to_β] = f.prior
    f.db_marginal[f.β] = new_marginal
    return Gamma.KL_divergence(old_marginal, new_marginal)
end

struct GaussianMeanFactor <: Factor
    db_marginal::GaussianBag
    db_msg::GaussianBag
    s::Int64
    p::Int64
    β²::Float64
    msg_to_s::Int64
    msg_to_p::Int64
end

GaussianMeanFactor(db_marginal::GaussianBag, db_msg::GaussianBag, s::Int64, p::Int64, β²::Float64) = GaussianMeanFactor(db_marginal, db_msg, s, p, β², add!(db_msg), add!(db_msg))
GaussianMeanFactor(db::GaussianBag, s::Int64, p::Int64, β²::Float64) = GaussianMeanFactor(db, db, s, p, β², add!(db), add!(db))

function update_msg_to_s!(f::GaussianMeanFactor)
    old_marginal = f.db_marginal[f.s]
    msg_s_to_f_dist = old_marginal / f.db_msg[f.msg_to_s]
    marginal_p = f.db_marginal[f.p]
    msg_p_to_f_dist = marginal_p / f.db_msg[f.msg_to_p]

    μ = Gaussian.mean(msg_p_to_f_dist)
    σ² = Gaussian.variance(msg_p_to_f_dist)
    msg_f_to_s_dist = Gaussian1DFromMeanVariance(μ, σ² + f.β²)
    new_marginal = msg_s_to_f_dist * msg_f_to_s_dist

    f.db_msg[f.msg_to_s] = msg_f_to_s_dist
    f.db_marginal[f.s] = new_marginal

    return Gaussian.KL_divergence(old_marginal, new_marginal)
end

function update_msg_to_p!(f::GaussianMeanFactor)
    old_marginal = f.db_marginal[f.p]
    msg_p_to_f_dist = old_marginal / f.db_msg[f.msg_to_p]

    marginal_s = f.db_marginal[f.s]
    msg_s_to_f_dist = marginal_s / f.db_msg[f.msg_to_s]
    μ = Gaussian.mean(msg_s_to_f_dist)
    σ² = Gaussian.variance(msg_s_to_f_dist)

    msg_f_to_p_dist = Gaussian1DFromMeanVariance(μ, σ² + f.β²)
    new_marginal = msg_p_to_f_dist * msg_f_to_p_dist

    f.db_msg[f.msg_to_p] = msg_f_to_p_dist
    f.db_marginal[f.p] = new_marginal

    return Gaussian.KL_divergence(old_marginal, new_marginal)
end

# Factor for the Student-t scale-mixture noise model:
#   x | m, omega ~ N(m, s2/omega),  omega ~ Gamma(nu/2, nu/2)
# where s2 is the noise scale and omega is the relative precision (E[omega]=1).
struct GaussianMeanFactorWithNoise <: Factor
  db_gauss_marginal::GaussianBag
  db_gauss_msg::GaussianBag
  db_gamma_marginal::GammaBag
  db_gamma_msg::GammaBag
  s::Int64        # skill (or skill-difference) node
  p::Int64        # noisy observation node
  b::Int64        # relative precision omega node
  msg_to_s::Int64
  msg_to_p::Int64
  msg_to_b::Int64
  s2::Float64     # noise scale: Var(x | m, omega) = s2/omega
end

function GaussianMeanFactorWithNoise(db_gauss_marginal, db_gauss_msg, db_gamma_marginal, db_gamma_msg, s, p, b, s2::Float64=1.0)
  return GaussianMeanFactorWithNoise(
    db_gauss_marginal,
    db_gauss_msg,
    db_gamma_marginal,
    db_gamma_msg,
    s,
    p,
    b,
    add!(db_gauss_msg),
    add!(db_gauss_msg),
    add!(db_gamma_msg),
    s2
  )
end

GaussianMeanFactorWithNoise(db_gauss::GaussianBag, db_gamma::GammaBag, s::Int64, p::Int64, b::Int64; s2::Float64=1.0) =
  GaussianMeanFactorWithNoise(db_gauss, db_gauss, db_gamma, db_gamma, s, p, b, s2)

function update_msg_to_s!(f::GaussianMeanFactorWithNoise)
  μ_s = Gamma.mean(f.db_gamma_marginal[f.b]) / f.s2  # E[omega]/s2 = effective precision
  μ_y = Gaussian.mean(f.db_gauss_marginal[f.p])
  new_msg_to_s = Gaussian1D(μ_s * μ_y, μ_s)

  old_marginal_s = f.db_gauss_marginal[f.s]
  msg_from_s = old_marginal_s / f.db_gauss_msg[f.msg_to_s]
  new_marginal_s = msg_from_s * new_msg_to_s
  f.db_gauss_marginal[f.s] = new_marginal_s
  f.db_gauss_msg[f.msg_to_s] = new_msg_to_s
  return Gaussian.KL_divergence(old_marginal_s, new_marginal_s)
end

function update_msg_to_p!(f::GaussianMeanFactorWithNoise)
  μ_s = Gamma.mean(f.db_gamma_marginal[f.b]) / f.s2  # E[omega]/s2 = effective precision
  μ_x = Gaussian.mean(f.db_gauss_marginal[f.s])
  new_msg_to_p = Gaussian1D(μ_s * μ_x, μ_s)

  old_marginal_p = f.db_gauss_marginal[f.p]
  msg_from_p = old_marginal_p / f.db_gauss_msg[f.msg_to_p]
  new_marginal_p = msg_from_p * new_msg_to_p
  f.db_gauss_marginal[f.p] = new_marginal_p
  f.db_gauss_msg[f.msg_to_p] = new_msg_to_p
  return Gaussian.KL_divergence(old_marginal_p, new_marginal_p)
end

function update_msg_to_b!(f::GaussianMeanFactorWithNoise)
  μ_x = Gaussian.mean(f.db_gauss_marginal[f.s])
  μ2_x = Gaussian.variance(f.db_gauss_marginal[f.s]) + μ_x^2
  μ_y = Gaussian.mean(f.db_gauss_marginal[f.p])
  μ2_y = Gaussian.variance(f.db_gauss_marginal[f.p]) + μ_y^2
  # Likelihood message to omega: shape contrib = 1/2, rate = E[r²]/(2*s2)
  new_msg_to_b = Gamma1D(1/2, max((μ2_x + μ2_y) / 2 - μ_x * μ_y, 0.0) / f.s2)

  old_marginal_b = f.db_gamma_marginal[f.b]
  msg_from_b = old_marginal_b / f.db_gamma_msg[f.msg_to_b]
  new_marginal_b = msg_from_b * new_msg_to_b
  f.db_gamma_marginal[f.b] = new_marginal_b
  f.db_gamma_msg[f.msg_to_b] = new_msg_to_b
  return Gamma.KL_divergence(old_marginal_b, new_marginal_b)
end


struct WeightedSumFactor <: Factor
    db_marginal::GaussianBag
    db_msg::GaussianBag
    x::Int64
    y::Int64
    z::Int64
    a::Float64
    b::Float64
    msg_to_x::Int64
    msg_to_y::Int64
    msg_to_z::Int64
end

DifferenceFactor(db_marginal::GaussianBag, db_msg::GaussianBag, x::Int64, y::Int64, z::Int64) = WeightedSumFactor(db_marginal, db_msg, x, y, z, 1.0, -1.0, add!(db_msg), add!(db_msg), add!(db_msg))
DifferenceFactor(db::GaussianBag, x::Int64, y::Int64, z::Int64) = WeightedSumFactor(db, db, x, y, z, 1.0, -1.0, add!(db), add!(db), add!(db))

function update_msg_to_x!(f::WeightedSumFactor)
    old_marginal = f.db_marginal[f.x]
    msg_x_to_f_dist = old_marginal / f.db_msg[f.msg_to_x]

    marginal_y = f.db_marginal[f.y]
    msg_y_to_f_dist = marginal_y / f.db_msg[f.msg_to_y]

    marginal_z = f.db_marginal[f.z]
    msg_z_to_f_dist = marginal_z / f.db_msg[f.msg_to_z]

    μ = (Gaussian.mean(msg_z_to_f_dist) - f.b * Gaussian.mean(msg_y_to_f_dist)) / f.a
    σ2 = (Gaussian.variance(msg_z_to_f_dist) + f.b^2 * Gaussian.variance(msg_y_to_f_dist)) / f.a^2

    msg_f_to_x_dist = Gaussian1DFromMeanVariance(μ, σ2)
    new_marginal = msg_x_to_f_dist * msg_f_to_x_dist

    f.db_msg[f.msg_to_x] = msg_f_to_x_dist
    f.db_marginal[f.x] = new_marginal

    return Gaussian.KL_divergence(old_marginal, new_marginal)
end

function update_msg_to_y!(f::WeightedSumFactor)
    old_marginal = f.db_marginal[f.y]

    marginal_x = f.db_marginal[f.x]
    msg_x_to_f_dist = marginal_x / f.db_msg[f.msg_to_x]

    marginal_y = f.db_marginal[f.y]
    msg_y_to_f_dist = marginal_y / f.db_msg[f.msg_to_y]

    marginal_z = f.db_marginal[f.z]
    msg_z_to_f_dist = marginal_z / f.db_msg[f.msg_to_z]

    μ = (Gaussian.mean(msg_z_to_f_dist) - f.a * Gaussian.mean(msg_x_to_f_dist)) / f.b
    σ2 = (Gaussian.variance(msg_z_to_f_dist) + f.a^2 * Gaussian.variance(msg_x_to_f_dist)) / f.b^2

    msg_f_to_y_dist = Gaussian1DFromMeanVariance(μ, σ2)
    new_marginal = msg_y_to_f_dist * msg_f_to_y_dist

    f.db_msg[f.msg_to_y] = msg_f_to_y_dist
    f.db_marginal[f.y] = new_marginal

    return Gaussian.KL_divergence(old_marginal, new_marginal)
end

function update_msg_to_z!(f::WeightedSumFactor)
    old_marginal = f.db_marginal[f.z]
    msg_z_to_f_dist = old_marginal / f.db_msg[f.msg_to_z]

    marginal_x = f.db_marginal[f.x]
    msg_x_to_f_dist = marginal_x / f.db_msg[f.msg_to_x]

    marginal_y = f.db_marginal[f.y]
    msg_y_to_f_dist = marginal_y / f.db_msg[f.msg_to_y]

    μ = f.a * Gaussian.mean(msg_x_to_f_dist) + f.b * Gaussian.mean(msg_y_to_f_dist)
    σ2 = f.a^2 * Gaussian.variance(msg_x_to_f_dist) + f.b^2 * Gaussian.variance(msg_y_to_f_dist)

    msg_f_to_z_dist = Gaussian1DFromMeanVariance(μ, σ2)
    new_marginal = msg_z_to_f_dist * msg_f_to_z_dist

    f.db_msg[f.msg_to_z] = msg_f_to_z_dist
    f.db_marginal[f.z] = new_marginal

    return Gaussian.KL_divergence(old_marginal, new_marginal)
end


struct TruncationFactor <: Factor
    db_marginal::GaussianBag
    db_msg::GaussianBag
    z::Int64
    lower::Float64
    upper::Float64
    msg_to_z::Int64
end

TruncationFactorWin(db_marginal::GaussianBag, db_msg::GaussianBag, z::Int64, ϵ::Float64) =
    TruncationFactor(db_marginal, db_msg, z, ϵ, Inf, add!(db_msg))

TruncationFactorLoss(db_marginal::GaussianBag, db_msg::GaussianBag, z::Int64, ϵ::Float64) =
    TruncationFactor(db_marginal, db_msg, z, -Inf, - ϵ, add!(db_msg))

TruncationFactorDraw(db_marginal::GaussianBag, db_msg::GaussianBag, z::Int64, ϵ::Float64) =
    TruncationFactor(db_marginal, db_msg, z, min(-ϵ, -1e-5), max(ϵ, 1e-5), add!(db_msg)) #smaller ϵ led to numerical instability

TruncationFactorWin(db::GaussianBag, z::Int64, ϵ::Float64) = TruncationFactor(db, db, z, ϵ, Inf, add!(db))
TruncationFactorLoss(db::GaussianBag, z::Int64, ϵ::Float64) = TruncationFactor(db, db, z, -Inf, -ϵ, add!(db))
TruncationFactorDraw(db::GaussianBag, z::Int64, ϵ::Float64) = TruncationFactor(db, db, z, min(-ϵ, -1e-5), max(ϵ, 1e-5), add!(db)) #smaller ϵ led to numerical instability


function _safe_cavity(old_marginal::Gaussian1D, msg_to_factor::Gaussian1D)::Gaussian1D
    ρ_old = old_marginal.ρ
    τ_old = old_marginal.τ
    ρ_msg = msg_to_factor.ρ
    τ_msg = msg_to_factor.τ

    # Desired cavity: q\m in natural parameters.
    ρ_cav = ρ_old - ρ_msg
    τ_cav = τ_old - τ_msg

    min_ρ = 1e-12 * max(1.0, abs(ρ_old), abs(ρ_msg), abs(τ_old), abs(τ_msg))
    if ρ_cav > min_ρ
        return Gaussian1D(τ_cav, ρ_cav)
    end

    # Fractional EP fallback: use q / m^η with maximal η that keeps cavity proper.
    if ρ_msg <= 0.0
        return Gaussian1D(τ_old, max(ρ_old, min_ρ))
    end

    η = (ρ_old - min_ρ) / ρ_msg
    η = clamp(η, 0.0, 1.0)

    ρ_frac = ρ_old - η * ρ_msg
    τ_frac = τ_old - η * τ_msg
    return Gaussian1D(τ_frac, max(ρ_frac, min_ρ))
end


function _safe_message_ratio(new_marginal::Gaussian1D, cavity::Gaussian1D)::Gaussian1D
    ρ_new = new_marginal.ρ
    τ_new = new_marginal.τ
    ρ_cav = cavity.ρ
    τ_cav = cavity.τ

    ρ_msg = ρ_new - ρ_cav
    τ_msg = τ_new - τ_cav

    min_ρ = 1e-12 * max(1.0, abs(ρ_new), abs(ρ_cav), abs(τ_new), abs(τ_cav))
    if ρ_msg > min_ρ
        return Gaussian1D(τ_msg, ρ_msg)
    end

    # Project to the closest proper message in natural-parameter space.
    return Gaussian1D(τ_msg, min_ρ)
end


function update_msg_to_z!(f::TruncationFactor)
    old_marginal = f.db_marginal[f.z]
    msg_z_to_f_dist = _safe_cavity(old_marginal, f.db_msg[f.msg_to_z])

    μ_z = Gaussian.mean(msg_z_to_f_dist)
    σ_z = sqrt(Gaussian.variance(msg_z_to_f_dist))

    v = v_lu(μ_z, σ_z, f.lower, f.upper)
    w = w_lu(μ_z, σ_z, f.lower, f.upper)

    μ = μ_z + σ_z * v
    # Numerical guard: for extreme tails, roundoff can make (1 - w) slightly
    # negative even though the true variance must be non-negative.
    σ2 = max(σ_z^2 * (1 - w), 1e-12)

    new_marginal = Gaussian1DFromMeanVariance(μ, σ2)
    msg_f_to_z_dist = _safe_message_ratio(new_marginal, msg_z_to_f_dist)

    f.db_msg[f.msg_to_z] = msg_f_to_z_dist
    f.db_marginal[f.z] = new_marginal

    return Gaussian.KL_divergence(old_marginal, new_marginal)
end

# ── Gaussian observation factor ──────────────────────────────────────────────
# Encodes the likelihood  p(m_obs | d) = N(m_obs; d, τ²)
# where d is the (normalised) performance-difference node and m_obs is an
# observed score margin.  The EP message to d is exact: N(d; m_obs, τ²).
struct GaussianObservationFactor <: Factor
    db_marginal::GaussianBag
    db_msg::GaussianBag
    d::Int64         # performance-difference node
    m_obs::Float64   # observed (normalised) margin
    τ²::Float64      # margin observation noise variance
    msg_to_d::Int64
end

GaussianObservationFactor(db::GaussianBag, d::Int64, m_obs::Float64, τ²::Float64) =
    GaussianObservationFactor(db, db, d, m_obs, τ², add!(db))

function update_msg_to_d!(f::GaussianObservationFactor)
    old_marginal = f.db_marginal[f.d]
    # Exact EP: likelihood N(m_obs; d, τ²) gives message N(d; m_obs, τ²)
    new_msg = Gaussian1DFromMeanVariance(f.m_obs, f.τ²)
    cavity  = _safe_cavity(old_marginal, f.db_msg[f.msg_to_d])
    new_marginal = cavity * new_msg
    f.db_msg[f.msg_to_d]  = new_msg
    f.db_marginal[f.d]    = new_marginal
    return Gaussian.KL_divergence(old_marginal, new_marginal)
end

end
