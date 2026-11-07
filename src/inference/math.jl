module MyMath

using Distributions
using SpecialFunctions: erfcx

φ(z) = Distributions.pdf(Distributions.Normal(0,1), z)
Φ(z) = Distributions.cdf(Distributions.Normal(0,1), z)

const SQRT_2 = sqrt(2.0)
const SQRT_2_OVER_PI = sqrt(2.0 / pi)

function _phi_over_cdf(x::Float64)
    if !isfinite(x)
        return x > 0 ? 0.0 : Inf
    end
    if x < -8.0
        # Stable Mills ratio in the extreme left tail.
        return SQRT_2_OVER_PI / erfcx(-x / SQRT_2)
    end
    den = max(Φ(x), 1e-300)
    return φ(x) / den
end

function _phi_over_ccdf(x::Float64)
    # Use symmetry: 1 - Φ(x) = Φ(-x), φ(x) = φ(-x)
    return _phi_over_cdf(-x)
end

# One sided truncation
v(t) = φ(t) / Φ(t)
w(t) = v(t) * (v(t) + t)

function v_lu(μ, σ, lower, upper)
    if !isfinite(μ) || !isfinite(σ) || σ <= 0.0
        return 0.0
    end

    α = (lower - μ) / σ
    β = (upper - μ) / σ

    if isfinite(α) && isfinite(β)
        z_mass = max(Φ(β) - Φ(α), 1e-300)
        return (φ(α) - φ(β)) / z_mass
    elseif isfinite(α) && !isfinite(β)   # win (lower truncation)
        return _phi_over_ccdf(α)
    elseif !isfinite(α) && isfinite(β)   # loss (upper truncation)
        return -_phi_over_cdf(β)
    else
        return 0.0
    end
end

function w_lu(μ, σ, lower, upper)
    if !isfinite(μ) || !isfinite(σ) || σ <= 0.0
        return 0.0
    end

    α = (lower - μ) / σ
    β = (upper - μ) / σ
    v_val = v_lu(μ, σ, lower, upper)
    if !isfinite(v_val)
        return 0.0
    end

    if isfinite(α) && isfinite(β)
        z_mass = max(Φ(β) - Φ(α), 1e-300)
        # For a Gaussian truncated to [α, β], the variance is
        # 1 + (αφ(α) - βφ(β)) / Z - v².  `w` is the complementary
        # variance-reduction factor used by the EP update.
        return v_val^2 + (β * φ(β) - α * φ(α)) / z_mass
    elseif isfinite(α) && !isfinite(β)   # win (lower truncation)
        return v_val * (v_val - α)
    elseif !isfinite(α) && isfinite(β)   # loss (upper truncation)
        return v_val * (v_val - β)
    else
        return 0.0
    end
end

macro max(acc, expr)
    return :($(esc(acc)) = max($(esc(acc)), $(esc(expr))))
end

function kendall_tau(v1::Vector, v2::Vector)::Float64
    # Kendall's τ (tau) between two orderings v1 and v2.
    # Returns 1.0 for a perfect positive agreement, -1.0 for a perfect reversal, and 0.0 for no correlation.
    n = length(v1)
    n == 1 && return 1.0
    C = D = 0
    for i in 1:n-1, j in i+1:n
        sg = sign(v1[i] - v1[j]) * sign(v2[i] - v2[j])
        sg > 0 ? (C += 1) : sg < 0 ? (D += 1) : nothing
    end
    return (C - D) / (n * (n - 1) ÷ 2)
end


export φ, Φ, v, w, v_lu, w_lu, @max, kendall_tau

end
