function _iterate_until_convergence(factors::NamedTuple, tol::Float64, max_iters::Int)::Float64
    kl_out = 0.0
    for i in 1:max_iters
        kl = 0.0
        @max kl update_msg_to_p!(factors[:performance1])
        @max kl update_msg_to_p!(factors[:performance2])
        @max kl update_msg_to_z!(factors[:diff])
        @max kl update_msg_to_z!(factors[:win])
        @max kl update_msg_to_x!(factors[:diff])
        @max kl update_msg_to_y!(factors[:diff])
        @max kl update_msg_to_b!(factors[:performance1])
        @max kl update_msg_to_s!(factors[:performance1])
        @max kl update_msg_to_b!(factors[:performance2])
        @max kl update_msg_to_s!(factors[:performance2])
        @max kl_out kl
        kl < tol && break
    end
    return kl_out
end

"""Locally equilibrate the three Gaussian--Gamma messages of noise factors.

This is a block-coordinate alternative to the default interleaved EP/VI
schedule.  The surrounding EP messages (outcome and difference factors) are
held fixed while each Gaussian--Gamma factor repeatedly updates its message to
the performance, skill, and precision variables.
"""
function _iterate_local_vi!(performance_factors, tol::Float64, local_max_iters::Int)::Float64
    local_max_iters >= 1 || error("local_max_iters must be at least one")
    kl_out = 0.0
    for _ in 1:local_max_iters
        kl = 0.0
        for factor in performance_factors
            @max kl update_msg_to_p!(factor)
            @max kl update_msg_to_b!(factor)
            @max kl update_msg_to_s!(factor)
        end
        @max kl_out kl
        kl < tol && break
    end
    return kl_out
end

function consistent_trueskill(
    priors_s::GaussianBag,
    priors_b::GammaBag,
    matches::Vector{Tuple{Int64, Int64}},
    ϵ::Float64;
    draw_mask::BitVector=falses(length(matches)),
    s2::Float64=1.0,
    inner_tol::Float64=1e-4,
    inner_max_iters::Int=100,
    schedule::Symbol=:interleaved,
    local_max_iters::Int=1,
    local_tol::Float64=inner_tol,
    outer_tol::Float64=1e-4,
    outer_max_iters::Int=100,
    diagnostics::Bool=false,
    verbose::Bool=false,
)

    db_gauss = DistributionBag(Gaussian1DUniform())
    db_gamma = DistributionBag(Gamma1DUniform())

    n_players, n_matches = length(priors_s), length(matches)
    @assert length(draw_mask) == n_matches "draw_mask length must match number of matches"

    # @assert sorted(unique(vcat([m[1] for m in matches]; [m[2] for m in matches]))) == collect(1:n_players) "Player indices in matches must be 1-based and contiguous up to n_players"
    @assert length(priors_b) == n_players "Length of priors_b must match number of players"

    s = add!(db_gauss, n_players)
    b = add!(db_gamma, n_players)

    # Priors for skills s and per-player performance precisions b
    s_prior_factors = [GaussianFactor(db_gauss, s[i], priors_s[i]) for i in 1:n_players]
    b_prior_factors = [GammaFactor(db_gamma, b[i], priors_b[i]) for i in 1:n_players]
    for i in 1:n_players
        update_msg_to_s!(s_prior_factors[i])
        update_msg_to_b!(b_prior_factors[i])
    end

    # Latent performance variables p1 & p2 and difference variables d for each match
    p1 = add!(db_gauss, n_matches)
    p2 = add!(db_gauss, n_matches)
    d = add!(db_gauss, n_matches)

    # Factors for each match
    p1_factors = [GaussianMeanFactorWithNoise(db_gauss, db_gamma, s[matches[i][1]], p1[i], b[matches[i][1]]; s2=s2) for i in 1:n_matches]
    p2_factors = [GaussianMeanFactorWithNoise(db_gauss, db_gamma, s[matches[i][2]], p2[i], b[matches[i][2]]; s2=s2) for i in 1:n_matches]
    d_factors = [DifferenceFactor(db_gauss, p1[i], p2[i], d[i]) for i in 1:n_matches]
    win_factors = [draw_mask[i] ? TruncationFactorDraw(db_gauss, d[i], ϵ) :
                                  TruncationFactorWin(db_gauss, d[i], ϵ)
                   for i in 1:n_matches]

    # Iterate over full graph
    converged = false
    outer_iterations = 0
    kl = Inf
    for iteration in 1:outer_max_iters
        outer_iterations = iteration
        kl = 0.0

        # Iterate over each match
        for i in 1:n_matches
            if schedule === :interleaved
                # Original schedule: interleave EP and VI messages once per
                # local sweep.
                @max kl _iterate_until_convergence((
                    performance1 = p1_factors[i],
                    performance2 = p2_factors[i],
                    diff = d_factors[i],
                    win = win_factors[i]
                ), inner_tol, inner_max_iters)
            elseif schedule === :local_vi
                # Send EP messages through the local outcome subgraph, then
                # solve the two Gaussian--Gamma factors as local VI blocks.
                for _ in 1:inner_max_iters
                    iter_kl = 0.0
                    @max iter_kl update_msg_to_p!(p1_factors[i])
                    @max iter_kl update_msg_to_p!(p2_factors[i])
                    @max iter_kl update_msg_to_z!(d_factors[i])
                    @max iter_kl update_msg_to_z!(win_factors[i])
                    @max iter_kl update_msg_to_x!(d_factors[i])
                    @max iter_kl update_msg_to_y!(d_factors[i])
                    @max iter_kl _iterate_local_vi!((p1_factors[i], p2_factors[i]), local_tol, local_max_iters)
                    @max kl iter_kl
                    iter_kl < inner_tol && break
                end
            else
                error("Unknown schedule $(schedule); use :interleaved or :local_vi")
            end
        end

        if kl < outer_tol
            converged = true
            break
        end
    end
    if verbose
        if converged
            println("\nInference converged.")
        else
            println("\nInference stopped at max outer iterations without meeting tolerance.")
        end
    end

    result = (DistributionCollections.subset(db_gauss, s), DistributionCollections.subset(db_gamma, b))
    return diagnostics ? (result..., (converged=converged, outer_iterations=outer_iterations,
        final_residual=kl)) : result
end


function robust_trueskill(
    priors_s::GaussianBag,
    priors_b::GammaBag,
    matches::Vector{Tuple{Int64, Int64}},
    ϵ::Float64;
    draw_mask::BitVector=falses(length(matches)),
    s2::Float64=1.0,
    inner_tol::Float64=1e-4,
    inner_max_iters::Int=20,
    schedule::Symbol=:interleaved,
    local_max_iters::Int=1,
    local_tol::Float64=inner_tol,
    outer_tol::Float64=1e-4,
    outer_max_iters::Int=200,
    diagnostics::Bool=false,
    verbose::Bool=false,
)

    db_gauss = DistributionBag(Gaussian1DUniform())
    db_gamma = DistributionBag(Gamma1DUniform())

    n_players, n_matches = length(priors_s), length(matches)
    @assert length(draw_mask) == n_matches "draw_mask length must match number of matches"

    @assert length(priors_b) == n_matches "Length of priors_b must match number of matches"

    s = add!(db_gauss, n_players)
    b = add!(db_gamma, n_matches)

    # Priors for skills s and per-match performance precisions b
    s_prior_factors = [GaussianFactor(db_gauss, s[i], priors_s[i]) for i in 1:n_players]
    b_prior_factors = [GammaFactor(db_gamma, b[i], priors_b[i]) for i in 1:n_matches]
    for i in 1:n_players
        update_msg_to_s!(s_prior_factors[i])
    end
    for i in 1:n_matches
        update_msg_to_b!(b_prior_factors[i])
    end

    # Skill-difference variable m and noisy observation variable d for each match.
    m = add!(db_gauss, n_matches)
    d = add!(db_gauss, n_matches)

    diff_factors  = [DifferenceFactor(db_gauss, s[matches[i][1]], s[matches[i][2]], m[i]) for i in 1:n_matches]
    noise_factors = [GaussianMeanFactorWithNoise(db_gauss, db_gamma, m[i], d[i], b[i]; s2=s2) for i in 1:n_matches]
    win_factors   = [draw_mask[i] ? TruncationFactorDraw(db_gauss, d[i], ϵ) :
                                    TruncationFactorWin(db_gauss, d[i], ϵ)
                     for i in 1:n_matches]

    converged = false
    outer_iterations = 0
    kl = Inf
    for iteration in 1:outer_max_iters
        outer_iterations = iteration
        kl = 0.0
        for i in 1:n_matches
            for _ in 1:inner_max_iters
                iter_kl = 0.0
                @max iter_kl update_msg_to_z!(diff_factors[i])
                @max iter_kl update_msg_to_p!(noise_factors[i])
                @max iter_kl update_msg_to_z!(win_factors[i])
                if schedule === :interleaved
                    @max iter_kl update_msg_to_s!(noise_factors[i])
                    @max iter_kl update_msg_to_b!(noise_factors[i])
                    @max iter_kl update_msg_to_x!(diff_factors[i])
                    @max iter_kl update_msg_to_y!(diff_factors[i])
                elseif schedule === :local_vi
                    @max iter_kl update_msg_to_x!(diff_factors[i])
                    @max iter_kl update_msg_to_y!(diff_factors[i])
                    @max iter_kl _iterate_local_vi!((noise_factors[i],), local_tol, local_max_iters)
                else
                    error("Unknown schedule $(schedule); use :interleaved or :local_vi")
                end
                @max kl iter_kl
                iter_kl < inner_tol && break
            end
        end
        if kl < outer_tol
            converged = true
            break
        end
    end
    verbose && println(converged ? "\nInference converged." : "\nInference stopped at max outer iterations without meeting tolerance.")
    result = (DistributionCollections.subset(db_gauss, s), DistributionCollections.subset(db_gamma, b))
    return diagnostics ? (result..., (converged=converged, outer_iterations=outer_iterations,
        final_residual=kl)) : result
end

function vanilla_trueskill(
    priors_s::GaussianBag,
    β²::Float64,
    matches::Vector{Tuple{Int64, Int64}},
    ϵ::Float64;
    draw_mask::BitVector=falses(length(matches)),
    outer_tol::Float64=1e-4,
    outer_max_iters::Int=200,
    verbose::Bool=false,
)::GaussianBag

    db_gauss = DistributionBag(Gaussian1DUniform())

    n_players, n_matches = length(priors_s), length(matches)
    @assert length(draw_mask) == n_matches "draw_mask length must match number of matches"

    s = add!(db_gauss, n_players)

    # Priors for skills s
    s_prior_factors = [GaussianFactor(db_gauss, s[i], priors_s[i]) for i in 1:n_players]
    for i in 1:n_players
        update_msg_to_s!(s_prior_factors[i])
    end

    # Latent performance variables p1 & p2 and difference variables d for each match
    p1 = add!(db_gauss, n_matches)
    p2 = add!(db_gauss, n_matches)
    d = add!(db_gauss, n_matches)

    # Factors for each match
    p1_factors = [GaussianMeanFactor(db_gauss, db_gauss, s[matches[i][1]], p1[i], β²) for i in 1:n_matches]
    p2_factors = [GaussianMeanFactor(db_gauss, db_gauss, s[matches[i][2]], p2[i], β²) for i in 1:n_matches]
    d_factors = [DifferenceFactor(db_gauss, p1[i], p2[i], d[i]) for i in 1:n_matches]
    win_factors = [draw_mask[i] ? TruncationFactorDraw(db_gauss, d[i], ϵ) :
                                  TruncationFactorWin(db_gauss, d[i], ϵ)
                   for i in 1:n_matches]

    # Iterate over full graph
    converged = false
    for _ in 1:outer_max_iters
        kl = 0.0

        # Iterate over each match
        for i in 1:n_matches
            @max kl update_msg_to_p!(p1_factors[i])
            @max kl update_msg_to_p!(p2_factors[i])
            @max kl update_msg_to_z!(d_factors[i])
            @max kl update_msg_to_z!(win_factors[i])
            @max kl update_msg_to_x!(d_factors[i])
            @max kl update_msg_to_y!(d_factors[i])
            @max kl update_msg_to_s!(p1_factors[i])
            @max kl update_msg_to_s!(p2_factors[i])
        end

        if kl < outer_tol
            converged = true
            break
        end
    end
    if verbose
        if converged
            println("\nInference converged.")
        else
            println("\nInference stopped at max outer iterations without meeting tolerance.")
        end
    end

    return DistributionCollections.subset(db_gauss, s)
end
