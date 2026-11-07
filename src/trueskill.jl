module TrueSkill

include("distributions/collection.jl")
include("distributions/gaussian.jl")
include("distributions/gamma.jl")
include("inference/math.jl")
include("inference/factors.jl")

using CSV
using DataFrames
using Dates
using Distributions
using Plots
using ProgressMeter
using Base.Threads
using Printf
using Random
using Statistics
using LinearAlgebra
using LaTeXStrings
using SpecialFunctions: digamma
using .Gaussian: mean, variance, Gaussian1D, Gaussian1DUniform, Gaussian1DFromMeanVariance, win_prob
using .Gamma
using .MyMath
using .DistributionCollections
using .Factors

const GaussianBag = DistributionBag{Gaussian1D}
const GammaBag = DistributionBag{Gamma1D}

include("synthetic/corruption.jl")
include("models/ratings.jl")

end # module
