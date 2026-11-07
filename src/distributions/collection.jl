module DistributionCollections

struct DistributionBag{T} <: AbstractArray{T, 1}
    uniform::T      # the uniform distribution of distribution type T
    bag::Vector{T}  # the actual resizable vector that stores the distributions
    # default constructor
    DistributionBag{T}(uniform::T) where {T} = new(uniform, Vector{T}(undef, 0))
end

DistributionBag(uniform::T) where {T} = DistributionBag{T}(uniform)

function DistributionBag(d::T, n::Int) where {T}
    db = DistributionBag{T}(d)
    add!(db, n)
    return db
end

function Base.show(io::IO, db::DistributionBag{T}) where {T}
    println(io, "Uniform: ", db.uniform)
    if (length(db.bag) == 0)
        println(io, "  empty bag")
    else
        for i in eachindex(db.bag)
            println(io, "  [", i, "]: ", db.bag[i])
        end
    end
end

function add!(db::DistributionBag{T}) where {T}
    push!(db.bag, db.uniform)
    return (length(db.bag))
end

function add!(db::DistributionBag{T}, n::Int) where {T}
    out = Vector{Int}(undef, n)
    for i in 1:n
        out[i] = add!(db)
    end
    return Tuple(out)
end

function reset!(db::DistributionBag{T}) where {T}
    for i in eachindex(db.bag)
        db.bag[i] = db.uniform
    end
    return
end

function subset(db::DistributionBag{T}, idxs::AbstractVector{<:Integer}) where {T}
    out = DistributionBag{T}(db.uniform)
    for i in idxs
        push!(out.bag, db.bag[i])
    end
    return out
end

subset(db::DistributionBag{T}, idxs::Tuple{Vararg{<:Integer}}) where {T} = subset(db, collect(idxs))

Base.getindex(db::DistributionBag{T}, i::Int64) where {T} = db.bag[i]

function Base.setindex!(db::DistributionBag{T}, d::T, i::Int64) where {T}
    db.bag[i] = d
end

Base.firstindex(db::DistributionBag{T}) where {T} = return (1)

Base.lastindex(db::DistributionBag{T}) where {T} = return (length(db.bag))

Base.size(db::DistributionBag{T}) where {T} = (length(db.bag),)

Base.IndexStyle(::Type{<:DistributionBag{T}}) where {T} = IndexLinear()

Base.eltype(::Type{<:DistributionBag{T}}) where {T} = T

Base.iterate(db::DistributionBag{T}, i=1) where {T} = (i > length(db.bag)) ? nothing : (db.bag[i], i+1)

export DistributionBag, add!, reset!, subset

end
