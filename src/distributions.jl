#=
    SOURCE: POMDPModelTools.jl
    --------------------------
    A deterministic distribution over only one value.
    This is for creating a distribution, but the outcome is deterministic.
    It is equivalent to the Kronecker Delta distribution.
=#
struct Deterministic{T}
    val::T
end

@inline Base.rand(d::Deterministic) = d.val;
@inline pdf(d::Deterministic, x) = convert(Float64, x == d.val);
@inline mode(d::Deterministic) = d.val;
@inline mean(d::Deterministic{N}) where N<:Number = d.val / 1;
@inline mean(d::Deterministic) = d.val;
@inline support(d::Deterministic) = (d.val,);
@inline weighted_iterator(d) = (x=>pdf(d, x) for x in support(d));

#=
    SOURCE: POMDPModelTools.jl
    --------------------------
    A sparse categorical distribution.
=#
struct SparseCat{V, P}
    vals::V
    probs::P
end

function Base.iterate(d::SparseCat)
    val, vstate = iterate(d.vals)
    prob, pstate = iterate(d.probs)
    return ((val=>prob), (vstate, pstate));
end

function Base.iterate(d::SparseCat, dstate::Tuple)
    vstate, pstate = dstate
    vnext = iterate(d.vals, vstate)
    pnext = iterate(d.probs, pstate)
    if vnext === nothing || pnext === nothing
        return nothing;
    end
    val, vstate_next = vnext
    prob, pstate_next = pnext
    return ((val=>prob), (vstate_next, pstate_next));
end

const Indexed = Union{AbstractArray, Tuple, NamedTuple}

@inline support(d::SparseCat) = d.vals;
@inline weighted_iterator(d::SparseCat) = d;

@inline Base.length(d::SparseCat) = min(length(d.vals), length(d.probs));

function Base.iterate(d::SparseCat{V,P}, state::Integer=1) where {V<:Indexed, P<:Indexed}
    if state > length(d)
        return nothing;
    end
    return (d.vals[state]=>d.probs[state], state+1);
end

function rand(d::SparseCat)
    r = sum(d.probs) * rand()
    tot = zero(eltype(d.probs))
    for (v, p) in d
        tot += p
        if r < tot
            return v;
        end
    end
    if sum(d.probs) <= 0.0
        error("""
              Tried to sample from a SparseCat distribution with probabilities that sum to $(sum(d.probs)).
              vals = $(d.vals)
              probs = $(d.probs)
              """)
    end
    error("Error sampling from SparseCat distribution with vals $(d.vals) and probs $(d.probs)")
end

function mean(d::SparseCat)
    vsum = zero(eltype(d.vals))
    for (v, p) in d
        vsum += v * p
    end
    return vsum / sum(d.probs);
end

function mode(d::SparseCat)
    bestp = zero(eltype(d.probs))
    bestv = first(d.vals)
    for (v, p) in d
        if p >= bestp
            bestp = p
            bestv = v
        end
    end
    return bestv;
end
