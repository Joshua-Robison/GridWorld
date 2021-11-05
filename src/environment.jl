# Grid World: Environment Definitions
using Printf

export Grid, State

# Grid Position
struct State
    x::Int # x-position
    y::Int # y-position
    function State(x::Int, y::Int)
        return new(x, y);
    end
end

# State Equality
@inline Base.:(==)(s1::State, s2::State)::Bool = (s1.x == s2.x) && (s1.y == s2.y);

# Grid Structure
struct Grid
    dims::Tuple{Int,Int}     # size of the grid
    null_state::State        # terminal state off the grid
    p_transition::Real       # probability of transitioning correctly
    rewards::Matrix{Float64} # reward matrix
    function Grid(dims::Tuple{Int,Int} = (10, 10);
                  null_state::State = State(-1, -1),
                  p_transition::Real = 0.70)
        # initialize empty rewards matrix
        rewards = zeros(dims)
        return new(dims, null_state, p_transition, rewards);
    end
end

# Grid Interfaces
function Base.getindex(g::Grid, i::Int, j::Int)::Float64
    if (1 ≤ i ≤ g.dims[1]) && (1 ≤ j ≤ g.dims[2])
        return g.rewards[i,j];
    end
    throw(BoundsError(g));
end

function Base.setindex!(g::Grid, v::Real, i::Int, j::Int)
    value = Float64(v)
    if (1 ≤ i ≤ g.dims[1]) && (1 ≤ j ≤ g.dims[2])
        g.rewards[i,j] = value;
    else
        throw(BoundsError(g));
    end
end

# Display Grid World
function Base.show(io::IO, g::Grid)::Nothing
    println("\033[2J") # clear repl
    println("\033[$(displaysize(stdout)[1])A")
    rows, cols = g.dims
    border = "\t" * " "^7 * "-"^(cols * 11)
    println(border)
    for i in rows:-1:1
        print("\t")
        for j in 1:cols
            if j == 1
                @printf "%-5i  | %5.1f   |" i g.rewards[j,i]
            else
                @printf "  %5.1f   |" g.rewards[j,i]
            end
        end
        println("\n" * border)
    end
    print("\t" * " "^4)
    for k in 1:cols
        @printf "    %5.0f  " k
    end
end

# Generate State Space: including the defined null state
@inline statespace(g::Grid)::Vector{State} = [[State(x,y) for x=1:g.dims[1], y=1:g.dims[2]]..., g.null_state];
@inline stateindex(g::Grid, s::State)::Int = findall(x->x == s, statespace(g))[1];

# Determine if a State Position is Valid (in the Grid)
@inline inbounds(g::Grid, s::State)::Bool = (1 ≤ s.x ≤ g.dims[1]) && (1 ≤ s.y ≤ g.dims[2]);
