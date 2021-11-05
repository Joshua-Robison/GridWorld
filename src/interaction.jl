# Grid World: Environment Interactions
@enum Action UP=1 DOWN LEFT RIGHT

# Generate Action Space
const actionspace = [UP, DOWN, LEFT, RIGHT]
const actions = ["↑", "↓", "←", "→"] # alias

# Action Space: Movement
const MOVEMENTS = Dict(
    UP    => State(0, 1),
    DOWN  => State(0, -1),
    LEFT  => State(-1, 0),
    RIGHT => State(1, 0)
)

# State Transition Function
@inline Base.:+(s1::State, s2::State)::State = State(s1.x + s2.x, s1.y + s2.y);

# Reward Function
function reward(g::Grid, s::State)::Real
    return inbounds(g, s) ? g[s.x, s.y] : 0;
end

# Transition Function
function transition(g::Grid, s::State, a::Action)
    if reward(g, s) != 0 # reached a terminal state
        return Deterministic(g.null_state);
    end

    N = length(actionspace)
    next_states = Vector{State}(undef, N+1)
    probabilities = zeros(N+1)
    p_transition = g.p_transition

    for (i, a′) in enumerate(actionspace)
        prob = (a′ == a) ? p_transition : (1 - p_transition) / (N - 1)
        destination = s + MOVEMENTS[a′]
        next_states[i+1] = destination
        if inbounds(g, destination)
            probabilities[i+1] += prob
        end
    end

    # handle out of bounds transitions
    next_states[1] = s
    probabilities[1] = 1 - sum(probabilities)

    return SparseCat(next_states, probabilities);
end

# Terminal State: Goal, Error, Out of Bounds
@inline terminal(g::Grid, s::State)::Bool = s == g.null_state;
