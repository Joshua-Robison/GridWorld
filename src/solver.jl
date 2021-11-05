# Grid World: Value Iteration Solver
export Policy
export solve

struct Policy
    policy::Matrix{Real}
    actions::Vector{String}
    function Policy(p::Matrix; a::Vector{String} = actions)
        return new(p, a);
    end
end

function Base.show(io::IO, p::Policy)::Nothing
    # println("\033[2J") # clear repl
    # println("\033[$(displaysize(stdout)[1])A")
    rows, cols = size(p.policy)
    border = "\t" * " "^7 * "-"^(cols * 7)
    println(border)
    for i in rows:-1:1
        print("\t")
        for j in 1:cols
            if j == 1
                @printf "%-5i  | %s   |" i p.actions[p.policy[j,i]]
            else
                @printf "  %s   |" p.actions[p.policy[j,i]]
            end
        end
        println("\n" * border)
    end
    print("\t   ")
    for k in 1:cols
        @printf "    %3.0f" k
    end
end

function solve(g::Grid; iterations::Int = 100, belres::Float64 = 1e-3, discount_factor::Float64 = 0.90)
    ns = length(statespace(g)) # number of states
    na = length(actionspace)   # number of actions

    # initialize the policy matrix
    pol = zeros(Int64, ns)
    util = zeros(ns)

    # time solver
    total_time = 0.0
    iter_time = 0.0

    # main loop
    for i in 1:iterations
        residual = 0.0
        iter_time = @elapsed begin
            # state loop
            for (istate, s) in enumerate(statespace(g))
                if terminal(g, s)
                    pol[istate] = 1
                else
                    old_util = util[istate] # for residual
                    max_util = -Inf
                    # action loop
                    # util(s) = max_a( R(s,a) + discount_factor * sum(T(s'|s,a)util(s') )
                    for (iaction, a) in enumerate(actionspace)
                        dist = transition(g, s, a)
                        u = 0.0
                        for (sp, p) in weighted_iterator(dist)
                            p == 0.0 ? continue : nothing # skip if zero prob
                            r = reward(g, sp)
                            isp = stateindex(g, sp)
                            u += p * (r + discount_factor * util[isp])
                        end
                        new_util = u
                        if new_util > max_util
                            max_util = new_util
                            pol[istate] = iaction
                        end
                    end # action
                    # update the value array
                    util[istate] = max_util
                    diff = abs(max_util - old_util)
                    diff > residual ? (residual = diff) : nothing
                end
            end # state
        end # time
        total_time += iter_time
        @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time)
        residual < belres ? break : nothing
    end # main
    rows, cols = g.dims
    policy = reshape(pol[1:end-1], rows, cols)
    return Policy(policy);
end
