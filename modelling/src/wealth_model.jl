using Agents
using Plots; gr()
using AgentsPlots

mutable struct WealthAgent <: AbstractAgent
    id::Int64
    pos::NTuple{2, Int64}
    wealth::Int64
end

WealthAgent(id, pos; wealth) = WealthAgent(id, pos, wealth)

function wealth_model_2D(; dims = (20, 20), wealth = 1, M = 1000)
    space = GridSpace(dims, periodic = false)
    model = ABM(WealthAgent, space; scheduler = random_activation)
    for I in 1:M
        add_agent!(model, wealth)
    end
    return model
end

model2D = wealth_model_2D()

function agent_step!(agent, model)
    agent.wealth == 0 && return
    agent_node = Agents.coord2vertex(agent.pos, model)
    neighbour_nodes = node_neighbors(agent_node, model)
    push!(neighbour_nodes, agent_node)
    rnode = rand(neighbour_nodes)
    a_ids = get_node_contents(rnode, model)
    if length(a_ids) > 0
        r_nagent = model[rand(a_ids)]
        agent.wealth -= 1
        r_nagent.wealth += 1
    end
end

init_wealth = 4
model = wealth_model_2D(; wealth = init_wealth)
adata = [:wealth, :pos]
data, _ = run!(model, agent_step!, 10; adata = adata, when = [1, 5, 9])
data[(end - 20):end, :]


function wealth_distr(data, model, n)
    W = zeros(Int, size(model.space))
    for row in eachrow(filter(r -> r.step == n, data))
        W[row.pos...] += row.wealth
    end
    return W
end

W1 = wealth_distr(data, model2D, 1)

Plots.heatmap(W1)
