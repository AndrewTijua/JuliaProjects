using Agents
using Plots
plotlyjs()
using Statistics
using Random
using Distributions

mutable struct OpinionAgent <: AbstractAgent
    id::Int
    old_o::Float64
    new_o::Float64
    pre_o::Float64
    p_eps::Float64
end

OpinionAgent(id; old_o, new_o, pre_o, p_eps) = OpinionAgent(id, old_o, new_o, pre_o, p_eps)

function opinion_model(; numagents = 100, ϵ = 0.2)
    model = ABM(OpinionAgent, scheduler = fastest, properties = Dict(:ϵ => ϵ))
    peps_dist = truncated(Normal(ϵ, sqrt(ϵ)), 0., 1.)
    opin_dist = truncated(Normal(0.5, 0.35), 0., 1.)
    for i in 1:numagents
        opinion = rand(opin_dist)
        p_eps = rand(peps_dist)
        add_agent!(model, opinion, opinion, -1., p_eps)
    end
    return model
end

model = opinion_model()

function boundfilter(agent, model)
    filter(j -> abs(agent.old_o - j) < model.ϵ, [a.old_o for a in allagents(model)],)
end

function agent_step!(agent, model)
    agent.pre_o = agent.old_o
    agent.new_o = (0.85*agent.pre_o + 0.15*mean(boundfilter(agent, model)))
end

function model_step!(model)
    for a in allagents(model)
        a.old_o = a.new_o
    end
end

function terminate(model, s)
    if any(
        !isapprox(a.pre_o, a.new_o; rtol = 1e-4) for a in allagents(model)
    )
        return false
    else
        return true
    end
end

function model_run(; kwargs...)
    model = opinion_model(; kwargs...)
    agent_data, _ = run!(model, agent_step!, model_step!, terminate; adata = [:new_o])
    return agent_data
end

#k = model_run(ϵ = 0.14)

plotsim(data, ϵ) = plot(
    data.step,
    data.new_o,
    leg = false,
    group = data.id,
    title = "E(ϵ) = $(ϵ)", ylims = (0,1))

plt1, plt2, plt3, plt4 =
    map(e -> (model_run(ϵ = e), e) |> t -> plotsim(t[1], t[2]), [0.10, 0.15, 0.20, 0.30])

plot(plt1, plt2, plt3, plt4, layout = (4, 1))
