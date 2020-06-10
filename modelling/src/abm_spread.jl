using Agents
using LightGraphs
using LinearAlgebra
using DataFrames
using Plots;
plotlyjs();
using DrWatson: @dict
using Random
using Distributions: Poisson, DiscreteNonParametric, Exponential, Gamma, truncated, Normal, Binomial

mutable struct indiv <: AbstractAgent
    id::Int64
    pos::Int64
    age::Int8
    days_exposed_cd::Int8
    days_infected::Int8
    status::Symbol #1:S, 2:E, 3:I, 4:R
    will_iso::Bool
    iso::Bool #true if isolating
end

indiv(id, pos; age, days_exposed_cd, days_infected, status, will_iso, iso) = indiv(id, pos, age, days_exposed_cd, days_infected, status, will_iso, iso)

function model_initialisation(;
    Ns, #city populations
    m_rates, #migration rates
    β_und, #transmission for undetected
    β_det, #transmission for detected
    Λ = 1 / 14, #exponential parameter for incubation
    infection_period = 14,
    reinfection_prob = 0.02,
    detection_time = 7,
    death_med = 75,
    death_gr = 0.2,
    death_mv = 0.98,
    Is = [zeros(Int64, length(Ns) - 1)..., 1],
    seed = 0,
    a_age = 40 * (40 / 30^2),
    b_age = 40 / 30^2,
    iso_prob,
)

    Random.seed!(seed)

    #test inputs make sense
    @assert detection_time <= infection_period "infection period should be longer than detection time"
    @assert Λ > 0 "Exponential parameter must be greater than 0"
    @assert reinfection_prob >= 0 "Reinfection probability must be at least 0"
    @assert length(Ns) == length(Is) == length(β_und) == length(β_det) == size(m_rates, 1) "Length of Ns, Is, β, migration must be the same"
    @assert size(m_rates, 1) == size(m_rates, 2) "migration matrix must be square"

    C = length(Ns) #cities

    #normalise migration rates
    m_rates_sum = sum(m_rates, dims = 2)
    for c = 1:C
        m_rates[c, :] ./= m_rates_sum[c]
    end

    properties = @dict(Ns, Is, β_und, β_det, Λ, infection_period, reinfection_prob, detection_time, death_med, death_gr, death_mv, C)

    age_dist = truncated(Gamma(a_age, 1 / b_age), 0, 100)

    space = GraphSpace(complete_digraph(C)) #directed graph with edges between all nodes (can go to and from any area)

    model = ABM(indiv, space; properties = properties)


    for city = 1:C, n = 1:Ns[city]
        ind = add_agent!(
            city,
            model;
            age = floor(rand(age_dist)),
            days_exposed_cd = 0,
            days_infected = 0,
            status = :S,
            will_iso = convert(Bool, rand(Binomial(1, iso_prob))),
            iso = false,
        )
    end
    for city = 1:C
        inds = get_node_contents(city, model)
        for n = 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I
            agent.days_infected = 1
        end
    end
    return model
end

function generate_cities_params(;C::Int64, max_travel_rate::Float64, pop_params = [3e2, (1e2)])
    pop_dist = truncated(Normal(pop_params[1], pop_params[2]), 1_00, Inf)
    Ns = floor.(Int64, rand(pop_dist, C))
    β_und = rand(0.3:0.02:0.6, C)
    β_det = β_und ./ 10

    m_rates = zeros(C, C)
    for c1 = 1:C
        for c2 = 1:C
            m_rates[c1, c2] = (Ns[c1] + Ns[c2]) / Ns[c1]
        end
    end
    maxM = maximum(m_rates)
    m_rates = (m_rates .* max_travel_rate) ./ maxM
    m_rates[diagind(m_rates)] .= 1.0

    city_params = @dict(
    C,
    Ns,
    β_und,
    β_det,
    m_rates)

    return city_params
end

c_params = generate_cities_params(C = 8, max_travel_rate = 0.01)

Ns = c_params[:Ns]
C = c_params[:C]
β_und = c_params[:β_und]
β_det = c_params[:β_det]
m_rates = c_params[:m_rates]
Λ = 1 / 14 #exponential parameter for incubation
infection_period = 14
reinfection_prob = 0.02
detection_time = 7
death_med = 75
death_gr = 0.2
death_mv = 0.98
seed = 0
a_age = 40 * (40 / 30^2)
b_age = 40 / 30^2
Is = [zeros(Int64, length(Ns) - 1)..., 100]
iso_prob = 0.0

params = @dict(
Ns,
β_und,
β_det,
m_rates,
Λ,
infection_period,
reinfection_prob,
detection_time,
death_med,
death_gr,
death_mv,
seed,
a_age,
b_age,
Is,
iso_prob,
)

model = model_initialisation(; params...)

using AgentsPlots

plotargs = (node_size = 0.2, method = :circular, linealpha = 0.4)
g = model.space.graph
edgewidthsdict = Dict()
for node in 1:nv(g)
    nbs = neighbors(g, node)
    for nb in nbs
        edgewidthsdict[(node, nb)] = params[:m_rates][node, nb]
    end
end
edgewidthsf(s, d, w) = edgewidthsdict[(s, d)] * 250
plotargs = merge(plotargs, (edgewidth = edgewidthsf,))
infected_fraction(x) = cgrad(:inferno)[count(a.status == :I for a in x) / length(x)]
plotabm(model; ac = infected_fraction, plotargs...)

function agent_step(agent, model)
    isolate!(agent, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end


function migrate!(agent, model)
    agent.iso && return
    nodeid = agent.pos
    d = DiscreteNonParametric(1:(model.C), model.m_rates[nodeid, :])
    m = rand(d)
    if m ≠ nodeid
        move_agent!(agent, m, model)
    end
end
