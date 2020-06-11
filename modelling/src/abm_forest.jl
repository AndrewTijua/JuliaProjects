using Agents, Random
using Plots;
gr();
using AgentsPlots

mutable struct Tree <: AbstractAgent
    id::Int64
    pos::Tuple{Int64,Int64}
    status::Bool #true if not burning, false if burning
    health::Float64 #decrease a random amount when burning
end

tree(id, pos; status, health) = tree(id, pos, status, health)

function model_init(; f = 0.02, d = 0.8, p = 0.01, griddims = (100, 100), seed = 1, health = 2.0)
    Random.seed!(seed)
    space = GridSpace(griddims, moore = true)
    properties = Dict(:f => f, :d => d, :p => p, :h => health)
    forest = AgentBasedModel(Tree, space; properties = properties)

    for node in nodes(forest)
        if rand() <= forest.d
            add_agent!(node, forest, true, forest.h)
        end
    end
    return forest
end

function forest_step!(forest)
    for node in nodes(forest, by = :id)
        contents = get_node_contents(node, forest)
        if length(contents) == 0
            rand() <= forest.p && add_agent!(node, forest, true, forest.h)
        else
            tree = forest[contents[1]]
            if tree.status == true
                tree.health = tree.health + 0.1*rand()
            elseif tree.status == false
                tree.health = tree.health - rand()
            end
            if tree.health <= 0
                kill_agent!(tree, forest)
            end
            if rand() <= forest.f
                tree.status = false
            else
                for cell in node_neighbors(node, forest)
                    neighbors = get_node_contents(cell, forest)
                    length(neighbors) == 0 && continue
                    if any(n -> !forest.agents[n].status, neighbors)
                        tree.status = false
                        break
                    end
                end
            end
        end
    end
end

forest = model_init(f = 0.02, d = 0.8, p = 0.05, griddims = (20, 20), seed = 1, health = 2.0)
treecolor(a) = a.status == 1 ? :green : :red
step!(forest, dummystep, forest_step!, 1)
# forest = model_init(griddims = (20, 20), seed = 2)
# percentage(x) = count(x) / nv(forest)
# adata = [(:status, percentage)]
# data, _ = run!(forest, dummystep, forest_step!, 10; adata = adata)
# data

gd = (50, 50)
step = 10
frames = 150

forest = model_init(f = 0.00005, p = 0.001, seed = 30, d = 0.2, griddims = gd)
step!(forest, dummystep, forest_step!, 100)
anim = @animate for i = 0:step:(step*frames)
    i > 0 && step!(forest, dummystep, forest_step!, step)
    p1 = plotabm(forest; ac = treecolor, ms = 6, msw = 0, xlims = (0, gd[1] + 1), ylims = (0, gd[2] + 1), size = (780, 780))
    title!(p1, "step $(i)")
end

gif(anim, "fire_forest.gif", fps = 15)
