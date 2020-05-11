include("defines.jl")

using Plots;
plotlyjs();

timestep = 86400 * 1.5

G = 6.67730e-11
G_km = G / (1e3)^3


bodies_list = read_bodies("bodies.csv", timestep)
momentum_removal!(bodies_list)

target_time = 36500 * timestep * 2
n_steps = ceil(Int, target_time / timestep)

n_bodies = size(bodies_list, 1)


pos_array = Array{Vector{Float64},2}(undef, n_steps, n_bodies)
sysenergy = Vector{Float64}(undef, n_steps)

Threads.@threads for body in bodies_list
    force_update!(body, bodies_list, G_km)
end

for step = 1:n_steps
    sysenergy[step] = 0
    force_update_list!(bodies_list, G_km)
    Threads.@threads for body in bodies_list
        #force_update!(body, bodies_list, G_km)
        velocity_update!(body)
    end
    Threads.@threads for body in bodies_list
        position_update!(body)
        pos_array[step, body.number] = body.position
    end
    for body in bodies_list
        sysenergy[step] -= body.ke + body.pe * 0.5
    end
end

pos_plot = plot(
    title = "Position plot",
    marker = 2,
    aspect_ratio = :equal,
    camera = [45, 60],#, legend = :none,
)

ax_lim = 7e9
xlims!(-ax_lim, ax_lim)
ylims!(-ax_lim, ax_lim)
zlims!(-ax_lim, ax_lim)

plot_every = 10

for n = 1:n_bodies
    pa = pos_array[:, n]
    n_plot = floor(Int, n_steps / plot_every)
    pa_x = Vector{Float64}(undef, n_plot)
    pa_y = Vector{Float64}(undef, n_plot)
    pa_z = Vector{Float64}(undef, n_plot)
    Threads.@threads for i = 1:n_plot
        pa_x[i] = pa[i*plot_every][1]
        pa_y[i] = pa[i*plot_every][2]
        pa_z[i] = pa[i*plot_every][3]
    end
    plot3d!(pa_x, pa_y, pa_z, lab = bodies_list[n].name)
end

pos_plot

plot(sysenergy[1:plot_every:n_steps])
#xlims!(0, n_steps)
ylims!(minimum(sysenergy)/1.001, maximum(sysenergy)*1.001)
