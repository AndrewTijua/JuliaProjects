using DelimitedFiles
using LinearAlgebra
using Optim

mutable struct body
    position::Vector{Float64}
    velocity::Vector{Float64}
    force_old::Vector{Float64}
    force::Vector{Float64}
    mass::Float64
    timestep::Float64
    name::String
    number::Int64
    ke::Float64
    pe::Float64
end

body() =
    body([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0, 0, "undef", -1, 0, 0)

function fast_norm_sq(x::Vector)
    l::Float64 = 0
    for i in x
        l += i^2
    end
    return l
end

function fast_norm(x::Vector)
    l::Float64 = 0
    for i in x
        l += i^2
    end
    return sqrt(l)
end

function velocity_update!(x::body)
    x.velocity += 0.5 * (x.force + x.force_old) * x.timestep / x.mass
    x.ke = 0.5 * x.mass * fast_norm_sq(x.velocity)
end

function position_update!(x::body)
    x.position +=
        x.velocity * x.timestep + (x.force / (2 * x.mass)) * (x.timestep^2)
end

function force_update!(x::body, bodylist, G)
    of = x.force
    x.force_old = of
    x.force = [0, 0, 0]
    for other in bodylist
        if (x != other)
            sepvec = other.position - x.position
            sep = fast_norm(sepvec)
            x.force += G * sepvec * x.mass * other.mass / (sep^3)
        else
        end
    end
end

function force_update_list!(bodylist, G)
    n_bodies = size(bodylist, 1)
    for body in bodylist
        body.force_old = body.force
        body.force = [0, 0, 0]
        body.pe = 0
    end
    for i = 2:n_bodies
        for j = 1:(i-1)
            sepvec = bodylist[j].position - bodylist[i].position
            sep = fast_norm(sepvec)
            force =
                G * sepvec * (bodylist[i].mass) * (bodylist[j].mass) / (sep^3)
            potential = G * (bodylist[i].mass) * (bodylist[j].mass) / sep
            bodylist[i].force += force
            bodylist[i].pe -= potential
            bodylist[j].force -= force
            bodylist[j].pe -= potential
        end
    end
end

function read_bodies(filename, timestep)
    filearr = readdlm(filename, ',')
    n_l = size(filearr, 1) - 1
    blist = Array{body,1}(undef, n_l)
    for lineno = 1:n_l
        tempbody = body()
        line = filearr[lineno+1, :]
        tempbody.position[1] = line[1]
        tempbody.position[2] = line[2]
        tempbody.position[3] = line[3]
        tempbody.velocity[1] = line[4]
        tempbody.velocity[2] = line[5]
        tempbody.velocity[3] = line[6]
        tempbody.force[1] = line[7]
        tempbody.force[2] = line[8]
        tempbody.force[3] = line[9]
        tempbody.mass = line[10]
        tempbody.timestep = timestep
        tempbody.name = line[11]
        tempbody.number = lineno
        blist[lineno] = tempbody
    end
    return (blist)
end

function momentum_removal!(bodylist)
    momentum = [0, 0, 0]
    mass = 0
    for body in bodylist
        mass += body.mass
        momentum += body.mass * body.velocity
    end
    adj_vel = momentum / mass
    for body in bodylist
        body.velocity -= adj_vel
    end
end

function quadratic_form(coefs::Vector{Float64}, x, y)
    A = coefs[1]
    B = coefs[2]
    C = coefs[3]
    D = coefs[4]
    E = coefs[5]
    F = coefs[6]
    return (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F)
end

function matrix_rep(coefs::Vector{Float64})
    A = coefs[1]
    B = coefs[2]
    C = coefs[3]
    D = coefs[4]
    E = coefs[5]
    F = coefs[6]
    Q = [A B / 2 D / 2; B / 2 C E / 2; D / 2 E / 2 F]
    return (Q)
end

function quadratic_solver(xs, ys, f = -1.0)
    n = size(xs, 1)
    A = hcat(xs .^ 2, xs .* ys, ys .^ 2, xs, ys)
    b = ones(n)
    coefs = A \ b
    A = coefs[1]
    B = coefs[2]
    C = coefs[3]
    D = coefs[4]
    E = coefs[5]
    F = f
    Q = [A B / 2 D / 2; B / 2 C E / 2; D / 2 E / 2 F]
    return (Q)
end

function get_qf_properties(Q)
    center = inv(Q[1:2, 1:2]) * Q[1:2, 3]
    f = eigen(Q[1:2, 1:2])
    v = f.vectors
    d = f.values

    semiaxis_lengths = sqrt.(1.0 ./ d)
    p = sortperm(semiaxis_lengths)
    semiaxes = semiaxis_lengths[p]

    axis_vecs = v[:, p]

    rotation_angle_ccw = atan(axis_vecs[:, 1][1], axis_vecs[:, 1][2])

    return ((
        c = center,
        ax = axis_vecs,
        al = semiaxes,
        th = rotation_angle_ccw,
    ))
end

function quadratic_difference(A_q, x, y)

end

function get_observables(prop_list, ts, sequential_points)

end
