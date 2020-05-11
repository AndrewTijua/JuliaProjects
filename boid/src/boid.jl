mutable struct boid
    position::Vector{Float64}
    velocity::Vector{Float64}
    acceleration::Vector{Float64}
    sight::Float64
    max_vel::Float64
end

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

function close_boids(boidlist)
    num_boids = length(boidlist)
    close_list = [[] for boid in boidlist]
    ct = 0
    for boid in boidlist
        ct += 1
        for otherboid in boidlist
            if boid != otherboid
                if fast_norm(boid.position - otherboid.position) <= boid.sight
                    push!(close_list[ct], otherboid)
                end
            end
        end
    end
    return close_list
end

function boid_separate(boid, closeboids)
    sep_force = [0.0 for elem in boid.velocity]
    if length(closeboids) == 0
        return sep_force
    end
    for n_boid in closeboids
        sep = (boid.position - n_boid.position)
        sep_force += sep / fast_norm_sq(sep)
    end
    return sep_force / length(closeboids)
end

function boid_centroid(boid, closeboids)
    cent_force = [0.0 for elem in boid.velocity]
    centroid = [0.0 for elem in boid.velocity]
    if length(closeboids) == 0
        return cent_force
    end
    for n_boid in closeboids
        centroid += n_boid.position
    end
    centroid /= length(closeboids)
    cent_force += centroid - boid.distance
    return cent_force
end

function boid_align(boid, closeboids)
    align_force = [0.0 for elem in boid.velocity]
    alignment = [0.0 for elem in boid.velocity]
    if length(closeboids) == 0
        return align_force
    end
    for n_boid in closeboids
        alignment += n_boid.velocity
    end
    alignment /= length(closeboids)
    align_force += alignment - boid.velocity
    return align_force
end

function step_time(boid, nearby_boids, timestep = 0.1)
    f_vec = [0.0, 0.0, 0.0]
    group_centroid = [0.0, 0.0, 0.0]
    for near in nearby_boids
        d_vec = near.position - boid.position
        group_centroid += d_vec
        distsq = fast_norm_sq
        f_vec += d_vec / distsq
    end
end
