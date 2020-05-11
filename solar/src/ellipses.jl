function quad2std(Q)
    A = Q[1, 1]
    B = 2 * Q[1, 2]
    C = Q[2, 2]
    D = 2 * Q[1, 3]
    E = 2 * Q[2, 3]
    F = Q[3, 3]

    Q = [A B / 2; B / 2 C]
    b = [D; E]

    beta = Q \ b
    rhs = 0.25 * b' * beta - F

    S = Q / rhs
    center = -0.5 * beta
    return ([S, center])
end

function rotation_mat(angle::Real; ccw = true)
    rotate_mat = zeros(2, 2)
    if ccw
        rotate_mat = [
            cos(angle) -sin(angle)
            sin(angle) cos(angle)
        ]
    else
        rotate_mat = [
            cos(angle) sin(angle)
            -sin(angle) cos(angle)
        ]
    end

    return rotate_mat
end

function elementwise_pseudoinvert(v::AbstractArray, tol = 1e-10)
    m = maximum(abs.(v))
    if m == 0
        return v
    end
    v = v ./ m
    reciprocal = 1 ./ v
    reciprocal[abs.(reciprocal).>=1/tol] .= 0

    return reciprocal / m
end

function std2param(std)
    f = eigen(std[1])
    V = f.vectors
    D = f.values

    semiaxis_lengths = sqrt.(elementwise_pseudoinvert(D))
    p = sortperm(semiaxis_lengths, rev = true)
    sorted_semiaxes = semiaxis_lengths[p]
    sorted_eig_vecs = V[:, p]
    major_axis = sorted_eig_vecs[:, 1]
    ccw_angle = atan(major_axis[2], major_axis[1])

    return ([sorted_semiaxes, std[2], ccw_angle])
end

function quad2param(Q)
    qf = quad2std(Q)
    return std2param(qf)
end

function ellipse_to_plot_points(param; n = 1000::Int)
    theta_plot_vals = range(0, 2 * pi, length = n)
    unit_circle = [cos.(theta_plot_vals) sin.(theta_plot_vals)]'
    p = param
    onaxis_ellipse = vec(p[1]) .* unit_circle
    U = rotation_mat(p[3])
    rotated_ellipse = (U * onaxis_ellipse)'
    shifted_ellipse = vec(p[2])' .+ rotated_ellipse
    return shifted_ellipse
end
