using DataAssim
using ForwardDiff
using ReverseDiff
using LinearAlgebra
using StaticArrays
using DifferentialEquations

using Plots
plotlyjs();



###

# state size x
n = 2;

# number of observation per time instance
m = 1;

# observation operator
H = [1 0];
𝓗 = ModelMatrix(H)

# initial condition
xi = [1; 1];

# error covariance of the initial condition
Pi = Matrix(I, n, n)

# error covariance of the observations
R = 0.1 * Matrix(I, m, m)

ℳ = ModelMatrix([1 0.1; -0.1 1])

nmax = 100;
no = 10:5:nmax
yo = randn(m, length(no))


xai, = fourDVar(xi, Pi, ℳ, yo, R, H, nmax, no);
Q = zeros(size(Pi))
xa, = FreeRun(ℳ, xai, Q, H, nmax, no)

#𝓗
#𝓜
xa3, = KalmanFilter(xi, Pi, ℳ, Q, yo, R, H, nmax, no);
# should be ~0
#@test xa[:,end] ≈ xa3[:,end]  atol=1e-5
time = 1:nmax+1

plot(time, xa3[1, :], label = "KF")
plot!(time, xa[1, :], label = "4DVar")
plot!(time[no], yo[1, :], label = "observations")
#legend()
#grid("on")
###

function fc_op(x)
    tf_matrix = [1 0.1; -0.1 1]
    return (tf_matrix * x)
end

function l_forecast(t, x, η)
    return fc_op(x)
end

function tangent_l_model(t, x, dx)
    jac = ForwardDiff.jacobian(fc_op, x)
    return (jac * dx)
end

function adj_model(t, x, dx)
    jac = ForwardDiff.jacobian(fc_op, x)
    return (transpose(jac) * dx)
end

M = ModelFun(l_forecast, tangent_l_model, adj_model)

H = [1 0]

n = 2
m = 1

R = 2 * Matrix(I, m, m)

MM = ModelMatrix([1 0.1; -0.1 1])

ξ = [1; 1]
Π = Matrix(I, n, n)

nmax = 100;
no = 10:5:nmax

yo = randn(m, length(no))

Q = zeros(size(Π))

xai, = fourDVar(ξ, Π, M, yo, R, H, nmax, no)
xaim, = fourDVar(ξ, Π, MM, yo, R, H, nmax, no)

xa, = FreeRun(M, xai, Q, H, nmax, no)
xam, = FreeRun(MM, xaim, Q, H, nmax, no)

plot(time, xa[1, :], label = "4DVar")
#plot!(time, xam[1, :], label = "4DVarm")
plot!(time[no], yo[1, :], label = "observations")
###

mutable struct PendulumModel <: AbstractModel
    dt::Float64
    g::Float64
    l::Float64
    # m::Float64
    # b::Float64
end

#=
x'' + b/mx' + g/l sin(x) = 0
take b = 0
x'' = - g/l sin(x)
write
x' = y
y' = x'' = -g/l sin(x)
=#

function pendulum_sys(x)
    a = x[1]
    b = x[2]
    tr_mat = [1 0; 0 1]
    aff = [b; -sin(a)]
    return (aff)
end

function pendulum_forecaster(t, x, η, dt)
    k1 = dt * pendulum_sys(x)
    k2 = dt * pendulum_sys(x + k1 / 2)

    nx = x + k2
    return (nx)
end

function pendulum_linear_tangent(t, x, dx)
    jac = ForwardDiff.jacobian(pendulum_sys, x)
    return (jac * dx)
end

function pendulum_adjoint(t, x, dx)
    jac = ForwardDiff.jacobian(pendulum_sys, x)
    return (transpose(jac) * dx)
end

# function pendulum_linear_tangent(t, x, dx)
#     Ddxdt = similar(x)
#     Ddxdt[1] = dx[2]
#     Ddxdt[2] = -sin(dx[1])
#     return Ddxdt
# end

# function pendulum_adjoint(t, x, dx)
#     #Ddxdt[1] = dx[2]
#     ddx = similar(x)
#     ddx[2] = dx[1]
#     ddx[1] = -sin(dx[2])
#     return ddx
# end

function pendulum_lt_step(t, x, dt, f, Dx, Df)
    k1 = dt * f(t, x)
    Dk1 = dt * Df(t, x, Dx)

    Dk2 = dt * Df(t + dt / 2, x + k1 / 2, Dx + Dk1 / 2)
    Dxn = Dx + Dk2
    return Dxn
end

function pendulum_adj_step(t, x, dt, f, Dxn, Df_adj)
    k1 = dt * f(t, x)

    Dtmp2 = Df_adj(t + dt / 2, x + k1 / 2, dt * Dxn)
    Dx = Dxn + Dtmp2
    Dk1 = Dtmp2 / 2
    Dx = Dx + dt * Df_adj(t, x, Dk1)

    return Dx
end

import DataAssim.tgl
import DataAssim.adj

function (ℳ::PendulumModel)(t, x, η = zeros(eltype(x), size(x)))
    return (pendulum_forecaster(t, x, η, ℳ.dt))
end

function tgl(ℳ::PendulumModel, t, x, dx::AbstractVector)
    f_tgl(t, x, dx) = pendulum_linear_tangent(t, x, dx)
    f(t, x) = pendulum_sys(x)

    return pendulum_lt_step(t, x, ℳ.dt, f, dx, f_tgl)
end

function adj(ℳ::PendulumModel, t, x, dx::AbstractVector)
    f_adj(t, x, dx) = pendulum_adjoint(t, x, dx)
    f(t, x) = pendulum_sys(x)

    return pendulum_adj_step(t, x, ℳ.dt, f, dx, f_adj)
end

u0 = [0, pi / 4]
tspan = (0.0, 10.0)

function pendulum(du, u, p, t)
    theta = u[1]
    dtheta = u[2]
    du[1] = dtheta
    du[2] = -sin(theta)
end

penprob = ODEProblem(pendulum, u0, tspan)
soln = solve(penprob)

#state size
n = 2

#num obs per time
m = 1

#obs operator
H = [1 0]

#initial condition
ξ = [0; pi / 4]

#error covariance of initial condition
Π = [0.001 0; 0 1]

#error covariance of observations
r = 0.05
R = r * Matrix(I, m, m)

dt_obs = 1.
t = collect(1.:dt_obs:10.)
obs = [v[1] for (u, v) in tuples(soln(t))]
n_obs = obs + (sqrt(r) * randn(length(obs)))

ℳ = PendulumModel(0.01, 1, 1)

x = zeros(2, 1000)
x[:,1] = ξ
for k = 1:size(x,2)-1
    x[:,k+1] = ℳ(k, x[:,k])
end
plot(x[1,:])

steps = floor(Int64, tspan[2] / ℳ.dt)
o_arr = reshape(n_obs, 1, :)


xmmxm, = fourDVar(ξ, Π, ℳ, o_arr, R, H, steps, t)
Q = zeros(size(Π))
xaaxa, = FreeRun(ℳ, xmmxm, Q, H, steps, t)

time = collect(0:0.01:10)
soln_time = [v[1] for (u, v) in tuples(soln(time))]

plot(time, xaaxa[1,:], label = "4DVar")
plot!(t, n_obs, label = "Observations")
plot!(time, soln_time, label = "Ground Truth")
