using Revise
using SparseArrays
using LinearAlgebra
using OSQP
using SequentialQP
using Revise

function objective_function(x)
    diag_mat = Diagonal([1., 1.])
    val = 0.5 * transpose(x) * diag_mat * x
    jac = reshape(diag_mat * x, :, 1)
    return val, jac
end

function equality_constraint(x)
    func = [(x[1] - 1)^2 + 1 - x[2]]
    jac = reshape([2 * (x[1] - 1.0), -1], :, 1)
    return func, jac
end

function inequality_constraint(x)
    func = [x[1] - 3]
    jac = reshape([1.0, 0.0], :, 1)
    return func, jac
end

mutable struct SubProblem
    n_dec::Int
    n_eq::Int
    n_ineq::Int
    A::Array{Float64, 2}
    l::Vector{Float64}
    u::Vector{Float64}
end

function SubProblem(n_dec::Int, n_eq::Int, n_ineq::Int)
    n_total = n_ineq + n_eq
    l = zeros(n_total)
    u = zeros(n_total)
    u[1:n_ineq] .= Inf
    A = zeros(n_total, n_dec)
    SubProblem(n_dec, n_eq, n_ineq, A, l, u)
end

function set_subproblem_constraint(problem::SubProblem, g, h, x)
    c_ineq, A_ineq = g(x)
    c_eq, A_eq = h(x)

    problem.l[1:problem.n_ineq] = -c_ineq
    # upper side of ineq is already set to Inf
    problem.l[problem.n_ineq+1:end] = -c_eq
    problem.u[problem.n_ineq+1:end] = -c_eq

    problem.A[1:problem.n_ineq, :] = transpose(A_ineq)
    problem.A[problem.n_ineq+1:end, :] = transpose(A_eq)
end

function example()
    n_objective = 2
    n_eq = 1

    lambda = [0] # somehow lambda must be 0 here
    mu = [0]
    x = [1.466666, -1.133333]
    
    z = vcat(x, lambda)

    options = Dict(:verbose => false)
    model = OSQP.Model()
    n_dec = 2
    n_eq = 1
    n_ineq = 1
    problem = SubProblem(n_dec, n_eq, n_ineq)

    for i in 1:10
        lag_x(x) = SequentialQP.lagrangian(x, lambda, mu, objective_function, inequality_constraint, equality_constraint)
        W = SequentialQP.hessian_finite_difference(lag_x, x)
        c_ineq, A_ineq = inequality_constraint(x)
        c_eq, A_eq = equality_constraint(x)
        f, df = objective_function(x)
        # Now, make a quadratic subproblem
        # 1/2 p^T W p + ∇ f^T p 
        # s.t. Ak * p + c = 0

        set_subproblem_constraint(problem, inequality_constraint, equality_constraint, x)
        P = SparseMatrixCSC(W)
        q = vec(df)
        Asp = SparseMatrixCSC(problem.A)


        OSQP.setup!(model; P=P, q=q, A=Asp, l=problem.l, u=problem.u, options...)
        results = OSQP.solve!(model)

        x += results.x # p in Wright's book
        lambda = [results.y[1]] # dual for ineq
        mu = [results.y[2]] # dual for eq
    end

    return x
end

using BenchmarkTools
@benchmark x = example()
