using Revise
using SparseArrays
using LinearAlgebra
using OSQP
using SequentialQP
using Revise

struct QuadraticForm
    P::Array{Float64, 2}
end
hessian(qf::QuadraticForm) = P

function (qf::QuadraticForm)(x) 
    val = 0.5 * transpose(x) * qf.M * x
    grad = qf.M * x
    return val, grad
end

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

mutable struct EvaluationCache
    fval::Union{Float64, Nothing}
    fgrad::Union{Vector{Float64}, Nothing}
    gval::Union{Vector{Float64}, Nothing}
    gjac::Union{Array{Float64, 2}, Nothing}
    hval::Union{Vector{Float64}, Nothing}
    hjac::Union{Array{Float64, 2}, Nothing}
end
EvaluationCache() = EvaluationCache(nothing, nothing, nothing, nothing, nothing, nothing)

function clear_cache!(eval_cache::EvaluationCache) 
    for field_name in fieldname(EvaluationCache)
        setfield!(eval_cache, field_name, nothing)
    end
end

mutable struct SQPWorkspace
    n_dec::Int
    n_ineq::Int
    n_eq::Int

    lambda::Vector{Float64} # ineq dual
    mu::Vector{Float64} # eq dual
    eval_cache::EvaluationCache
end

function SQPWorkspace(n_dec::Int, n_ineq::Int, n_eq::Int)
    lambda = zeros(n_ineq)
    mu = zeros(n_eq)
    SQPWorkspace(n_dec, n_ineq, n_eq, lambda, mu, EvaluationCache())
end

function create_subploblem(workspace::SQPWorkspace, x, f, g, h)
    fval, fgrad = f(x)
    gval, gjac = g(x)
    hval, hjac = h(x)

    # compute approximiate lagrangian hessian
    fterm = hessian(f)
    gterm = transpose(gjac) * gjac
    hterm = transpose(hjac) * hjac
    W = fterm + lambda 
end

function approximiate_lagrangian_hessian(workspace::SQPWorkspace, x, f, g, h)
    fval, fgrad = f(x)
end

mutable struct SubProblem
    n_dec::Int
    n_eq::Int
    n_ineq::Int
    #P::Array{Float64, 2}
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

function set_subproblem_constraint!(problem::SubProblem, g, h, x)
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
    x = [20, 20.0]
    
    z = vcat(x, lambda)

    options = Dict(:verbose => false)
    model = OSQP.Model()
    n_dec = 2
    n_eq = 1
    n_ineq = 1
    problem = SubProblem(n_dec, n_eq, n_ineq)

    W = Matrix(1.0I, n_dec, n_dec)
    x_pre = zeros(n_dec)

    use_bfgs = true

    for i in 1:10
        lag_x(x) = SequentialQP.lagrangian(x, lambda, mu, objective_function, inequality_constraint, equality_constraint)
        _, lag_grad = lag_x(x)
        _, lag_grad_pre = lag_x(x_pre) # TODO 

        # damped BFGS update
        if use_bfgs
            s = x - x_pre
            x_pre = x
            y = lag_grad - lag_grad_pre
            sWs = transpose(s) * W * s
            sy = dot(s, y)
            isPositive = (sy - 0.2 * sWs < 0)
            theta = (isPositive ? 0.8 * sWs / (sWs - sy) : 1.0)
            r = theta * y + (1 - theta) * W * s
            W = W - (W*s*transpose(s)*W)/sWs + r*transpose(r)/dot(s, r)
        else
            W = SequentialQP.hessian_finite_difference(lag_x, x)
        end

        c_ineq, A_ineq = inequality_constraint(x)
        c_eq, A_eq = equality_constraint(x)
        f, df = objective_function(x)
        # Now, make a quadratic subproblem
        # 1/2 p^T W p + ∇ f^T p 
        # s.t. Ak * p + c = 0

        set_subproblem_constraint!(problem, inequality_constraint, equality_constraint, x)
        P = SparseMatrixCSC(W)
        q = vec(df)
        Asp = SparseMatrixCSC(problem.A)


        OSQP.setup!(model; P=P, q=q, A=Asp, l=problem.l, u=problem.u, options...)
        results = OSQP.solve!(model)

        x += results.x # p in Wright's book
        lambda = [results.y[1]] # dual for ineq
        mu = [results.y[2]] # dual for eq
        println(x)
    end

    return x, problem
end

using BenchmarkTools
x, problem = example()
