module SequentialQP

using SparseArrays
using LinearAlgebra
using OSQP

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

function set_subproblem_constraint!(problem::SubProblem, c_ineq, c_eq, A_ineq, A_eq)
    problem.l[1:problem.n_ineq] = -c_ineq
    # upper side of ineq is already set to Inf
    problem.l[problem.n_ineq+1:end] = -c_eq
    problem.u[problem.n_ineq+1:end] = -c_eq

    problem.A[1:problem.n_ineq, :] = transpose(A_ineq)
    problem.A[problem.n_ineq+1:end, :] = transpose(A_eq)
end

function optimize(x, lambda, mu, objective_function, inequality_constraint, equality_constraint; ftol=1e-4)
    n_dec = length(x)
    n_ineq = length(lambda)
    n_eq = length(mu)

    options = Dict(:verbose => false)
    model = OSQP.Model()
    problem = SubProblem(n_dec, n_eq, n_ineq)

    W = Matrix(1.0I, n_dec, n_dec)
    x_pre = zeros(n_dec)
    fval_pre = Inf
    gval_pre = Inf
    hval_pre = Inf
    fgrad_pre::Vector{Float64} = zeros(n_dec)
    gjac_pre::Array{Float64, 2} = zeros(n_ineq, n_dec)
    hjac_pre::Array{Float64, 2} = zeros(n_eq, n_dec)

    counter = 0
    for i in 1:10
        fval, fgrad = objective_function(x)
        gval, gjac = inequality_constraint(x)
        hval, hjac = equality_constraint(x)
        lag_grad = fgrad .+ sum(broadcast(*, reshape(lambda, 1, :), gjac), dims=2) + sum(broadcast(*, reshape(mu, 1, :), hjac), dims=2)

        # damped BFGS update
        if counter > 0
            lag_grad_pre = fgrad_pre .+ sum(broadcast(*, reshape(lambda, 1, :), gjac_pre), dims=2) + sum(broadcast(*, reshape(mu, 1, :), hjac_pre), dims=2)
            s = x - x_pre
            y = lag_grad - lag_grad_pre
            sWs = transpose(s) * W * s
            sy = dot(s, y)
            isPositive = (sy - 0.2 * sWs < 0)
            theta = (isPositive ? 0.8 * sWs / (sWs - sy) : 1.0)
            r = theta * y + (1 - theta) * W * s
            W = W - (W*s*transpose(s)*W)/sWs + r*transpose(r)/dot(s, r)
        end

        # Now, make a quadratic subproblem
        # 1/2 p^T W p + ∇ f^T p 
        # s.t. Ak * p + c = 0
        set_subproblem_constraint!(problem, gval, hval, gjac, hjac)
        P = SparseMatrixCSC(W)
        q = vec(fgrad)
        Asp = SparseMatrixCSC(problem.A)


        OSQP.setup!(model; P=P, q=q, A=Asp, l=problem.l, u=problem.u, options...)
        results = OSQP.solve!(model)

        # update 
        x_pre = x
        fgrad_pre = fgrad
        gjac_pre = gjac
        hjac_pre = hjac

        x += results.x # p in Wright's book
        lambda = [results.y[1]] # dual for ineq
        mu = [results.y[2]] # dual for eq
        counter += 1
    end
    return x, problem
end

export lagrangian, hessian_finite_difference
export optimize

end # module
