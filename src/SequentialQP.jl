module SequentialQP

using SparseArrays
using LinearAlgebra
using OSQP

function lagrangian(x, lambda, mu, f, g, h)
    fval, fjac = f(x)
    gval, gjac = g(x)
    hval, hjac = h(x)
    lag_val = fval .+ dot(lambda, gval) .+ dot(mu, hval)

    gterm = sum(broadcast(*, reshape(lambda, 1, :), gjac), dims=2)
    hterm = sum(broadcast(*, reshape(mu, 1, :), hjac), dims=2)
    lag_jac_x = fjac .+ gterm .+ hterm
    return lag_val, lag_jac_x
end

function hessian_finite_difference(lag, x0)
    eps = 1e-7
    dim = length(x0)
    hess = zeros(dim, dim)
    _, jac0 = lag(x0)
    for i in 1:dim
        x1 = copy(x0)
        x1[i] += eps
        _, jac1 = lag(x1)
        hess[i, :] = (jac1 - jac0)/eps
    end
    return hess
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

function optimize(x, lambda, mu, objective_function, inequality_constraint, equality_constraint; use_bfgs=true, ftol=1e-4)
    n_dec = length(x)
    n_ineq = length(lambda)
    n_eq = length(mu)

    options = Dict(:verbose => false)
    model = OSQP.Model()
    problem = SubProblem(n_dec, n_eq, n_ineq)

    W = Matrix(1.0I, n_dec, n_dec)
    x_pre = zeros(n_dec)

    for i in 1:10
        lag_x(x) = SequentialQP.lagrangian(x, lambda, mu, objective_function, inequality_constraint, equality_constraint)
        _, lag_grad = lag_x(x)
        _, lag_grad_pre = lag_x(x_pre) # TODO 

        # damped BFGS update
        if use_bfgs
            s = x - x_pre
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

        x_pre = x
        x += results.x # p in Wright's book
        lambda = [results.y[1]] # dual for ineq
        mu = [results.y[2]] # dual for eq
    end
    return x, problem
end

export lagrangian, hessian_finite_difference
export optimize

end # module
