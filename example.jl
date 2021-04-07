using Revise
using SparseArrays
using LinearAlgebra
using OSQP
using SequentialQP

function objective_function(x)
    diag_mat = Diagonal([1., 1.])
    val = 0.5 * transpose(x) * diag_mat * x
    jac = reshape(diag_mat * x, :, 1)
    return val, jac
end

function equality_constraint(x)
    func = (x[1] - 1)^2 + 1 - x[2]
    jac = reshape([2 * (x[1] - 1.0), -1], :, 1)
    return func, jac
end

function inequality_constraint(x)
    func = x[1] - 3
    jac = reshape([1.0, 0.0], :, 1)
    return func, jac
end

function example()

    n_objective = 2
    n_eq = 1

    lambda = [0] # somehow lambda must be 0 here
    mu = [0]
    x = [1.466666, -1.133333]
    
    z = vcat(x, lambda)

    options = Dict(:verbose => false)

    for i in 1:10
        model = OSQP.Model()
        lag_x(x) = SequentialQP.lagrangian(x, lambda, mu, objective_function, inequality_constraint, equality_constraint)
        W = SequentialQP.hessian_finite_difference(lag_x, x)
        c_ineq, A_ineq = inequality_constraint(x)
        c_eq, A_eq = equality_constraint(x)
        f, df = objective_function(x)
        # Now, make a quadratic subproblem
        # 1/2 p^T W p + ∇ f^T p 
        # s.t. Ak * p + c = 0

        P = SparseMatrixCSC(W)
        q = vec(df)
        A = hcat(A_ineq, A_eq)
        Asp = SparseMatrixCSC(transpose(A))
        l = [-c_ineq, -c_eq]
        u = [Inf, -c_eq]
        OSQP.setup!(model; P=P, q=q, A=Asp, l=l, u=u, options...)
        results = OSQP.solve!(model)

        x += results.x # p in Wright's book
        lambda = [results.y[1]] # dual for ineq
        mu = [results.y[2]] # dual for eq
        println(x)
        println(lambda)
    end

    return x
end

x = example()
