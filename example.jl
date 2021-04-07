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


function example()

    n_objective = 2
    n_eq = 1

    lambda = [0] # somehow lambda must be 0 here
    x = [1.466666, -1.133333]
    
    z = vcat(x, lambda)

    options = Dict(:verbose => false)

    for i in 1:200
        model = OSQP.Model()
        lag_x(x) = SequentialQP.lagrangian(x, lambda, objective_function, equality_constraint)
        W = SequentialQP.hessian_finite_difference(lag_x, x)
        c, A = equality_constraint(x)
        f, df = objective_function(x)
        # Now, make a quadratic subproblem
        # 1/2 p^T W p + ∇ f^T p 
        # s.t. Ak * p + c = 0

        P = SparseMatrixCSC(W)
        q = vec(df)
        Asp = SparseMatrixCSC(transpose(A))
        l = [-c]
        u = [-c]
        OSQP.setup!(model; P=P, q=q, A=Asp, l=l, u=u, options...)
        results = OSQP.solve!(model)

        # the KKT optimality condition
        println(W * results.x + df + A * results.y)
        println(dot(A, results.x) + c)

        x += results.x # p in Wright's book
        lambda = results.y # thd dual
        println(x)
        println(lambda)
    end

    return x
end

x = example()


