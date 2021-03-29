using Revise
using SparseArrays
using LinearAlgebra
using OSQP

function objective_function(x)
    diag_mat = Diagonal([1., 1.])
    val = 0.5 * transpose(x) * diag_mat * x
    jac = reshape(diag_mat * x, :, 1)
    return val, jac
end

function equality_constraint(x)
    func = (x[1] - 1)^2 + 1 - x[2]
    jac = reshape([2 * x[1], -1], :, 1)
    return func, jac
end

lambda = [0.0]
x = [3.0, 3.0]

model = OSQP.Model()
for i in 1:100
    lag(x) = lagrangian(x, lambda, objective_function, equality_constraint)
    lagval, lagjac = lag(x)
    H = hessian_finite_difference(lag, x)

    P = SparseMatrixCSC(H)
    q = lagjac

    hval, hjac = equality_constraint(x)
    A = SparseMatrixCSC(transpose(hjac))
    q = x
    l = [-hval]
    u = l
    println(size(P))
    println(size(A))
    println(q)
    println(l)
    OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u)
    results = OSQP.solve!(model)
    x = results.x

    objective_function(x)

    println(results.x)
end



