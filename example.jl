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

using BenchmarkTools

lambda = [0]
mu = [0]
x = [20, 20.0]
    
SequentialQP.optimize(x, lambda, mu, objective_function, inequality_constraint, equality_constraint)
