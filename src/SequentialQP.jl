module SequentialQP

using SparseArrays
using LinearAlgebra
using OSQP

function lagrangian(x, lambda, f, h)
    fval, fjac = f(x)
    hval, hjac = h(x)
    lag_val = fval .- dot(lambda, hval)
    lag_jac = fjac .- sum(broadcast(*, reshape(lambda, 1, :), hjac), dims=2)
    return lag_val, lag_jac
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

export lagrangian, hessian_finite_difference

end # module
