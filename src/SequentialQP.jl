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

export lagrangian, hessian_finite_difference

end # module
