using DifferentialEquations
using BenchmarkTools

function derivs(dy,y,p,t)
    n = 8
    if y[3] >= 1
        for i in 1:n
            dy[i] = 0
        end
    else
        if y[3] > 1e-8
            dy[1] = - sqrt(y[3] / (4 * pi))
        else
            dy[1] = 0
        end

        dy[2] = y[2] * y[3]
        dy[3] = y[3] * (y[4] + 2 * y[3])
        dy[4] = 2 * y[5] - 5 * y[3] * y[4] - 12 * y[3] * y[3]

        for i in 5:(n-1)
            dy[i] = (0.5 * (i-4) * y[4] + (i-5) * y[3]) * y[i] + y[i+1]
        end

        dy[n] = (0.5 * (n-4) * y[4] + (n-5) * y[3]) * y[n]
    end
end

condition(y, t, integrator) = y[3] - 1
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

y0 = [0.00000000000000000e+00; 1.00000000000000000e+00; 6.18030713497347561e-02; 1.96951848408954078e-01; 2.31052934228472423e-02; -1.45983228440004012e-02; 9.78980674130092415e-04; 2.11226691201307683e-04]
prob = ODEProblem(derivs, y0, (1000, 0))

