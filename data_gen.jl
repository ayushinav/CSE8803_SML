using Pkg;
Pkg.activate(".");

using LinearAlgebra
using Distributions
using CairoMakie
using JLD2

include("../mt_fwd.jl")

n_layers = 50
dz = 50
h = dz .+ zeros(n_layers - 1)
μrange = -2:0.1:5;

"""
`∂(n)`: returns a `n`x`n` matrix for a 1D finite difference stencil using 2 points.
"""
function ∂(n)
    D = I(n) .+ 0.01
    D .= D .- 0.01
    for i in 2:n
        D[i, i - 1] = -1
    end
    # D[1, 1] = 0.0
    return D
end

L = ∂(n_layers)

ξ_dist = product_distribution([Uniform(-2, 5)
                               [Normal(0, 1.0) for i in 1:(n_layers - 1)]...])

ξ_dist = product_distribution([Normal(0, 1.0) for i in 1:(n_layers)])

L_inv = inv(L);

fig = Figure(; backgroundcolor=:transparent)
ax = Axis(fig[1, 1]; xscale=log10, backgroundcolor=:transparent)

σ(x) = inv(1 + exp(-x))

i = 1
while i <= 1000
    ξ = rand(ξ_dist)
    μ = 10.0^(rand(Uniform(0, 3)))
    m = L_inv * ξ ./ sqrt(μ)
    @. m = σ(m) * 7 - 2
    plot_model!(ax, m, h; color=:steelblue3, alpha=0.4)
    i += 1
end

xlims!(ax, [1e-2, 1e5])
ax.xticklabelcolor = :steelblue3
ax.yticklabelcolor = :steelblue3

fig
# save("assets/apriori_v2.png", fig)
## data gen

n_data = 10_000
data_m = zeros(n_layers, n_data);
μ_vec = zeros(Float32, n_data);

i = 1
while i <= n_data
    ξ = rand(ξ_dist)
    μ = 10.0^(rand(Uniform(0, 3)))
    m = L_inv * ξ ./ sqrt(μ)
    m .= σ.(m) .* 7 .- 2
    data_m[:, i] .= m
    μ_vec[i] = μ
    i += 1
end

data_m .= data_m .|> Float32;
@show mean(data_m, dims = 2)
@show std(data_m, dims = 2)

T = 10.0 .^ collect(-2:0.1:4);
ω_grid = Float32.(2π ./ T);
nf = length(ω_grid)
data_appres = zeros(Float32, nf, n_data);
data_phase = zeros(Float32, nf, n_data);

@time for i in 1:n_data
    @views forward!(data_appres[:, i], data_phase[:, i], data_m[:, i], h, ω_grid)
end

data_m[:, 1]
data_appres[:, 1]
data_phase[:, 1]
jldsave("data_mt.jld2"; data_m, data_appres, data_phase, ω_grid, μ_vec)
