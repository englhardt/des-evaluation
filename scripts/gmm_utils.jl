using Distributions
using GaussianMixtures

struct DataGenGMM
    dimensions::Vector{Int}
    num_gaussians::Vector{Int}
    num_repetitions::Int
    num_data::Int
    outlier_threshold::Float64
    num_gaussians_train::Vector{Int}
end

function DataGenGMM(dimensions::Vector{Int}, num_gaussians::Vector{Int}, num_repetitions::Int, num_data::Int, outlier_threshold::Float64)
    return DataGenGMM(dimensions, num_gaussians, num_repetitions, num_data, outlier_threshold, num_gaussians)
end

function rand_gmm(; num_gaussians=1, d=2)
    μ = rand(Normal(0.5, 0.1), num_gaussians, d)
    Σ = Array{eltype(GaussianMixtures.FullCov{Float64})}(undef, num_gaussians)
    for i=1:num_gaussians
        T = rand(Normal(0, 0.1), d, d)
        Σ[i] = GaussianMixtures.cholinv(T' * T / d)
    end
    w = ones(num_gaussians) / num_gaussians
    hist = History("custom")
    return GMM(w, μ, Σ, [hist], 0)
end

function generate_gmm_test_data(gmm_sample::GMM, gmm_test::GMM, threshold::Float64, n_inlier::Int, n_outlier::Int)
    range_check(x) = vec(all(x .>= 0, dims=1)) .& vec(all(x .<= 1, dims=1))
    dist = MixtureModel(gmm_test)
    data_inliers, data_outliers = [], []
    while length(data_inliers) < n_inlier
        candidates = copy(transpose(rand(gmm_sample, n_inlier * 2)))
        mask = range_check(candidates) .& (pdf(dist, candidates) .>= threshold)
        candidates = candidates[:, mask]
        i = 1
        while i <= size(candidates, 2) && length(data_inliers) < n_inlier
            push!(data_inliers, candidates[:, i])
            i += 1
        end
    end
    @info "Inlier generation done."
    while length(data_outliers) < n_outlier
        candidates = rand(gmm_sample.d, n_outlier * 2)
        mask = range_check(candidates) .& (pdf(dist, candidates) .< threshold)
        candidates = candidates[:, mask]
        i = 1
        while i <= size(candidates, 2) && length(data_outliers) < n_outlier
            push!(data_outliers, candidates[:, i])
            i += 1
        end
    end
    @info "Outlier generation done."
    data = hcat(data_inliers..., data_outliers...)
    labels = vcat(fill(:inlier, n_inlier), fill(:outlier, n_outlier))
    return data, labels
end
generate_gmm_test_data(gmm::GMM, threshold::Float64, n_inlier::Int, n_outlier::Int) = generate_gmm_test_data(gmm, gmm, threshold, n_inlier, n_outlier)
generate_gmm_test_data(gmm::GMM, threshold::Float64, n::Int) = generate_gmm_test_data(gmm, threshold, n, n)

function deconstruct_gmm(g::GMM, components::Vector{Int})
    all([x ∈ Set(1:length(g.w)) for x in components]) || throw(ArgumentError("GMM does not contain all components: $components."))
    return GMM(ones(length(components)) / length(components), g.μ[components, :], g.Σ[components], g.hist, g.nx)
end

function gmm_to_string(g)
    s = join([g.n, g.d], ',') * '\n'
    s *= join(g.w, ',') * '\n'
    for i in 1:g.n
        s *= join(g.μ[i, :], ',') * '\n'
        s *= join(vcat(transpose(GaussianMixtures.covar(g.Σ[i]))...), ',') * '\n'
    end
    return s
end
