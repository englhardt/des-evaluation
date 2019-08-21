
function density_biased_sample(data, labels, num_samples=25; from_oversample=false, kde_bw="scott")
    indices = collect(1:length(labels))
    inlier_indices = indices[labels .== :inlier]
    d = data[:, inlier_indices]
    p = multi_kde(d, kde_bw)(d)
    ranking = sortperm(p)
    if from_oversample
        selection = ranking[1:min(end, num_samples * 2)]
        selection = selection[randperm(length(selection))[1:min(end, num_samples)]]
    else
        selection = ranking[1:num_samples]
    end
    return inlier_indices[selection]
end

import NearestNeighbors

function nn_biased_sample(data, labels, num_samples=25; kwargs...)
    indices = collect(1:length(labels))
    inlier_indices = indices[labels .== :inlier]
    d = data[:, inlier_indices]
    kdtree = NearestNeighbors.KDTree(d)
    neighbors = NearestNeighbors.knn(kdtree, d, length(inlier_indices), true)[1]

    selection = Set([rand(1:length(inlier_indices))])
    last = first(selection)
    for _ in 1:num_samples-1
        for i in 2:length(inlier_indices)
            candidate = neighbors[last][i]
            if candidate ∉ selection
                push!(selection, candidate)
                last = candidate
                break
            end
        end
    end
    selection = collect(selection)
    return inlier_indices[selection]
end
