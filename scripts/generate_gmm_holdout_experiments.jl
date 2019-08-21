# Command line args
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
config_file = ARGS[1]

include(joinpath(@__DIR__, "gmm_utils.jl"))
using SVDD, OneClassActiveLearning
using MLKernels
using Memento
using Serialization, DelimitedFiles, Logging, Random

Random.seed!(0)
include(config_file)

function generate_experiments(data_dir, data_gen_settings::DataGenGMM, data_split_strategy, initial_pool_strategy, solver, num_resamples_initial_pool=5)
    exp_dir = joinpath(data_output_root, exp_name)
    println(exp_dir)
    if isdir(exp_dir)
        print("Type 'yes' or 'y' to delete and overwrite experiment $(exp_dir): ")
        argin = readline()
        if argin == "yes" || argin == "y"
            rm(exp_dir, recursive=true)
        else
            error("Overwriting anyways... Just kidding, nothing happened.")
        end
    else
        !isdir(exp_dir) || error("The experiment directory $(exp_dir) already exists.")
    end

    isdir(exp_dir) || mkpath(exp_dir)
    mkpath(joinpath(exp_dir, "data"))
    mkpath(joinpath(exp_dir, "log", "experiment"))
    mkpath(joinpath(exp_dir, "log", "worker"))
    mkpath(joinpath(exp_dir, "results"))
    @info "Generating experiment directory with name: $exp_dir and config: $(config_file). ($num_resamples_initial_pool resamples of the initial pool)"

    experiments = []

    for x in 1:data_gen_settings.num_repetitions
        for dim in data_gen_settings.dimensions
            for num_g in data_gen_settings.num_gaussians
                @info "[Repetition $x] Starting data generation with $dim dimensions and $num_g gaussians."
                Random.seed!(x)
                gmm = rand_gmm(num_gaussians=num_g, d=dim)
                Random.seed!(x)
                data_test, labels_test = generate_gmm_test_data(gmm, data_gen_settings.outlier_threshold, data_gen_settings.num_data)
                oracle = QuerySynthesisGMMOracle(gmm, data_gen_settings.outlier_threshold)
                @info "[Repetition $x] Data generation done."

                for num_g_train in data_gen_settings.num_gaussians_train
                    gmm_train = deconstruct_gmm(gmm, collect(1:num_g_train))
                    data_train, labels_train = generate_gmm_test_data(gmm_train, gmm, data_gen_settings.outlier_threshold, data_gen_settings.num_data, 0)
                    data, labels = hcat(data_train, data_test), vcat(labels_train, labels_test)
                    train_mask = vcat(trues(length(labels_train)), falses(length(labels_test)))
                    split_strategy = DataSplits(train_mask, .~train_mask, LabeledSplitStrat(), FullSplitStrat())

                    @assert size(data_test, 2) == length(labels_test)
                    @assert size(data_train, 2) == length(labels_train)
                    @assert size(data_train, 1) == size(data_test, 1)
                    @assert size(data, 2) == length(labels)
                    @assert length(labels) == length(train_mask)

                    # write data file
                    data_file = joinpath(exp_dir, "data", "gmm_holdout_$(x)_seed_$(dim)_dim_$(num_g)_gaussians_$(num_g_train)_num_gaussians_train.csv")
                    writedlm(data_file, hcat(data', labels), ',')

                    for ip in initial_pool_strategy
                        ip[1] == "Pnin" || error("Cannot generate experiment with initial pools '$(ip[1])'.")
                        haskey(ip[2], :n) || error("Initial pools '$(ip[1])' has no parameter 'n'.")
                        num_train = ip[2][:n]
                        for n in 1:num_resamples_initial_pool
                            Random.seed!(n)
                            initial_pools = get_initial_pools(data, labels, split_strategy, ip[1], n=num_train)
                            @assert length(labels) == length(initial_pools)

                            param = Dict(:num_al_iterations => num_al_iterations,
                                         :solver => solver,
                                         :initial_pools => initial_pools,
                                         :adjust_K => true,
                                         :initial_pool_resample_version => n,
                                         :classify_precision => classify_precision)

                            for model in models
                                for init_strategy in init_strategies
                                    for query_strategy in query_strategies
                                        out_dir = split(data_file, '/')[end-1]
                                        output_path = joinpath(exp_dir, "results", out_dir)
                                        isdir(output_path) || mkdir(output_path)
                                        experiment = Dict{Symbol, Any}(
                                                :data_file => data_file,
                                                :data_set_name => out_dir,
                                                :split_strategy_name => "Shl",
                                                :initial_pool_strategy_name => ip,
                                                :model => merge(model, Dict(:init_strategy => init_strategy)),
                                                :query_strategy => Dict(:type => query_strategy[:type],
                                                                        :param => query_strategy[:param]),
                                                :split_strategy => split_strategy,
                                                :oracle => oracle,
                                                :param => param)

                                        exp_hash = hash(sprint(print, experiment))
                                        @show data_file, ip, n, exp_hash
                                        @assert exp_hash == hash(sprint(print, deepcopy(experiment)))
                                        experiment[:hash] = "$exp_hash"

                                        out_name = splitext(splitdir(data_file)[2])[1]
                                        out_name = joinpath(output_path, "$(out_name)_$(query_strategy[:type])_$(model[:type])_$(exp_hash).json")
                                        experiment[:output_file] = out_name
                                        experiment[:log_dir] = joinpath(exp_dir, "log")
                                        push!(experiments, deepcopy(experiment))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # save the experimental setup
    cp(@__FILE__, joinpath(exp_dir, splitdir(@__FILE__)[2]), follow_symlinks=true)
    cp(config_file, joinpath(exp_dir, splitdir(config_file)[2]), follow_symlinks=true)

    @info "Created $exp_dir with $(length(experiments)) instances."

    open(joinpath(exp_dir, "experiment_hashes"), "a") do f
        for e in experiments
            write(f, "$(e[:hash])\n")
        end
    end
    serialize(open(joinpath(exp_dir, "experiments.jser"), "w"), experiments)
    return experiments
end

generate_experiments(data_dir, data_gen_settings, split_strategy, initial_pool_strategy, solver, num_resamples_initial_pool)
