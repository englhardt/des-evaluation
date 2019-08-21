# Command line args
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
config_file = ARGS[1]

using SVDD, OneClassActiveLearning
using MLKernels
using Memento
using Serialization, Logging, Random

Random.seed!(0)
include(config_file)
include(joinpath(@__DIR__, "biased_sample_utils.jl"))

function generate_experiments(data_files, data_split_strategy, initial_pool_strategy, solver, oracle_param, num_resamples_initial_pool=5, initial_sample_strat=:density_biased_sample)
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
    mkpath(joinpath(exp_dir, "log", "experiment"))
    mkpath(joinpath(exp_dir, "log", "worker"))
    mkpath(joinpath(exp_dir, "results"))
    @info "Generating experiment directory with name: $exp_dir and config: $(config_file). ($num_resamples_initial_pool resamples of the initial pool)"

    experiments = []

    for data_file in data_files
        data, labels = load_data(data_file)
        if !isa(oracle_param, Array)
            oracle_param = [oracle_param]
        end
        for cur_oracle in oracle_param
            Random.seed!(0)
            @info "Learning oracle $(cur_oracle[:type])."
            @info cur_oracle[:type], cur_oracle[:param], data_file
            oracle = OneClassActiveLearning.initialize_oracle(cur_oracle[:type], data, labels, cur_oracle[:param])
            if isa(oracle, QuerySynthesisSVMOracle)
                gamma = oracle.classifier.gamma
            elseif isa(oracle, QuerySynthesisOCCOracle)
                gamma = MLKernels.getvalue(oracle.classifier.kernel_fct.alpha)
            else
                @info "Cannot find gamma in oracle. Using default 'scott'."
                gamma = "scott"
            end
            @info "Learning oracle done."
            @info "Note that parameter 'split_strategy' is ignored."
            @info "Initial sample strat '$initial_sample_strat'."
            for ip in initial_pool_strategy
                for n in 1:num_resamples_initial_pool
                    Random.seed!(n)
                    ip[1] == "Pnin" || error("Cannot generate experiment with initial pools '$(ip[1])'.")
                    haskey(ip[2], :n) || error("Initial pools '$(ip[1])' has no parameter 'n'.")
                    if isa(oracle, PoolOracle)
                        # Train on labeled data, test on full data and query on full data
                        split_strategy = DataSplits(trues(length(labels)), LabeledSplitStrat(), FullSplitStrat(), FullSplitStrat())
                    else # Query synthesis
                        # Train on labeled data, test on full data and query with labeled data
                        split_strategy = DataSplits(trues(length(labels)), LabeledSplitStrat(), FullSplitStrat(), LabeledSplitStrat())
                    end
                    num_train = ip[2][:n]
                    initial_pools = fill(:U, length(labels))
                    sample_indices = eval(initial_sample_strat)(data, labels, num_train, from_oversample=true, kde_bw=gamma)
                    initial_pools[sample_indices] .= :Lin

                    @assert size(data, 2) == length(labels)
                    @assert length(labels) == length(split_strategy.train)
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
                                        :split_strategy_name => "Sl",
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

all(isdir.(joinpath.(data_input_root, data_dirs))) || error("Not all data dirs are valid.")
data_files = vcat(([joinpath.(data_input_root, x, readdir(joinpath(data_input_root, x))) for x in data_dirs])...)
println("Found $(length(data_files)) data files.")

generate_experiments(data_files, split_strategy, initial_pool_strategy, solver, oracle_param, num_resamples_initial_pool, initial_sample_strat)
