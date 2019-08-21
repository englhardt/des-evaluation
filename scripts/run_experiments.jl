# Command line args
!isempty(ARGS) || error("No config supplied.")
isfile(ARGS[1]) || error("Cannot read '$(ARGS[1])'")
isabspath(ARGS[1]) || error("Please use an absolute path for the config.")
println("Config supplied: '$(ARGS[1])'")
using SVDD, OneClassActiveLearning
include(ARGS[1])

# Load packages
using Serialization, Distributed, Logging

# setup workers
localhost = filter(x -> x[1] == "localhost", worker_list)
remote_servers = filter(x -> x[1] != "localhost", worker_list)

length(localhost) > 0 && addprocs(localhost[1][2], exeflags=exeflags)
length(remote_servers) > 0 && addprocs(remote_servers, sshflags=sshflags, exeflags=exeflags)

# validate package versions
@everywhere function get_git_hash(path)
    cmd = `git -C $path rev-parse HEAD`
    (gethostname(), strip(read(cmd, String)))
end

function setup_julia_environment()
    local_githash = get_git_hash(JULIA_ENV)[2]
    for id in workers()
        remote_name, remote_githash = remotecall_fetch(get_git_hash, id, JULIA_ENV)
        @assert remote_githash == local_githash "Host: $remote_name has version mismatch." *
                                                    "Hash is '$remote_githash' instead of '$local_githash'."
    end
end

# Load remote packages and functions
@info "Loading packages on all workers."
@everywhere using Pkg
setup_julia_environment()
@everywhere using SVDD, OneClassActiveLearning, Memento, Gurobi, Random
@everywhere import SVDD: RandomOCClassifier

@everywhere fmt_string = "[{name} | {date} | {level}]: {msg}"
@everywhere loglevel = "debug"

@everywhere function setup_logging()
    setlevel!(getlogger("root"), "error")
    setlevel!(getlogger(OneClassActiveLearning), loglevel)
    setlevel!(getlogger(SVDD), loglevel)
    return nothing
end

@everywhere function Memento.warn(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::Gurobi.GurobiError)
    Memento.warn(logger, "GurobiError(exit_code=$(error.code), msg='$(error.msg)')")
end

@everywhere function Memento.debug(logger::Logger, error::ErrorException)
    Memento.warn(logger, "Caught ErrorException, msg='$(error.msg)')")
end

@everywhere function run_experiment(experiment::Dict)
    # Make experiments deterministic
    Random.seed!(0)

    setup_logging()
    @info "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash]) starting"
    if isfile(experiment[:output_file])
        @info "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash]) skipped"
        return nothing
    end

    res = Result(experiment)
    try
        time_exp = @elapsed res = OneClassActiveLearning.active_learn(experiment)
        res.al_summary[:runtime] = Dict(:time_exp => time_exp)
    catch e
        res.status[:exit_code] = Symbol(typeof(e))
        @warn "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash]) error"
        @warn "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash]): $e"
        @warn "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash]): $(stacktrace(catch_backtrace()))"
    finally
        OneClassActiveLearning.write_result_to_file(experiment[:output_file], res)
    end
    @info "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash]) done"
    return nothing
end

# load and run experiments
all_experiments = []
for s in readdir(data_output_root)
    if occursin(".csv", s)
        continue
    end
    @info "Running experiments in directory $s"
    exp_dir = joinpath(data_output_root, s)
    @info "Loading experiments.jld"
    # load experiments
    experiments = deserialize(open(joinpath(exp_dir, "experiments.jser")))
    append!(all_experiments, experiments)
end
@info "Running $(length(all_experiments)) experiments."
@info "Running experiments..."
# Shuffle experiments
Random.seed!(0)
shuffle!(all_experiments)
pmap(run_experiment, all_experiments, on_error=ex->print("!!! ", ex))
@info "Done."

# cleanup
rmprocs(workers())
