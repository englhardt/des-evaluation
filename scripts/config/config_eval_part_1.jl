include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "..", "gmm_utils.jl"))
using JuMP, Gurobi, MLKernels

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

### data settings ###
data_dir = "gmm"
data_info = "gmm_outlier_detection"

data_gen_dimensions = [2, 4, 6, 8, 10]
data_gen_num_gaussians = [3, 5, 7]
data_gen_repetions = 100
data_gen_num_data = 1000
data_gen_threshold = 0.1
data_gen_num_gaussians_train = [1]
data_gen_settings = DataGenGMM(data_gen_dimensions, data_gen_num_gaussians,
                               data_gen_repetions, data_gen_num_data, data_gen_threshold,
                               data_gen_num_gaussians_train)

### learning scenario ###
initial_pool_strategy = [("Pnin", Dict(:n => 25))]
split_strategy = [("Sl", Dict())]

#### models ####
models = [Dict(:type => :SVDDneg, :param => Dict{Symbol, Any}())]
init_strategies = [WangCombinedInitializationStrategy(solver, 2.0.^range(-4, stop=4, step=1.0), FixedCStrategy(1.0))]
classify_precision = SVDD.OPT_PRECISION

#### oracle ####
# the oracle is initialized during the data generation process

#### query strategies ####
basline_query_strategies = [
    Dict(:type => :RandomQss, :param => Dict{Symbol, Any}(:epsilon => 0.1)),
    Dict(:type => :RandomOutlierQss, :param => Dict{Symbol, Any}(:epsilon => 0.1))
]

fancy_query_strategies = [
    Dict(:type => :DecisionBoundaryQss, :param => Dict{Symbol, Any}()),
    Dict(:type => :ExplorativeMarginQss, :param => Dict{Symbol, Any}(:solver => solver, :lambda => 1.0, :use_penalty => false)),
    Dict(:type => :ExplorativeMarginQss, :param => Dict{Symbol, Any}(:solver => solver, :lambda => 1.0, :use_penalty => true)),
]

qs_optimizers = [
    ParticleSwarmOptimization(eps=1.0, swarmsize=1_000, maxiter=1_000),
    BlackBoxOptimization(:dxnes, eps=1.0),
]

query_strategies = basline_query_strategies

for qs in fancy_query_strategies
    for o in qs_optimizers
        qs_parametrized = deepcopy(qs)
        qs_parametrized[:param][:optimizer] = o
        push!(query_strategies, qs_parametrized)
    end
end

num_al_iterations = 100
num_resamples_initial_pool = 1
exp_name = "eval_part_1"
