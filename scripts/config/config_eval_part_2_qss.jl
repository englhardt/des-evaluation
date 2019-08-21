include(joinpath(@__DIR__, "config.jl"))
using JuMP, Gurobi, MLKernels

solver = with_optimizer(Gurobi.Optimizer; OutputFlag=0, Threads=1)

### data settings ###
data_sets = ["Annthyroid", "Arrhythmia", "Arrhythmia", "Cardiotocography", "Glass",
             "HeartDisease", "Hepatitis", "InternetAds", "Ionosphere", "PageBlocks",
             "Parkinson", "Pima", "SpamBase", "Stamps", "WPBC", "Wilt"]
data_dirs = [joinpath("dami", x) for x in data_sets]
data_info = "dami_outlier_detection"

### learning scenario ###
initial_pool_strategy = [("Pnin", Dict(:n => 25))]
split_strategy = [("Sl", Dict())]
initial_sample_strat = :nn_biased_sample

#### models ####
models = [Dict(:type => :SVDDneg, :param => Dict{Symbol, Any}())]
init_strategies = [WangCombinedInitializationStrategy(solver, 2.0.^range(-4, stop=4, step=1.0), FixedCStrategy(1.0))]
classify_precision = SVDD.OPT_PRECISION

#### oracle ####
oracle_param = [
    Dict{Symbol, Any}(
        :type => QuerySynthesisCVWrapperOracle,
        :param => Dict{Symbol, Any}(
            :subtype => QuerySynthesisSVMOracle,
        )
    ),
]

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
num_resamples_initial_pool = 10
exp_name = "eval_part_2_qss"
