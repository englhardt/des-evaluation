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
        :type => PoolOracle,
        :param => Dict{Symbol, Any}()
    )
]

#### query strategies ####
query_strategies = [
    Dict(:type => :RandomPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :RandomOutlierPQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :HighConfidencePQs, :param => Dict{Symbol, Any}()),
    Dict(:type => :DecisionBoundaryPQs, :param => Dict{Symbol, Any}())
]

num_al_iterations = 100
num_resamples_initial_pool = 10
exp_name = "eval_part_2_pqs"
