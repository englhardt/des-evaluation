# Domain Expansion Strategy Evaluation

This repository contains scripts and notebooks to reproduce the experiments and analyses of the paper

> Adrian Englhardt and Klemens Böhm. 2019. Exploring the Unknown - Query Synthesis in One-Class Active Learning.

For more information about this research project, see also the [project website](https://www.ipd.kit.edu/des/).
For a general overview and a benchmark on one-class active learning see the [OCAL project website](https://www.ipd.kit.edu/ocal/).

The analysis and main results of the experiments can be found under [notebooks](https://github.com/englhardt/des-evaluation/tree/master/notebooks):
  * `domain_expansion_strategy.ipynb`: Figure 3
  * `experiment_evaluation.ipynb`: Figure 4 and Table 1
  * `svdd_neg_eps.ipynb`: Example for `SVDDnegEps`

To execute the notebooks, make sure you follow the [setup](#setup), and download the [raw results](https://www.ipd.kit.edu/des/output.zip) into `data/output/`.

## Prerequisites

The experiments are implemented in [Julia](https://julialang.org/), and some the evaluation notebooks are written in Python.
This repository contains code to setup, execute and analyze the experiments.
The one-class classifiers (SVDDneg) and active learning methods (all query synthesis strategies) are implemented in two separate Julia packages: [SVDD.jl](https://github.com/englhardt/SVDD.jl) and [OneClassActiveLearning.jl](https://github.com/englhardt/OneClassActiveLearning.jl).

### Setup

Just clone the repo.
```bash
$ git clone https://github.com/englhardt/des-evaluation.git
```
* Experiments require Julia 1.1.0, requirements are defined in `Manifest.toml`. To instantiate, start julia in the `des-evaluation` directory with `julia --project` and run `julia> ]instantiate`. See [Julia documentation](https://docs.julialang.org/en/v1.0/stdlib/Pkg/#Using-someone-else's-project-1) for general information on how to setup this project.
* Notebooks require
  * Julia 1.1.0 (dependencies are already installed in the previous step)
  * Python 3.7 and `pipenv`. Run `pipenv install` to install all dependencies

### Repo Overview

* `data`
  * `input`
    * `raw`: unprocessed data files
      * `dami`: contains data set collections `literature` and `semantic` from the [DAMI](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/) repository
    * `processed`: output directory of _preprocessing_dami.py_
  * `output`: output directory of experiments; _generate_biased_sample_experiments.jl_ and _generate_gmm_holdout_experiments.jl_ create the folder structure and experiments; _run_experiments.jl_ writes results and log files
* `notebooks`: jupyter notebooks to analyze experimental results
  * `domain_expansion_strategy.ipynb`: Figure 3
  * `experiment_evaluation.ipynb`: Figure 4 and Table 1
  * `svdd_neg_eps.ipynb`: Example for `SVDDnegEps`
* `scripts`
  * `config`: configuration files for experiments
    * `config.jl`: high-level configuration, e.g., for number of workers
    * `config_eval_part_1.jl`: experiment config for synthetic data sets
    * `config_eval_part_2_qss.jl`: experiment config for real-world data sets
  * `biased_sample_utils.jl`: utilities to generate biased samples in existing data sets
  * `generate_biased_sample_experiments.jl`: generate experiments on real-world data
  * `generate_gmm_holdout_experiments.jl`: generates experiments on synthetic data
  * `gmm_utils.jl`: utilities to generate synthetic domain expansion problems
  * `preprocessing_dami.py`: preprocess DAMI data sets
  * `reduce_results`: combine result files into a single CSV
  * `run_experiments`: executes experiments

## Overview

Each step of the experiments can be reproduced, from the raw data files to the final plots that are presented in the paper.
The experiment is a pipeline of several dependent processing steps.
Each of the steps can be executed standalone, and takes a well-defined input, and produces a specified output.
The Section [Experiment Pipeline](#experiment-pipeline) describes each of the process steps.

Running the benchmark is compute intensive and takes many CPU hours.
Therefore, we also provide the [results to download](https://www.ipd.kit.edu/des/output.zip) (866 MB).
This allows to analyze the results in the notebooks without having to run the whole pipeline.

The code is licensed under a [MIT License](https://github.com/englhardt/des-evaluation/blob/master/LICENSE.md) and the result data under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
If you use this code or data set in your scientific work, please reference the companion paper.

## Experiment Pipeline

The experiment pipeline uses config files to set paths and experiment parameters.
There are two types of config files:
* `scripts/config.jl`: this config defines high-level information on the experiment, such as number of workers, where the data files are located, and log levels.
* `scripts/<config_eval_part_1|config_eval_part_2_qss>.jl`: These config files define the experimental grid, including the data sets, classifiers, and active-learning strategies.

1. _Data Preprocessing_: The preprocessing step transforms publicly available benchmark data sets into a common csv format, and performs feature selection.
   * **Input:** Download [semantic.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/semantic.tar.gz) and [literature.tar.gz](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/literature.tar.gz) containing the .arff files from the DAMI benchmark repository and extract into `data/input/raw/dami/<data set>` (e.g. `data/input/raw/dami/literature/ALOI/` or `data/input/raw/dami/semantic/Annthyroid`).
   * **Execution:**
   ```bash
      $ pipenv run preprocessing
   ```
   * **Output:** .csv files in `data/input/processed/dami/`

   We also provide our preprocessed data to [download](https://www.ipd.kit.edu/des/input.zip) (3.7 MB).

2. _Generate Experiments_: This step creates a set of experiments. For the synthetic evaluation the scripts generate the data as well.
   * **Input**: Full path to config file `<config_eval_part_1.jl|config_eval_part_2_qss.jl>` (e.g., config/config_eval_part_1.jl), preprocessed data files
   * **Execution:**
   ```bash
    $ julia --project scripts/generate_experiments.jl $(DIR)/scripts/config/config_eval_part_1.jl
    $ julia --project scripts/generate_experiments.jl $(DIR)/scripts/config/config_eval_part_2_qss.jl
   ```
   * **Output:**
     * Creates an experiment directory with the naming `<exp_name>`. The directories created contains several items:
       * `log` directory: skeleton for experiment logs (one file per experiment), and worker logs (one file per worker)
       * `results` directory: skeleton for result files
       * `experiments.jser`: this contains a serialized Julia Array with experiments. Each experiment is a Dict that contains the specific combination. Each experiment can be identified by a unique hash value.
       * `experiment_hashes`: file that contains the hash values of the experiments stored in `experiments.jser`
       * `generate_<gmm_holdout|biased_sample>_experiments.jl`: a copy of the file that generated the experiments
       * `<config_eval_part_1.jl|config_eval_part_2_qss.jl>`: a copy of the config file used to generate the experiments

3. _Run Experiments_: This step executes the experiments created in Step 2.
Each experiment is executed on a worker. In the default configuration, a worker is one process on the localhost.
For distributed workers, see Section [Infrastructure and Parallelization](#infrastructure-and-parallelization).
A worker takes one specific configuration, runs the active learning experiment, and writes result and log files.
  * **Input:** Generated experiments from step 2, full path to high-level config `scripts/config/config.jl`
  * **Execution:**
  ```bash
     $ julia --project scripts/run_experiments.jl $(DIR)/scripts/config/config.jl
  ```
  * **Output:** The output files are named by the experiment hash and are .json files (e.g., `data/output/eval_part_1/results/data/gmm_holdout_1_seed_2_dim_3_gaussians_1_num_gaussians_train_DecisionBoundaryQss_SVDDneg_16283024028153567650.json`)

4. _Reduce Results_: Merge of an experiment directory into one .csv by using summary statistics
    * **Input:** Full path to finished experiments.
    * **Execution:**
    ```bash
       $ julia --project scripts/reduce_results.jl </full/path/to/data/output>
    ```
    * **Output:** A result csv file, `data/output/output.csv`.

5. _Analyze Results:_ jupyter notebooks in the `notebooks`directory to analyze the reduced `.csv`. Run the following to produce the figures and tables in the experiment section of the paper:
  ```bash
    $ pipenv run evaluation
  ```

## Infrastructure and Parallelization

Step 3 _Run Experiments_ can be parallelized over several workers. In general, one can use any [ClusterManager](https://github.com/JuliaParallel/ClusterManagers.jl). In this case, the node that executes `run_experiments.jl` is the driver node. The driver node loads the `experiments.jser`, and initiates a function call for each experiment on one of the workers via `pmap`.

## Authors
This package is developed and maintained by [Adrian Englhardt](https://github.com/englhardt/)
