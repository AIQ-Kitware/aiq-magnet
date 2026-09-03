# AIQ MAGNET 
Mathematical Assurance of Generative AI Network Evaluation Toolkit

## Introduction

This early version of the MAGNET package is intended to provide a look into how we're approaching TA1 evaluation (for algorithms that don't require model training or finetuning).  Currently we only provide a "predictor" style interface, but plan to extend the framework to support other TA1 algorithms that don't fit into this bucket.

**IMPORTANT:** As this is a preliminary release, interfaces are subject to change.

## Installing

The base package includes cards, claims, theory annotations, kwdagger
execution, and containers. HELM integration and endpoint leasing are optional:

```bash
pip install aiq-magnet              # cards, claims, theory, kwdagger, containers
pip install 'aiq-magnet[helm]'      # + HELM output loaders, predictors, materialize, demo data (crfm-helm, torch)
pip install 'aiq-magnet[leasing]'   # + infer-stack, for --per_node_leasing
pip install 'aiq-magnet[optional]'  # helm + leasing
pip install 'aiq-magnet[all]'       # everything, plus the test tools
```

Importing a HELM-backed module without the extra raises
``MissingOptionalDependency`` naming the extra to install.

## Developer Quick Start

Quick start: install and run tests

```bash
uv venv --python 3.11 --seed .venv-311-magnet
source .venv-311-magnet/bin/activate
uv pip install '.[all]'
pytest
```

## Running the examples

The examples below (which make use of the demo data) run [HELM](https://github.com/stanford-crfm/helm) on the backend the first time you run them.  Alternatively, the example predictors can be run against locally computed HELM outputs.

### Random Predictor

Both the `magnet/example_random_predictor.py` and `magnet/example_perturbation_predictor.py` examples include example invocations in their docstrings.  For example the random predictor's docstring:

```
    """
    Class to demonstrate a random stat prediction algorithm

    Example:
        >>> import magnet
        >>> outputs = magnet.HelmOutputs.demo()
        >>> suite_path = outputs.suites()[0].path
        >>> predictor_instance = ExampleRandomPredictor(num_eval_samples=5)
        >>> predictor_instance(suite_path)
    """
```

Which can be run with the following command (assuming you've followed the developer quick start instructions):

```
xdoctest magnet/example_random_predictor.py
```

In this example, we ask the framework to generate some demo data for us (which will run HELM on the backend).  After the demo data has been generated, we instantiate the `ExampleRandomPredictor` allowing it 5 response samples from the evaluation data.  Then we run the random predictor against the generated demo data, which should produce a `"predicted_exact_match"` metric in the form of a HELM `Stat` object, i.e.:

```
                                                                 run_spec  split    stat_name  predicted_mean  actual_mean perturbation_computed_on perturbation_name perturbation_fairness perturbation_robustness
0  mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2  valid  exact_match            0.42          0.0                     None              None                  None                    None
```

(Note that the exact values in your output may be different due to the random nature of this predictor)

### Perturbation Predictor

The perturbation predictor example builds a simple linear model with the strength of a "misspelling" perturbation to predict the `"exact_match"` score.  The example docstring is as follows:

```
    """
    Class to demonstrate a stat prediction algorithm based on strength of perturbation

    Example:
        >>> import magnet
        >>> outputs = magnet.HelmOutputs.demo(run_entries=["boolq:data_augmentation=misspelling_sweep,model=openai/gpt2"], max_eval_instances=20)
        >>> suite_path = outputs.suites()[0].path
        >>> predictor_instance = ExamplePerturbationPredictor(num_eval_samples=5)
        >>> predictor_instance(suite_path)
    """
```

Which can be run with the following command:

```
xdoctest magnet/example_perturbation_predictor.py
```

Note that in this example, we request demo data of the `boolq` scenario with `data_augmentation=misspelling_sweep` giving us outputs for a handful of perturbation strengths (in this case it's the probability that a given token is misspelled).  The rest of this example follows the same form as the random predictor example above.

Expected output for this example is:

```
                                                 run_spec  split    stat_name  predicted_mean  actual_mean perturbation_computed_on perturbation_name perturbation_fairness perturbation_robustness  
perturbation_prob
0  boolq,data_augmentation=misspelling0:model=openai_gpt2  valid  exact_match        0.663095         0.65                perturbed      misspellings                 False                    True                 
0
```

### Running on local HELM outputs

We also provide a command line interface to each of the example
predictors which allow you to point them at local precomputed HELM
outputs.

To run the random predictor against local outputs:
```
python magnet/example_random_predictor.py /path/to/benchmark_output/runs/name_of_suite
```

Run `python magnet/example_random_predictor.py --help` to see the full
list of arguments.

Note that many already computed HELM outputs (including for the `helm-lite` benchmark suite) are publicly available [here](https://console.cloud.google.com/storage/browser/crfm-helm-public).

## Implementing your own Predictor

The basic anatomy of a `Predictor` is as follows (assuming a clean Python file):

```
from magnet.predictor import RunPredictor, RunPrediction

class MyPredictor(Predictor):
    def predict(self,
                train_split: TrainSplit,
                sequestered_test_split: SequesteredTestSplit
                ) -> list[RunPrediction]:
        train_run_specs_df = train_split.run_specs
        train_scenario_states_df = train_split.scenario_state
        train_stats_df = train_split.stats

        eval_run_specs_df = sequestered_test_split.run_specs
        eval_scenario_state_df = sequestered_test_split.scenario_state

        # Interesting prediction algorithm code goes here
```

And this method should return a list of predictions (as
`RunPrediction` instances).  For example, assume we're only predicting
the `"exact_match"` stat:

```
    return [RunPrediction(run_spec_name=run_spec_name,
                          split="valid",
                          stat_name="exact_match",
                          mean=prediction)]
```

Where `split` should reflect dataset split at the HELM level (each record in the `*_scenario_states_df` dataframes indicates which split it belongs to).  And `mean` should be the predicted metric value mean.  The fields included above are the only required fields for a `RunPrediction`.

The arguments passed into the `predict` method are Pandas dataframes corresponding to the HELM data (flattened from it's nested form) for the relevant runs.  We've included an IPython notebook file here ([predict_inputs_exploration.ipynb](./predict_inputs_exploration.ipynb)) showing the exact form of the inputs to `predict`.

We also recommend looking at the `magnet/example_random_predictor.py` and/or `magnet/example_perturbation_predictor.py` examples to see what a complete (albeit simple) predictor looks like.

## Evaluation Cards

Verifiable empirical claims with symbol definitions are specified in Python and stored in structured `yaml` files called Evaluation Cards. Examples are provided in `magnet/cards`, including a simple dataset of integers and a particular benchmark from the latest HELM Lite runs.

### Simple Arithmetic Card
A basic example for getting familiar with the structure of an evaluation card is available at `magnet/cards/simple.yaml`. The claim tests the commutative property of consecutive integers on the range `[-10, 10]`. This maps to the symbol-based assertion `x + y = y + x`, when `x` is even integers `[-10, 10]` and `y` is odd integers `[-9, 11]`. An example usage of this card is provided in the `EvaluationCard` docstring:
```
    """
    Specification of an empirical claim with resolvable symbols and metadata

    Example:
        >>> from magnet.evaluation import EvaluationCard
        >>> card = EvaluationCard("magnet/cards/simple.yaml")
        >>> card.evaluate()
        VERIFIED
    """
```

Alternatively, you can evaluate any evaluation card using the `magnet evaluate` command. The following command evaluates the `simple.yaml` example card:

```
magnet evaluate magnet/cards/simple.yaml
```

In this example, we populate an `EvaluationCard` instance with the `simple.yaml` evaluation card, resolve the symbol values of the claim from their respective python definitions, and assert whether this claim was `VERIFIED` (true assertion), `FALSIFIED` (false assertion), or `INCONCLUSIVE` (failed). We can also call `.summarize()` to expose the contents of this card programmatically.
```
    >>> card.summarize() 
    Title:       Arithmetic - Addition Commutative Property
    Description: Addition is commutative on pairs of even and odd integers
    ================================
    SYMBOLS:     {'int_range_even': None, 'int_range_odd': None}
    CLAIM:       
    for even, odd in zip(int_range_even, int_range_odd):
        assert even + odd == odd + even, f"{even} + {odd} is not commutative"

    ================================
    STATUS:      UNVERIFIED
```
The above was called prior to `.evaluate()`, as shown by the unresolved (`None`) symbol values. A single `.evaluate()` call will execute the symbol definitions, run the claim, and print the result.
```
    >>> card.evaluate()
    VERIFIED

    >>> card.summarize()
    Title:       Arithmetic - Addition Commutative Property
    Description: Addition is commutative on pairs of even and odd integers
    ================================
    SYMBOLS:     {'int_range_even': [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], 'int_range_odd': [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11]}
    CLAIM:       
    for even, odd in zip(int_range_even, int_range_odd):
        assert even + odd == odd + even, f"{even} + {odd} is not commutative"

    ================================
    STATUS:      VERIFIED
```
Now, subsequent `.summarize()` calls for this instance will reflect the result of the claim subject to the symbol resolutions. 

### Llama Performance Consistency Card (HELM Lite)
The `magnet/cards/llama.yaml` card tests the claim that for a single benchmark, the entire llama model family performs consistently within a `threshold`. Specifically, the card reads helm-lite runs to verify that llama models achieve an `exact_match` score within `threshold` of each other on the MMLU benchmark. 

An example demonstration is provided below (assuming you've downloaded helm-lite runs to `/data/crfm-helm-public`):

```
    >>> from magnet.evaluation import EvaluationCard
    >>> card = EvaluationCard('magnet/cards/llama.yaml')
    >>> card.summarize()
    Title:       In-domain Model Consistency for Llama Family
    Description: Performance in a single domain benchmark should be consistent within a bound of variation for an entire model family

    ================================
    SYMBOLS:     {'threshold': 0.1, 'helm_runs_path': '/data/crfm-helm-public/lite/benchmark_output', 'run_specs': None, 'exact_match_scores': None}
    CLAIM:       
    for base_model, base_score in exact_match_scores:
    for comp_model, comp_score in exact_match_scores:
        assert abs(comp_score - base_score) < threshold, f"{comp_model} score ({comp_score:.2f}) exceeds consistency bound on {base_model} ({base_score:.2f})"

    ================================
    STATUS:      UNVERIFIED

    >>> card.evaluate()
    Assertion does not hold: meta/llama-3-70b score (0.69) exceeds consistency bound on meta/llama-2-13b (0.51)
    FALSIFIED
```
At least one pair of models in the llama family do not satisfy the assertion subject to the symbol values, therefore the claim is `FALSIFIED`.

The Llama examples require MMLU results from more than one HELM-Lite release.
See `magnet/examples/llama_consistency/README.md` for the exact incremental
download commands and a `materialize_helm_run` reuse smoke test.

Optionally, you could evaluate this card using the `magnet evaluate` command as follows:

```
magnet evaluate magnet/cards/llama.yaml
```

### Writing your own Evaluation Card
An `EvaluationCard` instance is expecting roughly the following structure in `yaml` format:

```
# Human-readable comments for distributing card

title: "A single line that clearly maps claim to context/implication"
description: |
  multi-line explanation of claim in natural language

  This is where you can discuss what conclusions are drawn from (dis)proving your claim

claim:
  python: |
    executable multi-line python assertion with failure handling

symbols: # list of symbols
  valid_python_variable:
    type: python.type
    depends_on: ['other_symbols_or_unspecified']
    python: |
      executable multi-line python that explicitly assigns valid_python_variable to a value with specified type

      context for any given symbol definition can be optionally passed through the depends_on field from other symbol
      assignment code blocks (e.g. imports/variables from other_valid_python_variable)
```

Once your card definition is complete, you can follow the basic workflow below to programatically inspect and evaluate the card.

```
from magnet.evaluation import EvaluationCard

card = EvaluationCard("path/to/mycard.yaml")

# print card contents with unresolved symbols
card.summarize()

# resolve symbols and execute claim
card.evaluate()

# expose resolved symbol definitions and claim status
card.summarize() 
```

Or evaluate from the command line using:

```
magnet evaluate path/to/mycard.yaml
```

#### Resolving Symbols as a Pipeline (kwdagger)
In the example above, symbols are explicitly defined in Python as code blocks, values, or sweeps (list) of values. [kwdagger](https://github.com/AIQ-Kitware/kwdagger) offers an alternative flexible approach to resolving symbols as pipelines of user scripts with a variety of backends (see [tutorials](https://github.com/AIQ-Kitware/kwdagger/tree/main/docs/source/manual/tutorials) for example definitions). MAGNET can dispatch these explicitly, by referencing a fully-defined pipeline, or generate from user-provided scaffolding in the Evaluation Card.

The example python module (`magnet/examples/llama_consistency`) represents how a user may structure their code for testing the claim seen in `magnet/cards/llama.yaml`. Each potential 'node', or script, of a pipeline satisfies the following conditions:
 1. defines a Python class with key, value (input, output) arguments
 2. writes relevant results to a file
 3. can run as a python script with its key, value pairs: `$ python ./code/script.py --key1 value1 ...`

#### Generated Pipeline (llama_consistency example)
 As a familiar example, `llama_predict.py` resolves `base_score` and `comp_score` by the same logic defined in the `llama.yaml` card. These symbols (along with `helm_runs_path`, `base_model`, `comp_model`, and `threshold`) are written to a unique and hashed result path for each particular setting of run arguments. An example command line usage is provided at the bottom of the file. 
 
 The `magnet/cards/llama_pipeline.yaml` card defines a one-node pipeline that invokes the `llama_predict.py` script directly with sweep combinations constructed from the provided parameter list. Each node will populate a `kwdagger.ProcessNode` definition and can accept suitable inputs (see `algo_params`, `perf_params`, and other tags in `kwdagger` docs) in the Evaluation Card. The format of an Evaluation Card to generate a pipeline from a script is outlined below: 

 ```
 ...
# Same fields as example card above with claim
...
pipeline:
  # Each unique key is a node
  first_node:
    # specify how code is called without arguments
    executable: 'python path/to/code/module/script.py'
    # performance dependent variables/arguments
    algo_params:
      dataset_name:
          - unique_benchmark
      # lists of parameters will expand into singular run combinations
      model_name:
          - openai/gpt-4o
          - meta/llama-3.3-70b
    # output filename
    out_paths:
      results_fpath: 'results.json'
...
symbols: # define any remaining values
  valid_python_variable:
    ...
```
Example output can be observed by running the example card `llama_pipeline.yaml`:
```
magnet evaluate magnet/cards/llama_pipeline.yaml --output_path './results'
```
A subdirectory for each unique sweep will be created in `{output_path}`.


#### Explicit kwdagger Pipeline (llama_consistency example)
For new cards, prefer a declarative `kwdagger` pipeline. The pipeline can live
in a separate YAML file, but small pipelines are often clearer inline in the
card. `magnet/examples/llama_consistency/llama_kwdagger.yaml` is a two-node
example: `llama_predict` writes model scores and `llama_compare` consumes that
artifact and writes the comparison used by the card.

The pipeline is the standard kwdagger YAML `nodes` / `edges` form. MAGNET adds
`result_node` to select the KWDagger aggregate rows used as claim evidence.
KWDagger owns result loading and qualified namespaces such as
`metrics.<result_node>.<field>`, `params.<node>.<field>`, and
`resolved_params.<node>.<field>`.

```yaml
claim:
  python: |
    assert metrics.second_node.score < 0.1

kwdagger:
  result_node: second_node
  pipeline:
    nodes:
      first_node:
        executable: 'python -m package.first_node'
        algo_params:
          model_name: null
        out_paths:
          result_fpath: result.json
        primary_out_key: result_fpath

      second_node:
        executable: 'python -m package.second_node'
        in_paths: [input_fpath]
        out_paths:
          result_fpath: result.json
        primary_out_key: result_fpath

    edges:
      - first_node.result_fpath -> second_node.input_fpath

  matrix:
    first_node.model_name:
      - openai/gpt-4o
      - meta/llama-3.3-70b
```

The example directory also contains a README with the exact HELM-Lite download, single-run materialization, and execution commands.

Run the Llama example with the kwdagger-native evaluator:

```
magnet evaluate_new magnet/examples/llama_consistency/llama_kwdagger.yaml \
    --output_path './results_kwdagger' \
    --backend serial
```

`evaluate_new` first submits the finite matrix requested by this invocation, then
uses KWDagger aggregate to discover currently available `result_node` rows from
the shared result store. A recipe can select `evidence.scope: all` (the default)
to evaluate all accumulated rows, or `evidence.scope: requested` to evaluate
only rows corresponding to result-node computations requested by that
invocation. Cached/skipped requested computations still qualify when their
output exists. The run's `requested_runs.json` records execution state
separately; execution failure does not count as a falsified claim.

During the migration, `magnet evaluate_legacy` names the historical evaluator and
`magnet evaluate` remains its compatibility alias. Both reject recipes with a
`kwdagger:` block and point to `magnet evaluate_new`. `magnet evaluate_new` is
the cleaner kwdagger-only path: execution parameters are passed directly with
`--params`, `--backend`, `--tmux_workers`, `--skip_existing`, `--cache`, and
`--max_configs` using KWDagger schedule semantics. Legacy `pipeline:`
computation or symbol sweeps are rejected. See the example README for the
complete setup, recomputation, and materialization commands.

Although varying slightly in methods, successful runs of `llama.yaml`, `llama_pipeline.yaml`, and `llama_kwdagger.yaml` should all yield a `FALSIFIED` aggregate result with output similar to below:
```
================================
Evidence Scope: requested
Available Evidence Rows: 36 (discovered: 36)
  Verified:     0.61
  Falsified:    0.39
  Inconclusive: 0.00
================================
```

`evaluate_new` keeps the existing MAGNET dashboard run-bundle contract:
`card.yaml`, `log`, `results/*/verdict.json`, and aggregate `verdict.json`.
Each per-evidence verdict retains the legacy `status`, `output`, `symbols`, and
`timestamp` fields. For the new evaluator, `symbols` contains resolved recipe
symbols plus the qualified KWDagger leaves actually consumed by the claim, so
the existing dashboard can display the concrete experiment inputs without a
new parser. The complete aggregate row is also recorded under `evidence`, and
the aggregate verdict records the evidence scope and request summary. The
`results/` directory is created even when no evidence is available.

Use `--provenance` to record caller-known facts that are not present in the
card or result, such as whether an endpoint alias resolved to real weights, a
simulator, or a replay fixture.

```bash
magnet evaluate_new card.yaml \
    --provenance '{endpoint: {kind: simulator, catalog: rehearsal}}'
```

The mapping is copied into aggregate `verdict.json` as `provenance`. Do not use
it to restate information MAGNET already has as ordinary invocation settings,
such as the kwdagger backend or container image.

### Downloading HELM results

We provide a utility to download precomputed HELM results. 

For a quick getting started, we can download the HELM lite results to `/data/crfm-helm-public`.

```bash
python -m magnet.backends.helm.download_helm_results --benchmark=lite --version=v1.13.0 --download-dir /data/crfm-helm-public
```

Using different command line options you can explore what data is available on
the remote, as well as download different benchmarks and versions or subsets of
results. For more details see:

```bash
python -m magnet.backends.helm.download_helm_results --help
```

## Roadmap

- More options for predict input (dataframes vs. HELM objects vs. dicts)
- Support for non-prediction style TA1 algorithms (feedback needed)
- Further evaluation card development and evaluation router implementation
- ...

## Citation

If you use MAGNET in your research, please cite our paper:

```bibtex
@inproceedings{crall2025magnet,
  title={{MAGNET}: Mathematical Assurance of Generative {AI} Network Evaluation Toolkit},
  author={Jon Crall and David Joy and Roderic Collins and Benjamin Fenelon and Anthony Hoogs and Brian H Hu},
  booktitle={NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling},
  year={2025},
  url={https://openreview.net/forum?id=ZypC0qCMhT}
}
```

## Acknowledgments

This material is based upon work supported by the Defense Advanced
Research Project Agency (DARPA) under Contract No. HR001125CE017. Any
opinions, findings and conclusions or recommendations expressed in
this material are those of the author(s) and do not necessarily
reflect the views of the Defense Advanced Research Project Agency
(DARPA).
