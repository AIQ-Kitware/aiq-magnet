# Introduction

This early version of the MAGNET package is intended to provide a look into how we're approaching TA1 evaluation (for algorithms that don't require model training or finetuning).  Currently we only provide a "predictor" style interface, but plan to extend the framework to support other TA1 algorithms that don't fit into this bucket.

**IMPORTANT:** As this is a preliminary release, interfaces are subject to change.

# Developer Quick Start

Quick start: install and run tests

```bash
uv venv --python 3.11 --seed .venv-311-magnet
source .venv-311-magnet/bin/activate
uv pip install .[tests]
pytest
```

# Running the examples

The examples below (which make use of the demo data) run [HELM](https://github.com/stanford-crfm/helm) on the backend the first time you run them.  Alternatively, the example predictors can be run against locally computed HELM outputs.

## Random Predictor

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

## Perturbation Predictor

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

## Running on local HELM outputs

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

# Implementing your own Predictor

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

Where `split` should reflect dataset split at the HELM level (each record in the `*_scenario_states_df` dataframes indiciates which split it belongs to).  And `mean` should be the predicted metric value mean.  The fields included above are the only required fields for a `RunPrediction`.

The arguments passed into the `predict` method are Pandas dataframes corresponding to the HELM data (flattened from it's nested form) for the relevant runs.  We've included an IPython notebook file here ([predict_inputs_exploration.ipynb](./predict_inputs_exploration.ipynb)) showing the exact form of the inputs to `predict`.

We also recommend looking at the `magnet/example_random_predictor.py` and/or `magnet/example_perturbation_predictor.py` examples to see what a complete (albeit simple) predictor looks like.

# Evaluation Cards

Verifiable empirical claims with symbol definitions are specified in Python and stored in structured `yaml` files called Evaluation Cards. Examples are provided in `magnet/cards`, including a simple dataset of integers and a particular benchmark from the latest HELM Lite runs.

## Simple Arithmetic Card
A basic example for getting familar with the structure of an evaluation card is available at `magnet/cards/simple.yaml`. The claim tests the commutative property of consecutive integers on the range `[-10, 10]`. This maps to the symbol-based assertion `x + y = y + x`, when `x` is even integers `[-10, 10]` and `y` is odd integers `[-9, 11]`. An example usage of this card is provided in the `EvaluationCard` docstring:
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

## Llama Performance Consistency Card (HELM Lite)
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
At least one pair of models in the llama family do not satisify the assertion subject to the symbol values, therefore the claim is `FALSIFIED`.

Optionally, you could evaluate this card using the `magnet evaluate` command as follows:

```
magnet evaluate magnet/cards/llama.yaml
```

(If you have not downloaded the entire helm-lite leaderboard, an example subset for this example can be downloaded to `/data/crfm-helm-public` using the following command:)
```
python -m magnet.backends.helm.download_helm_results /data/crfm-helm-public --benchmark=lite --version=v1.0.0 --runs regex:mmlu.*model=.*llama.*
```

## Writing your own Evaluation Card
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


## Downloading HELM results

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

# Roadmap

- More options for predict input (dataframes vs. HELM objects vs. dicts)
- Support for non-prediction style TA1 algorithms (feedback needed)
- Further evaluation card development and evaluation router implementation
- ...

# Acknowledgments

This material is based upon work supported by the Defense Advanced
Research Project Agency (DARPA) under Contract No. HR001125CE017. Any
opinions, findings and conclusions or recommendations expressed in
this material are those of the author(s) and do not necessarily
reflect the views of the Defense Advanced Research Project Agency
(DARPA).
