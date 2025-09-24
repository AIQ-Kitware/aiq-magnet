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
        >>> suite = outputs.suites()[0].name
        >>> root_dir = outputs.root_dir
        >>> predictor_instance = ExampleRandomPredictor(num_eval_samples=5)
        >>> predictor_instance(root_dir, suite)
    """
```

Which can be run with the following command (assuming you've followed the developer quick start instructions):

```
xdoctest magnet/example_random_predictor.py
```

In this example, we ask the framework to generate some demo data for us (which will run HELM on the backend).  After the demo data has been generated, we instantiate the `ExampleRandomPredictor` allowing it 5 response samples from the evaluation data.  Then we run the random predictor against the generated demo data, which should produce a `"predicted_exact_match"` metric in the form of a HELM `Stat` object, i.e.:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ run_spec                                                               ┃ source    ┃ name        ┃ mean  ┃ min   ┃ max   ┃ count ┃ stddev ┃ sum   ┃ sum_squared ┃ variance ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2 │ predicted │ exact_match │ 0.720 │ 0.720 │ 0.720 │ 1     │ 0.000  │ 0.720 │ 0.518       │ 0.000    │
│ mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2 │ actual    │ exact_match │ 0.000 │ 0.000 │ 0.000 │ 1     │ 0.000  │ 0.000 │ 0.000       │ 0.000    │
└────────────────────────────────────────────────────────────────────────┴───────────┴─────────────┴───────┴───────┴───────┴───────┴────────┴───────┴─────────────┴──────────┘
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
        >>> suite = outputs.suites()[0].name
        >>> root_dir = outputs.root_dir
        >>> predictor_instance = ExamplePerturbationPredictor(num_eval_samples=5)
        >>> predictor_instance(root_dir, suite)
    """
```

Which can be run with the following command:

```
xdoctest magnet/example_perturbation_predictor.py
```

Note that in this example, we request demo data of the `boolq` scenario with `data_augmentation=misspelling_sweep` giving us outputs for a handful of perturbation strengths (in this case it's the probability that a given token is misspelled).  The rest of this example follows the same form as the random predictor example above.

Expected output for this example is:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ run_spec                                               ┃ source    ┃ name        ┃ mean  ┃ min   ┃ max   ┃ count ┃ stddev ┃ sum   ┃ sum_squared ┃ variance ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ boolq,data_augmentation=misspelling0:model=openai_gpt2 │ predicted │ exact_match │ 0.663 │ 0.663 │ 0.663 │ 1     │ 0.000  │ 0.663 │ 0.440       │ 0.000    │
│ boolq,data_augmentation=misspelling0:model=openai_gpt2 │ actual    │ exact_match │ 0.650 │ 0.650 │ 0.650 │ 1     │ 0.000  │ 0.650 │ 0.423       │ 0.000    │
└────────────────────────────────────────────────────────┴───────────┴─────────────┴───────┴───────┴───────┴───────┴────────┴───────┴─────────────┴──────────┘
```

## Running on local HELM outputs

We also provide a command line interface to each of the example
predictors which allow you to point them at local precomputed HELM
outputs.

To run the random predictor against local outputs:
```
python magnet/example_random_predictor.py --root-dir /path/to/benchmark_output --suite name_of_suite
```

Run `python magnet/example_random_predictor.py --help` to see the full
list of arguments.

Note that many already computed HELM outputs (including for the `helm-lite` benchmark suite) are publicly available [here](https://console.cloud.google.com/storage/browser/crfm-helm-public).

# Implementing your own Predictor

The basic anatomy of a `Predictor` is as follows (assuming a clean Python file):

```
from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor

class MyPredictor(Predictor):
    def predict(self,
                train_run_specs_df,
                train_scenario_states_df,
                train_stats_df,
                eval_run_specs_df,
                eval_scenario_states_df) -> dict[str, list[Stat]]:
        # Interesting prediction algorithm code goes here
```

And this method should return a list of predicted stats (as HELM `Stat`
instances).  For example, assume we're predicting the `"exact_match"`
stat:

```
return {run_spec_name: [Stat(**{'name': {'name': 'predicted_exact_match',
                                         'split': 'valid'},
                                'count': 1,
                                'sum': prediction,
                                'sum_squared': prediction ** 2,
                                'min': prediction,
                                'max': prediction,
                                'mean': prediction,
                                'variance': 0.0,
                                'stddev': 0.0})]}
```

**NOTE:** In order for the `Predictor` superclass to match the predicted stats with the actual stats from eval data, the `name.name` should be the same (apart from a `'predicted_'` prefix), and any other parameters under the `name` field should match as well.

The arguments passed into the `predict` method are Pandas dataframes corresponding to the HELM data (flattened from it's nested form) for the relevant runs.  We've included an IPython notebook file here ([predict_inputs_exploration.ipynb](./predict_inputs_exploration.ipynb)) showing the exact form of the inputs to `predict`.

We also recommend looking at the `magnet/example_random_predictor.py` and/or `magnet/example_perturbation_predictor.py` examples to see what a complete (albeit simple) predictor looks like.

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
- Expose model weights for a given run
- Evaluation card & router implementations
- ...

# Acknowledgments

This material is based upon work supported by the Defense Advanced
Research Project Agency (DARPA) under Contract No. HR001125CE017. Any
opinions, findings and conclusions or recommendations expressed in
this material are those of the author(s) and do not necessarily
reflect the views of the Defense Advanced Research Project Agency
(DARPA).
