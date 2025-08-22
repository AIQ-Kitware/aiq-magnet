# Introduction

This early version of the MAGNET package is intended to provide a look into how we're approach TA1 evaluation (for algorithms that don't require model training or finetuning).  Currently we only provide a "predictor" style interface, but will extend the framework to support other TA1 algorithms that don't fit into this bucket.

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

In this example, we ask the framework to generate some demo data for us (which will run HELM on the backend).  After the demo data has been generate, we instantiate the `ExampleRandomPredictor` allowing it 5 response samples from the evaluation data.  Then we run the random predictor against the generated demo data, which should produce a `"predicted_exact_match"` metric in the form of a HELM `Stat` object, i.e.:

```
[{'name': 'predicted_exact_match', 'split': 'valid'}[min=0.82, mean=0.82, max=0.82, sum=0.82 (1)]]
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
[{'name': 'predicted_exact_match', 'split': 'valid', 'perturbation': {'name': 'misspellings', 'robustness': True, 'fairness': False, 'computed_on': 'perturbed', 'prob': 0.05}}[min=0.653, mean=0.653, max=0.653, sum=0.653 (1)]]

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

The basic anatomy of a `Pedictor` is as follows (assuming a clean Python file):

```
from helm.benchmark.metrics.statistic import Stat

from magnet.predictor import Predictor

class MyPredictor(Predictor):
    def predict(self,
                train_run_specs_df,
                train_scenario_states_df,
                train_stats_df,
                eval_run_specs_df,
                eval_scenario_states_df) -> List[Stat]:
        # Interesting prediction algorithm code goes here
```

And this method should return a list of predicted stats (as HELM Stat
instances).  For example, assume we're predicting the `"exact_match"`
stat:

```
return [Stat(**{'name': {'name': 'predicted_exact_match',
                         'split': 'valid'},
                'count': 1,
                'sum': prediction,
                'sum_squared': prediction ** 2,
                'min': prediction,
                'max': prediction,
                'mean': prediction,
                'variance': 0.0,
                'stddev': 0.0})]
```

The arguments passed into the `predict` method are Pandas dataframes corresponding to the HELM data (flattened from it's nested form) for the relevant runs.  We've included an IPython notebook file here ([predict_inputs_exploration.ipynb](./predict_inputs_exploration.ipynb)) showing the exact form of the inputs to `predict`.

We also recommend looking at the `magnet/example_random_predictor.py` and/or `magnet/example_perturbation_predictor.py` examples to see what a complete (albeit simple) predictor looks like.
