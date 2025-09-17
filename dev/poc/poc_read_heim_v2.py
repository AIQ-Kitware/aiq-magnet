r"""
I'm exploring the json files in the heim outputs, and I've identified a per_instance_stats.json that seems to have a mapping from an instance id to a clip score (its not exactly that but its close, I have a mean and max clip score, which presumably were created over multiple outputs -- looks like 4 -- from the same text prompt as they are different).

However, I'm noticing that the mapping is many to 1 as some instance ids have multiple entries, some with no perturbation and others with a perturbation.

In order to run SRI's code, I need to effectively build a list of prompts and their corresponding clip scores.

The instances.json file has the original and perturbed input text mapped to instance-ids, but there doesn't seem to be a great way to join on perturbation details. To work around this I'm hashing the perurbation class to make a perturb id.

So now I have a list of text and corresponding clip scores. But as I go back to SRI's code, I see they do each score from the multiple outputs, which I don't think exists in the heim outputs. Maybe I can fudge it by building a normal distribution using the mean and max stats I do have, and then sampling 4 items from it, which is kinda iffy, but I'm not sure there is a better way forward.

Issue with getting the HELM results.

Perterbation ids seem to tell you that it is perterbed, e.g. style, but you
don't get a unique id for what the perterbation is. The json does have a
"modifications" field, but the dataclass does not give us that information.

---


Question for HELM maintainer

In the HEIM outputs, I see examples in the run_spec.json under "adapater" where
there is a field: "modifications". But I don't see any dataclass attribute that
has the same name in the HELM codebase, even when I check in older versions.

E.g.:

In v1.0.0/mscoco:model=AlephAlpha_m-vader,prompt=pixel_art,max_eval_instances=100 I have:

/data/crfm-helm-public/heim/benchmark_output/runs/v1.0.0/mscoco:model=AlephAlpha_m-vader,prompt=pixel_art,max_eval_instances=100/run_spec.json


```json
{
  "name": "mscoco:model=AlephAlpha_m-vader,prompt=pixel_art,max_eval_instances=100",
  "scenario_spec": {
    "class_name": "helm.benchmark.scenarios.mscoco_scenario.MSCOCOScenario",
    "args": {}
  },
  "adapter_spec": {
    "method": "generation",
    "global_prefix": "",
    "instructions": "",
    "input_prefix": "",
    "input_suffix": "",
    "reference_prefix": "A. ",
    "reference_suffix": "\n",
    "output_prefix": "",
    "output_suffix": "",
    "instance_prefix": "\n",
    "substitutions": [],
    "max_train_instances": 0,
    "max_eval_instances": 100,
    "num_outputs": 1,
    "num_train_trials": 1,
    "num_random_trials": 1,
    "sample_train": true,
    "model": "AlephAlpha/m-vader",
    "temperature": 1,
    "max_tokens": 0,
    "stop_sequences": [],
    "modifications": [
      "pixel art"
    ]
  },
  ...
}
```

I don't see where the "modifications" field is being written anywhere in HELM.


Also in

v1.0.0/mscoco:model=AlephAlpha_m-vader,data_augmentation=art,max_eval_instances=17

/data/crfm-helm-public/heim/benchmark_output/runs/v1.0.0/mscoco:model=AlephAlpha_m-vader,data_augmentation=art,max_eval_instances=17/run_spec.json

I have:

```
{
  "name": "mscoco:model=AlephAlpha_m-vader,data_augmentation=art,max_eval_instances=17",
  "scenario_spec": {
    "class_name": "helm.benchmark.scenarios.mscoco_scenario.MSCOCOScenario",
    "args": {}
  },
  "adapter_spec": {
    "method": "generation",
    "global_prefix": "",
    "instructions": "",
    "input_prefix": "",
    "input_suffix": "",
    "reference_prefix": "A. ",
    "reference_suffix": "\n",
    "output_prefix": "",
    "output_suffix": "",
    "instance_prefix": "\n",
    "substitutions": [],
    "max_train_instances": 0,
    "max_eval_instances": 17,
    "num_outputs": 4,
    "num_train_trials": 1,
    "num_random_trials": 1,
    "sample_train": true,
    "model": "AlephAlpha/m-vader",
    "temperature": 1,
    "max_tokens": 0,
    "stop_sequences": []
  },
```

But there is no modifications in the adapter_spec, they are in data_augmenter_spec.perturbation_specs


```
  "data_augmenter_spec": {
    "perturbation_specs": [
      {
        "class_name": "helm.benchmark.augmentations.style_perturbation.StylePerturbation",
        "args": {
          "modifications": [
            "oil painting"
          ]
        }
      },
      {
        "class_name": "helm.benchmark.augmentations.style_perturbation.StylePerturbation",
        "args": {
          "modifications": [
            "watercolor"
          ]
        }
      },
      {
        "class_name": "helm.benchmark.augmentations.style_perturbation.StylePerturbation",
        "args": {
          "modifications": [
            "pencil sketch"
          ]
        }
      },
      {
        "class_name": "helm.benchmark.augmentations.style_perturbation.StylePerturbation",
        "args": {
          "modifications": [
            "animation"
          ]
        }
      },
      {
        "class_name": "helm.benchmark.augmentations.style_perturbation.StylePerturbation",
        "args": {
          "modifications": [
            "vector graphics"
          ]
        }
      },
      {
        "class_name": "helm.benchmark.augmentations.style_perturbation.StylePerturbation",
        "args": {
          "modifications": [
            "pixel art"
          ]
        }
      }
    ],
    "should_augment_train_instances": false,
    "should_include_original_train": true,
    "should_skip_unchanged_train": true,
    "should_augment_eval_instances": true,
    "should_include_original_eval": true,
    "should_skip_unchanged_eval": true,
    "seeds_per_instance": 1
  },
```


What is the difference between run specs:

    'mscoco:model=AlephAlpha_m-vader,data_augmentation=spanish,max_eval_instances=100', and
    'mscoco:model=AlephAlpha_m-vader,data_augmentation=spanish,max_eval_instances=100,groups=mscoco_spanish'

####

Lets actually go through v1.1.0


/data/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/mscoco:model=AlephAlpha_m-vader,data_augmentation=art,max_eval_instances=17,groups=mscoco_art_styles/run_spec.json


Why is it that in v1.1.0 of HEIM results, the following run specs:

    'mscoco:model=AlephAlpha_m-vader,data_augmentation=chinese,max_eval_instances=100,groups=mscoco_chinese',
    'mscoco:model=AlephAlpha_m-vader,data_augmentation=dialect_prob=1.0_source=SAE_target=AAVE,max_eval_instances=100,groups=mscoco_dialect',
    'mscoco:model=AlephAlpha_m-vader,data_augmentation=gender_terms_prob=1.0_source=male_target=female,max_eval_instances=100,groups=mscoco_gender',
    'mscoco:model=AlephAlpha_m-vader,data_augmentation=hindi,max_eval_instances=100,groups=mscoco_hindi',
    'mscoco:model=AlephAlpha_m-vader,data_augmentation=robustness,max_eval_instances=100,groups=mscoco_robustness',
    'mscoco:model=AlephAlpha_m-vader,data_augmentation=spanish,max_eval_instances=100,groups=mscoco_spanish',
    'mscoco:model=AlephAlpha_m-vader,max_eval_instances=100,groups=mscoco_base'

All contain the same unmodified prompt "A man standing over several bunches of
green bananas."? From instance id id414595.  Shouldn't the augmentations be
modifying these prompts?

It seems maybe the instances table contains both modified and unmodified
versions of the prompts?
"""

import ubelt as ub
import magnet
import kwutil
from line_profiler import profile
from functools import cache
from magnet.utils.util_pandas import DotDictDataFrame


@cache
def cached_hash(text):
    return ub.hash_data(text)


@profile
def main():
    heim_output_dpath = ub.Path('/data/crfm-helm-public/heim/benchmark_output')
    outs = magnet.HelmOutputs(heim_output_dpath)
    suite = outs.suites('v1.1.0')[0]
    runs = suite.runs('*').existing()

    pman = kwutil.ProgressManager()
    with pman:
        tables = []
        for run in pman.progiter(runs, desc='Loading HELM runs'):
            table = load_run_clip_info(run)
            if table is not None:
                tables.append(table)
            else:
                print(f'Skip {run}')

    import pandas as pd
    big_table = pd.concat(tables).reset_index(drop=True)

    from helm.common.object_spec import parse_object_spec
    run_specs = {k: parse_object_spec(k) for k in big_table['run_spec.name'].unique()}
    big_table['run_spec.model'] = big_table['run_spec.name'].apply(lambda k: run_specs[k].args['model'])
    big_table['input_text_id'] = big_table['request_states.instance.input.text'].apply(cached_hash)

    import numpy as np
    unique_indices = get_unique_model_prompt_indices(big_table)
    unique_indices = np.array(unique_indices)
    unique_big_table = big_table.loc[unique_indices]

    for key, group in unique_big_table.groupby(['run_spec.model']):
        print(key, len(group))

        group.prefix_subframe('per_instance_stats.stat.max_clip_score')

        group['per_instance_stats.stat.max_clip_score.max']
        # We now have a list of prompts and scores.
        prompts = group['request_states.instance.input.text']
        scores = group['per_instance_stats.stat.expected_clip_score.max']


def get_unique_model_prompt_indices(big_table):
    unique_indices = []
    for _, group in big_table.groupby(['run_spec.model', 'input_text_id']):
        # Just take the first to resolve duplicates
        unique_indices.append(group.index[0])
        if len(group) > 1:
            # I don't 100% understand why these columns have different values.
            # Part of it is that sometimes an instance specifies the full
            # perturbation but other times it seems to be specified implicitly.
            ignore_prefix = [
                # There seems to be some duplication with
                # benchmark_output/scenarios/cub200/images/CUB_200_2011/images/079.Belted_Kingfisher/Belted_Kingfisher_0024_70538.jpg
                # benchmark_output/scenarios/cub200/images/CUB_200_2011/images/051.Horned_Grebe/Horned_Grebe_0068_35111.jpg
                'request_states.instance.references',

                'adapter_spec.max_eval_instances',
                # I guess some instance ids have the same text?
                'per_instance_stats.instance_id',
                'request_states.instance.id',
                'per_instance_stats.perturb_id',
                'per_instance_stats.perturb_instance_id',
                'per_instance_stats.perturb_instance_id',
                'per_instance_stats.perturbation',
                'per_instance_stats.stat.expected_clip_score.name.perturbation',
                'per_instance_stats.stat.max_clip_score.name.perturbation',
                'request_states.instance.input.original_text',
                'request_states.instance.perturb_id',
                'request_states.instance.perturb_instance_id',
                'request_states.instance.perturbation',
            ]
            to_drop = group.prefix_subframe(ignore_prefix).columns
            munged = group.drop(to_drop, axis=1)
            varied = munged.varied_value_counts(min_variations=2, on_error='placeholder')
            varied.pop('run_path', None)
            varied.pop('run_spec.name', None)
            if varied:
                constant = group.drop(varied.keys(), axis=1).iloc[0].to_dict()
                # Just skip ones we cant figure out
                blocklist = {
                    '32ab299758c04c9b6aa6858dffd879695b928196adf7e25eeda3cb5e66ce1a9a81cdb5aec56642eca20124f5af08755a1e565f629b3610ee0a433825ce80076e'
                }
                if constant['input_text_id'] in blocklist:
                    continue
                print(f'constant = {ub.urepr(constant, nl=2)}')
                print(f'varied = {ub.urepr(varied, nl=2)}')
                raise Exception
    return unique_indices


def load_run_clip_info(run):
    """
    Build a data frame where perterbations and instances have unique ids.

    Relevant HELM data structures we are working with:

        helm.benchmark.metrics.metric.PerInstanceStats
        helm.benchmark.scenarios.scenario.Instance
        helm.benchmark.adaptation.request_state.RequestState
        helm.benchmark.augmentations.perturbation_description.PerturbationDescription

    """
    import pandas as pd

    # Only load data that has these stats computed
    stats_of_interest = [
        'expected_clip_score',
        'max_clip_score',
        # 'expected_clip_score_multilingual',
        # 'max_clip_score_multilingual'
    ]

    filtered_stats = []
    per_instance_stats = run.json.per_instance_stats()
    for instance_stats in per_instance_stats:
        # For this instance, determine if any of its statistics are of
        # interest.
        relevant_stats = [
            stat for stat in instance_stats['stats']
            if stat['name']['name'] in stats_of_interest
        ]
        if relevant_stats:
            base = ub.udict.difference(instance_stats, {'stats'})
            # Include custom id for future use
            perturbation = instance_stats.get('perturbation', None)
            perturb_id = ub.hash_data(perturbation)
            perturb_instance_id = ub.hash_data([instance_stats['instance_id'], perturb_id])

            # Expand the relevant stats
            flat_stats = {}
            for stat in relevant_stats:
                stat_name = stat['name']['name']
                flat_stat = kwutil.DotDict.from_nested(stat, prefix=stat_name)
                flat_stats.update(flat_stat)

            new_info = {
                'per_instance_stats': {
                    **base,
                    'stat': flat_stats,
                    'perturb_id': perturb_id,
                    'perturb_instance_id': perturb_instance_id
                },
                'run_spec.name': run.name
            }
            filtered_stats.append(new_info)
    if len(filtered_stats) == 0:
        return None

    # Load up instance information from the scenario state
    filtered_states = []
    scenario_state = run.json.scenario_state()
    request_states = scenario_state.pop('request_states')
    for request_state in request_states:
        instance = request_state['instance']

        request_state.get('reference_index', None)
        request_state.get('train_trial_index', None)

        # instance
        perturbation = instance.get('perturbation', None)
        # if perturbation is not None:
        #     raise Exception
        #     ...
        perturb_id = ub.hash_data(perturbation)

        perturb_instance_id = ub.hash_data([instance['id'], perturb_id])
        instance['perturb_id'] = perturb_id
        instance['perturb_instance_id'] = perturb_instance_id
        new_state = {
            **scenario_state,
            'request_states': request_state}
        filtered_states.append(new_state)

    flat_filtered_stats = [kwutil.DotDict.from_nested(item) for item in filtered_stats]
    flat_filtered_states = [kwutil.DotDict.from_nested(item) for item in filtered_states]
    filtered_stats_df = DotDictDataFrame(flat_filtered_stats)
    filtered_states_df = DotDictDataFrame(flat_filtered_states)

    # filtered_stats_df.search_columns('_id')
    # filtered_states_df.search_columns('_id')

    for key, idxs in ub.find_duplicates(filtered_stats_df['per_instance_stats.perturb_instance_id']).items():
        filtered_stats_df.iloc[idxs].varied_values(min_variations=2)
        raise AssertionError('dups found')

    for key, idxs in ub.find_duplicates(filtered_states_df['request_states.instance.perturb_instance_id']).items():
        filtered_states_df.iloc[idxs].map(str).varied_values(min_variations=2)
        raise AssertionError('dups found')

    table = pd.concat([
        filtered_stats_df.set_index('per_instance_stats.perturb_instance_id', drop=False),
        filtered_states_df.set_index('request_states.instance.perturb_instance_id', drop=False),
    ], axis=1).reset_index(drop=True)
    table['run_path'] = run.path
    return table


if __name__ == '__main__':
    """
    CommandLine:
        kernprof -lzvv -p magnet ~/code/magnet-sys-exploratory/dev/poc/poc_read_heim_v2.py
    """
    main()
