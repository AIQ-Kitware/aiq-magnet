r"""
I'm exploring the json files in the heim outputs, and I've identified a per_instance_stats.json that seems to have a mapping from an instance id to a clip score (its not exactly that but its close, I have a mean and max clip score, which presumably were created over multiple outputs -- looks like 4 -- from the same text prompt as they are different).

However, I'm noticing that the mapping is many to 1 as some instance ids have multiple entries, some with no perturbation and others with a perturbation.

In order to run SRI's code, I need to effectively build a list of prompts and their corresponding clip scores.

The instances.json file has the original and perturbed input text mapped to instance-ids, but there doesn't seem to be a great way to join on perturbation details. To work around this I'm hashing the perurbation class to make a perturb id.

So now I have a list of text and corresponding clip scores. But as I go back to SRI's code, I see they do each score from the multiple outputs, which I don't think exists in the heim outputs. Maybe I can fudge it by building a normal distribution using the mean and max stats I do have, and then sampling 4 items from it, which is kinda iffy, but I'm not sure there is a better way forward.


---

Something else that's fun. I thought - because the heim outputs are 226GB, that the cache / generated images were going to be there (they are referenced in the scenario_state.json file), but it turns out all that size is just from .json files. Oh, but that's all due to display_predictions.json files, which have base64 encoded images.


Questions on SRI code:

The code currently seems to rely there being exactly 5 generated results per prompt, but if I'm reading it right, that's only the case because they are taking advantage of the structure to only compute an embedding per prompt and then replicate that embedding to make embedding-clipscore corresponds.
 My first questions are:

* How important is it that the training method has multiple y values per X embedding? If we only had 1 clip score per prompt, would any assumptions be violated?

* Similarly, what if the number of clip scores per prompt varies (i.e. some have up to 4, but most have only 1).

Additionally, I've found that we can get the CLIP scores for multiple prompts in the HEIM benchmarks, but it will require recomputing them. The outputs only store the mean and maximum CLIP score over all generated images per prompt. It looks like most scenarios will generate 4 image variants, and then associate the mean/max clip score with the prompt. If possible, I would like to avoid recomputing CLIP scores, I had 2 ideas on what might be a reasonable proxy:

1. Just use the mean or max clip score as the y value. For the max case, this is actually fine because it must have been an original real data point. The mean isn't guaranteed to be a real clip score, but I could use it to get 2 y values per X embedding.
2. If we really need more than 2 y values per X embedding, maybe we could assume a normal distribution using the max to estimate the variance, and draw dummy samples to get 5 values. We can use the actual mean / max and then fill in with 3 of those sampled values to get slightly more representative training data, but making that normal assumption seems like it probably wont hold and might weaken the model.


----


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
from magnet.utils.util_pandas import DotDictDataFrame
from functools import cache


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
    big_table.suffix_subframe(['instance_id', 'mean', 'request_states.request.prompt', 'run_spec.model', 'run_spec.name']).shorten_columns()
    big_table['input_text_id'] = big_table['request_states.instance.input.text'].apply(cached_hash)

    big_table[[
        'request_states.request.prompt',
        'request_states.instance.input.text'
    ]]

    big_table.search_columns('perturb')
    big_table.search_columns('modifications')
    big_table[['run_spec.name'] + big_table.search_columns('modifications')]

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
    import numpy as np
    unique_indices = np.array(unique_indices)
    unique_big_table = big_table.loc[unique_indices]

    for _, group in unique_big_table.groupby(['run_spec.model']):
        print(len(group))
        if 0:
            print(group.prefix_subframe('per_instance_stats.stat.max_clip_score').shorten_columns().T)

        group.prefix_subframe('per_instance_stats.stat.max_clip_score')

        group['per_instance_stats.stat.max_clip_score.max']
        prompts = group['request_states.instance.input.text']
        scores = group['per_instance_stats.stat.expected_clip_score.max']


def load_run_clip_info(run):
    """
    Relevant HELM data structures we are working with:

        helm.benchmark.metrics.metric.PerInstanceStats
        helm.benchmark.scenarios.scenario.Instance
        helm.benchmark.adaptation.request_state.RequestState
        helm.benchmark.augmentations.perturbation_description.PerturbationDescription

    """
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
            # if perturbation is not None:
            #     raise Exception
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

    import pandas as pd
    table = pd.concat([
        filtered_stats_df.set_index('per_instance_stats.perturb_instance_id', drop=False),
        filtered_states_df.set_index('request_states.instance.perturb_instance_id', drop=False),
    ], axis=1).reset_index(drop=True)
    table['run_path'] = run.path
    return table


def debug():
    heim_output_dpath = ub.Path('/data/crfm-helm-public/heim/benchmark_output')
    outs = magnet.HelmOutputs(heim_output_dpath)
    suite = outs.suites('v1.0.0')[0]
    runs = suite.runs('mscoco:*').existing()

    stats_table = suite.runs('mscoco:*').stats()

    stats_of_interest = [
        'expected_clip_score',
        'max_clip_score',
        # 'expected_clip_score_multilingual',
        # 'max_clip_score_multilingual'
    ]
    flags = stats_table['stats.name.name'].apply(lambda x: x in stats_of_interest)
    subtable = stats_table[flags]
    run_specs_of_interest = subtable['run_spec.name'].unique()

    from magnet.helm_outputs import HelmSuiteRuns
    runs = HelmSuiteRuns([p for p in runs.paths if p.name in run_specs_of_interest])

    import kwutil
    import pandas as pd
    from magnet.utils.util_pandas import DotDictDataFrame
    from magnet.utils.util_msgspec import asdict as struct_asdict

    for run in runs:
        instance_rows = []
        # import timerit
        # ti = timerit.Timerit(100, bestof=10, verbose=2)
        # for timer in ti.reset('time'):
        #     with timer:
        scenario_state = run.msgspec.scenario_state()
        scenario_state = run.dataclass.scenario_state()

        instances = scenario_state.instances
        for instance in instances:
            print(f'instance.perturbation={instance.perturbation}')
            perturb_id = ub.hash_data(instance.perturbation)
            row = kwutil.DotDict.from_nested(struct_asdict(instance))
            row['perturb_id'] = perturb_id
            row['perturb_instance_id'] = ub.hash_data([instance.id, perturb_id])[0:24]
            instance_rows.append(row)
        instance_table = DotDictDataFrame(instance_rows)
        instance_table['input.text']

        clip_accum = []
        for instance_stat in run.dataclass.per_instance_stats():
            perturb_id = ub.hash_data(instance_stat.perturbation)

            mean_clip = None
            max_clip = None
            for stat in instance_stat.stats:
                if 'expected_clip_score' == stat.name.name:
                    mean_clip = stat.mean
                if 'max_clip_score' == stat.name.name:
                    max_clip = stat.mean
            assert (mean_clip is not None) == (max_clip is not None)
            if mean_clip is not None:
                new = ub.udict(instance_stat.__dict__) - {'stats'}
                new['mean_clip'] = mean_clip
                new['max_clip'] = max_clip
                new['perturb_id'] = perturb_id
                new['perturb_instance_id'] = ub.hash_data([instance_stat.instance_id, perturb_id])[0:24]
                clip_accum.append(new)

        clip_table = pd.DataFrame(clip_accum)

        combo = pd.concat([
            instance_table.set_index('perturb_instance_id'),
            clip_table.set_index('perturb_instance_id')
        ], axis=1)

        run.dataclass.run_spec().adapter_spec.model
        print(combo)

        texts = combo['input.text'].values
        clip_scores = combo['max_clip'].values

#     data = []
#     for row in combo.iterrows():
#         row['input.text']
#         row['max_clip']

#     combo['mean_clip']


#     clip_table.join(instance_table, on='perturb_instance_id')

#     self = run


#     run = runs[0]
#     stats_table = run.stats_dataframe()

#     run.scenario_state_dataframe()
#     run.run_spec_dataframe()

#     suite = outs.suites[-1]

#     run_specs = outs.list_run_specs(suite)
#     run_specs = run_specs[0:1]

#     stats = load_all_stats_as_dataframe(suite, run_specs, root_dir=root_dir)


if __name__ == '__main__':
    """
    CommandLine:
        kernprof -lzvv -p magnet ~/code/magnet-sys-exploratory/dev/poc/poc_read_heim_v2.py
    """
    main()
