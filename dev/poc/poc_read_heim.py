"""
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
"""

import ubelt as ub
import magnet
from magnet.loaders import load_all_stats_as_dataframe

heim_output_dpath = ub.Path('/data/crfm-helm-public/heim')
dpath = heim_output_dpath

root_dir = (dpath / 'benchmark_output')
suite = 'v1.1.0'

import xdev
dirwalker = xdev.DirectoryWalker(dpath).build()
dirwalker.write_report(max_depth=4)

# run_specs = ['mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2']
# stats = load_all_stats_as_dataframe(suite, run_specs, root_dir)

heim_output_dpath = ub.Path('/data/crfm-helm-public/heim')
outs = magnet.HelmOutputs(root_dir)
suite = outs.suites('v1.0.0')[0]
runs = suite.runs('mscoco:*').existing()

scenario_table = runs.scenario_state_dataframe()
spec_table = runs.run_spec_dataframe()
stats_table = runs.stats_dataframe()

stats_of_interest = [
    'expected_clip_score',
    'max_clip_score',
    # 'expected_clip_score_multilingual',
    # 'max_clip_score_multilingual'
]

flags = stats_table['stat.name.name'].apply(lambda x: x in stats_of_interest)
subtable = stats_table[flags]

run_specs_of_interest = subtable['run_spec.name'].unique()

from magnet.helm_outputs import HelmSuiteRuns
runs = HelmSuiteRuns([p for p in runs.paths if p.name in run_specs_of_interest])

run = runs[0]


import dataclasses
import kwutil
from magnet.utils.util_pandas import DotDictDataFrame
import pandas as pd
instance_rows = []
instances = run.scenario_state().instances
for instance in run.scenario_state().instances:
    perturb_id = ub.hash_data(None if instance.perturbation is None else dataclasses.asdict(instance.perturbation))
    row = kwutil.DotDict.from_nested(dataclasses.asdict(instance))
    row['perturb_id'] = perturb_id
    row['perturb_instance_id'] = ub.hash_data([instance.id, perturb_id])[0:24]
    instance_rows.append(row)
instance_table = DotDictDataFrame(instance_rows)
instance_table['input.text']


clip_accum = []
for instance_stat in run.per_instance_stats():
    perturb_id = ub.hash_data(None if instance_stat.perturbation is None else dataclasses.asdict(instance_stat.perturbation))

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

texts = combo['input.text'].values
clip_scores = combo['max_clip'].values

data = []
for row in combo.iterrows():
    row['input.text']
    row['max_clip']

combo['mean_clip']


clip_table.join(instance_table, on='perturb_instance_id')

self = run


run = runs[0]
stats_table = run.stats_dataframe()

run.scenario_state_dataframe()
run.run_spec_dataframe()

suite = outs.suites[-1]

run_specs = outs.list_run_specs(suite)
run_specs = run_specs[0:1]

stats = load_all_stats_as_dataframe(suite, run_specs, root_dir=root_dir)
