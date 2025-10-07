import argparse
import os
import json
from typing import List, Dict, Any

from helm.common.general import asdict_without_nones
from helm.benchmark.run_expander import PERTURBATION_SPECS_DICT
from helm.benchmark.augmentations.perturbation import (
    create_perturbation,
)
import dacite
from helm.benchmark.scenarios.scenario import (
    Instance,
    with_instance_ids,
)
import ubelt as ub


def main():
    parser = argparse.ArgumentParser(
        description="Run example random predictor")

    parser.add_argument('-i', '--instances_filepath',
                        type=str,
                        required=True,
                        help="Filepath to HELM Scenario Instances to perturb")
    parser.add_argument('-o', '--output_dir',
                        type=str,
                        required=True,
                        help='Output directory')
    parser.add_argument('-p', '--perturbation_specs',
                        type=str,
                        nargs='+',
                        required=True,
                        help='HELM perturbation specs (e.g. "misspelling_sweep")')

    apply_perturbations(**vars(parser.parse_args()))


def apply_perturbations(instances_filepath: str,
                        output_dir: str,
                        perturbation_specs: List[str]=[]) -> None:
    expanded_perturbation_specs = {}
    for pspec_str in perturbation_specs:
        if pspec_str not in PERTURBATION_SPECS_DICT:
            print(f"Warning: perturbation spec '{pspec_str}' not found, skipping")

        expanded_perturbation_specs.update(PERTURBATION_SPECS_DICT[pspec_str])

    with open(instances_filepath) as f:
        json_instances: List[Dict[str, Any]] = json.load(f)

    input_instances = [dacite.from_dict(Instance, instance) for instance in json_instances]

    # For some reason the original caching out of instances from
    # runner doesn't include instance IDs, so we need to add them if
    # they don't exist (perturbation requires instance IDs)
    if any(instance.id is None for instance in input_instances):
        input_instances = with_instance_ids(input_instances)

    for pname, pspecs in expanded_perturbation_specs.items():
        for pspec in pspecs:
            if len(pspecs) > 1:
                p_output_dir = os.path.join(output_dir, pname, ub.hash(pspec, hasher="md5"))
            else:
                p_output_dir = os.path.join(output_dir, pname)

            os.makedirs(p_output_dir, exist_ok=True)

            perturbation = create_perturbation(pspec)

            perturbed_instances = []
            for instance in input_instances:
                perturbed_instance: Instance = perturbation.apply(
                    instance, seed=None)
                perturbed_instances.append(perturbed_instance)

            with open(os.path.join(p_output_dir, 'instances.json'), 'w') as f:
                json.dump([asdict_without_nones(instance) for instance in perturbed_instances], f, indent=2)

            with open(os.path.join(p_output_dir, 'perturbation.object_spec.json'), 'w') as f:
                json.dump(pspec.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
