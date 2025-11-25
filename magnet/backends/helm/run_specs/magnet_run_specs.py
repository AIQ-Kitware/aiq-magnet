from dataclasses import replace

from helm.benchmark.run_spec import RunSpec, run_spec_function, get_run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

@run_spec_function("local_dataset")
def get_local_dataset_meta_spec(original_spec: str, instances_path: str, **kwargs) -> RunSpec:
    run_spec_function = get_run_spec_function(original_spec)

    original_run_spec = run_spec_function(**kwargs)

    local_dataset_scenario_spec = ScenarioSpec(
        class_name="magnet.backends.helm.scenarios.LocalDatasetScenario", args={"instances_path": instances_path}
    )

    run_spec = replace(original_run_spec, scenario_spec=local_dataset_scenario_spec)

    return run_spec
