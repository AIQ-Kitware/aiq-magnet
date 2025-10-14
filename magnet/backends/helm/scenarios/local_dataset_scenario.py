import json
from typing import List, Dict, Any

import dacite
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
)

class LocalDatasetScenario(Scenario):
    name = "local_dataset"
    description = "Generic scenario used to point at existing on-disk HELM Scenario Instances"
    tags = []

    def __init__(self, instances_path):
        super().__init__()
        self.instances_path = instances_path

    def get_instances(self, output_path: str) -> List[Instance]:
        # data_path: str = os.path.join(output_path, "data")
        # ensure_directory_exists(data_path)
        # ^ Don't need the above for now; maybe make a symlink to `self.instances_path`

        with open(self.instances_path) as f:
            json_instances: List[Dict[str, Any]] = json.load(f)

        instances = [dacite.from_dict(Instance, instance) for instance in json_instances]

        return instances
