import importlib
from inspect import getmembers, isclass
import pkgutil
from typing import Dict, Optional

import helm.benchmark.scenarios as HELM_SCENARIOS_PKG
from helm.benchmark.scenarios.scenario import Scenario
from helm.benchmark.run_spec import discover_run_spec_functions, _REGISTERED_RUN_SPEC_FUNCTIONS, RunSpec

HELM_IMPORTED_SCENARIOS: Dict[str, type[Scenario]] = {}

class HELMScenarios:
    """
    Simple collection of HELM scenarios for aggregate exploration


    Collects available Scenario and RunSpec definitions in HELM package (helm/benchmark/scenarios).

    Example:
        >>> from magnet.backends.helm.scenarios import HELMScenarios
        >>> scenarios = HELMScenarios()
        >>> len(scenarios) # subject to change; assumes crfm-helm[all] dependencies
        345
    """
    def __init__(self):
        if not _REGISTERED_RUN_SPEC_FUNCTIONS:
            discover_run_spec_functions()

        self.run_specs = _REGISTERED_RUN_SPEC_FUNCTIONS

        if not HELM_IMPORTED_SCENARIOS:
            discover_scenarios()

        self.scenarios = HELM_IMPORTED_SCENARIOS

    def get_run_specs(self, scenario_name: str) -> Optional[RunSpec]:
        """
        Return the run spec functions for provided scenario

        WIP: If RunSpecFunction has default args, then func().scenario_spec.class_name is path to Scenario
        Only 166/302 values of _REGISTERED_RUN_SPEC_FUNCTIONS have no required args
        """
        pass

    def __len__(self):
        """
        Retrieve number of implemented scenarios
        """
        return len(self.scenarios)

# Adapted from discover_run_spec_functions() and helpers in helm/benchmark/run_spec.py
def discover_scenarios() -> None:
    """
    Discover all scenarios under helm.benchmark.scenarios and store the classes by name
    """
    for finder, name, ispkg in pkgutil.walk_packages(HELM_SCENARIOS_PKG.__path__, HELM_SCENARIOS_PKG.__name__ + "."):
        try:
            module = importlib.import_module(name)
            for cls_name, cls in getmembers(module, isclass):
                # remove "test_" per docs/scenarios.md
                if issubclass(cls, Scenario) and cls is not Scenario and "test_" not in cls_name:
                    HELM_IMPORTED_SCENARIOS[cls_name] = cls
        except ModuleNotFoundError as e:
            print(f"Failed to import {name}; Missing {e}")