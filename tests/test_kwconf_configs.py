import kwconf

from magnet.backends.helm.cli.download_helm_results import DownloadHelmConfig
from magnet.backends.helm.cli.inspect_helm_models import InspectHelmModelsConfig
from magnet.backends.helm.cli.materialize_helm_run import MaterializeHelmRunConfig
from magnet.backends.helm.cli.materialize_helm_run_from_spec import (
    MaterializeHelmRunFromSpecConfig,
)
from magnet.evaluation import EvaluationConfig


def test_kwconf_schemas_are_valid():
    config_classes = [
        DownloadHelmConfig,
        InspectHelmModelsConfig,
        MaterializeHelmRunConfig,
        MaterializeHelmRunFromSpecConfig,
        EvaluationConfig,
    ]
    for config_cls in config_classes:
        assert issubclass(config_cls, kwconf.Config)
        config_cls.validate()


def test_comma_bearing_scalar_values_remain_strings():
    query = "model_name in ['openai/a', 'openai/b']"
    inspect_cfg = InspectHelmModelsConfig.cli(argv=['--query', query])
    assert inspect_cfg.query == query

    benchmark = 'regex:^foo{1,3}$'
    download_cfg = DownloadHelmConfig.cli(argv=['out', benchmark, 'v1'])
    assert download_cfg.benchmark == benchmark


def test_validate_alias():
    evaluation_cfg = EvaluationConfig.cli(
        argv=['card.yaml', '--validate', 'warning']
    )
    assert evaluation_cfg.validation == 'warning'

    evaluation_cfg = EvaluationConfig.cli(
        argv=False, data={'path': 'card.yaml', 'validate': 'off'}
    )
    assert evaluation_cfg.validation == 'off'
