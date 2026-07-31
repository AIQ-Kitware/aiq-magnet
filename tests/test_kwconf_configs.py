from __future__ import annotations

import kwconf
import pytest

from magnet.backends.helm.cli.download_helm_results import DownloadHelmConfig
from magnet.backends.helm.cli.inspect_helm_models import InspectHelmModelsConfig
from magnet.backends.helm.cli.materialize_helm_run import (
    MaterializeHelmRunConfig,
    _coerce_path_roots,
)
from magnet.backends.helm.cli.materialize_helm_run_from_spec import (
    MaterializeHelmRunFromSpecConfig,
)
from magnet.cli.main import MagnetCLI
from magnet.demo.helm_demodata import HelmDemoConfig
from magnet.evaluation import EvaluationConfig
from magnet.examples.llama_consistency.claim import ConsistencyClaimCLI
from magnet.examples.llama_consistency.llama_predict import (
    ExampleLlamaEndpointCLI,
)


CONFIG_CLASSES: list[type[kwconf.Config]] = [
    DownloadHelmConfig,
    InspectHelmModelsConfig,
    MaterializeHelmRunConfig,
    MaterializeHelmRunFromSpecConfig,
    HelmDemoConfig,
    EvaluationConfig,
    ConsistencyClaimCLI,
    ExampleLlamaEndpointCLI,
]


@pytest.mark.parametrize('config_cls', CONFIG_CLASSES)
def test_kwconf_schema_is_valid(
    config_cls: type[kwconf.Config],
) -> None:
    config_cls.validate()


def test_comma_bearing_scalar_values_remain_strings() -> None:
    query = "model_name in ['openai/a', 'openai/b']"
    inspect_cfg = InspectHelmModelsConfig.cli(
        argv=['--query', query, '--max-rows', '3', '--verbose']
    )
    assert inspect_cfg.query == query
    assert inspect_cfg.max_rows == 3
    assert inspect_cfg.verbose is True

    benchmark = 'regex:^foo{1,3}$'
    download_cfg = DownloadHelmConfig.cli(argv=['out', benchmark, 'v1'])
    assert download_cfg.benchmark == benchmark

    run_entry = 'mmlu:subject=philosophy,model=openai/gpt2'
    materialize_cfg = MaterializeHelmRunConfig.cli(
        argv=['--run-entry', run_entry]
    )
    assert materialize_cfg.run_entry == run_entry


def test_yaml_list_fields_parse_as_lists() -> None:
    download_cfg = DownloadHelmConfig.cli(
        argv=['out', 'lite', 'v1', '--runs', '[foo, bar]']
    )
    assert download_cfg.runs == ['foo', 'bar']

    materialize_cfg = MaterializeHelmRunConfig.cli(
        argv=['--precomputed-root', '[/a, /b]']
    )
    assert materialize_cfg.precomputed_root == ['/a', '/b']

    from_spec_cfg = MaterializeHelmRunFromSpecConfig.cli(
        argv=['--enable-huggingface-models', '[repo-a, repo-b]']
    )
    assert from_spec_cfg.enable_huggingface_models == [
        'repo-a',
        'repo-b',
    ]

    scalar_model_cfg = MaterializeHelmRunConfig.cli(
        argv=['--enable-huggingface-models', 'repo-a']
    )
    assert scalar_model_cfg.enable_huggingface_models == ['repo-a']

    demo_cfg = HelmDemoConfig.cli(argv=['--run-entries', '[a, b]'])
    assert demo_cfg.run_entries == ['a', 'b']


def test_path_root_normalization_does_not_split_strings() -> None:
    assert _coerce_path_roots('/data/results') == ['/data/results']
    assert _coerce_path_roots(['/a', '/b']) == ['/a', '/b']


def test_nargs_and_boolean_value_compatibility() -> None:
    inspect_cfg = InspectHelmModelsConfig.cli(
        argv=['--columns', 'deployment', 'model_name', '--sort', 'model_name']
    )
    assert inspect_cfg.columns == ['deployment', 'model_name']
    assert inspect_cfg.sort == ['model_name']

    materialize_cfg = MaterializeHelmRunConfig.cli(
        argv=[
            '--precomputed_root',
            '[/a, /b]',
            '--require_per_instance_stats',
            'false',
        ]
    )
    assert materialize_cfg.precomputed_root == ['/a', '/b']
    assert materialize_cfg.require_per_instance_stats is False

    endpoint_cfg = ExampleLlamaEndpointCLI.cli(
        argv=[
            '--base-model',
            'base',
            '--comp-model',
            'comparison',
            '--threshold',
            '0.25',
        ]
    )
    assert endpoint_cfg.threshold == 0.25


def test_evaluation_validate_cli_alias() -> None:
    cfg = EvaluationConfig.cli(
        argv=['card.yaml', '--validate', 'warning']
    )
    assert cfg.path == 'card.yaml'
    assert cfg.validation == 'warning'

    cfg = EvaluationConfig.cli(
        argv=False,
        data={'path': 'card.yaml', 'validate': 'off'},
    )
    assert cfg.validation == 'off'


@pytest.mark.parametrize(
    'config_cls',
    [ConsistencyClaimCLI, ExampleLlamaEndpointCLI],
)
def test_required_example_fields_are_enforced(
    config_cls: type[kwconf.Config],
) -> None:
    with pytest.raises(ValueError, match='Required'):
        config_cls.cli(argv=[])


def test_modal_dispatch_forwards_only_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import magnet.backends.helm.cli.download_helm_results as download_mod
    import magnet.evaluation as evaluation_mod

    calls: list[tuple[str, bool | None, dict[str, object]]] = []

    def fake_evaluate(
        argv: bool | None = None, **kwargs: object
    ) -> None:
        calls.append(('evaluate', argv, kwargs))

    def fake_download(
        argv: bool | None = None, **kwargs: object
    ) -> int:
        calls.append(('download', argv, kwargs))
        return 7

    monkeypatch.setattr(evaluation_mod, 'main', fake_evaluate)
    monkeypatch.setattr(download_mod, 'main', fake_download)

    assert (
        MagnetCLI.main(
            argv=['evaluate', 'card.yaml', '--validate', 'warning']
        )
        == 0
    )
    assert (
        MagnetCLI.main(
            argv=[
                'download',
                'helm',
                'out',
                'lite',
                'v1',
                '--checksum',
            ]
        )
        == 7
    )
    assert calls == [
        (
            'evaluate',
            False,
            {'path': 'card.yaml', 'validation': 'warning'},
        ),
        (
            'download',
            False,
            {
                'download_dir': 'out',
                'benchmark': 'lite',
                'version': 'v1',
                'checksum': True,
            },
        ),
    ]
