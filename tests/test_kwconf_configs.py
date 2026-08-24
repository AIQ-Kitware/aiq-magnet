import importlib
import inspect
import pkgutil

import kwconf

import magnet
from magnet.backends.helm.cli.download_helm_results import DownloadHelmConfig
from magnet.backends.helm.cli.inspect_helm_models import InspectHelmModelsConfig
from magnet.evaluation import EvaluationConfig
from magnet.evaluation_new import NewEvaluationConfig


def _discover_config_classes():
    """
    Every :class:`kwconf.Config` magnet defines.

    Discovered rather than listed, because a config added later is exactly the
    one nobody remembers to add to a list -- and the checks below are only worth
    anything if they cover configs written after them. Configs merely imported
    from a dependency are skipped; we do not control their option names.
    """
    found = {}
    for module_info in pkgutil.walk_packages(magnet.__path__, 'magnet.'):
        module = importlib.import_module(module_info.name)
        for obj in vars(module).values():
            if not inspect.isclass(obj) or not issubclass(obj, kwconf.Config):
                continue
            if obj is kwconf.Config or not obj.__module__.startswith('magnet.'):
                continue
            found[f'{obj.__module__}.{obj.__name__}'] = obj
    return [found[key] for key in sorted(found)]


ALL_CONFIG_CLASSES = _discover_config_classes()


def test_configs_were_discovered():
    # A discovery bug would make the checks below vacuously pass.
    assert EvaluationConfig in ALL_CONFIG_CLASSES
    assert NewEvaluationConfig in ALL_CONFIG_CLASSES
    assert len(ALL_CONFIG_CLASSES) >= 5


def test_kwconf_schemas_are_valid():
    for config_cls in ALL_CONFIG_CLASSES:
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
    assert evaluation_cfg['validate'] == 'warning'

    evaluation_cfg = EvaluationConfig.cli(
        argv=False, data={'path': 'card.yaml', 'validate': 'off'}
    )
    assert evaluation_cfg['validate'] == 'off'


def test_validate_behavior_has_not_changed():
    """
    Kwconf.Config defines a ``.validate`` method, and we have a validate
    CLI key, which shadows it. For some reason claude thought that the method
    would win, but it seems the user value wins. This test just asserts
    that behavior so we get a red dashboard if it ever changes.
    """
    cfg = EvaluationConfig.cli(argv=['card.yaml', '--validate', 'error'])

    assert cfg['validate'] == 'error'
    assert not callable(cfg.validate), ('The user item shadows the .validate method')
    assert cfg.validate == 'error'


def test_no_config_option_shadows_a_config_method():
    """
    No option in any config may collide with a ``kwconf.Config`` attribute.

    ``validate`` is the one that got through, and the names still available to
    collide -- ``get``, ``keys``, ``update``, ``load``, ``dump``, ``copy`` --
    are ordinary enough that the next one is a matter of time. A new option
    named for one of them should fail here rather than in a run that quietly
    skipped a step.
    """
    reserved = {name for name in dir(kwconf.Config) if not name.startswith('_')}

    known = {
        # Renaming it would break `--validate` and every card that sets it, so
        # it is read as an item instead. See magnet/evaluation.py:main.
        (EvaluationConfig, 'validate'),
        (NewEvaluationConfig, 'validate'),
    }

    collisions = set()
    for config_cls in ALL_CONFIG_CLASSES:
        for key in config_cls().keys():
            if key in reserved and (config_cls, key) not in known:
                collisions.add(f'{config_cls.__name__}.{key}')

    assert not collisions, (
        f'config options shadow kwconf.Config attributes: {sorted(collisions)}. '
        'Rename the option, or read it with item access and add it to `known`.'
    )


def test_new_evaluator_has_only_kwdagger_execution_controls():
    keys = set(NewEvaluationConfig().keys())
    assert {
        'path', 'output_path', 'params', 'backend', 'tmux_workers',
        'skip_existing', 'cache', 'max_configs',
    } <= keys
    assert not {
        'override', 'jobs', 'parallel_backend', 'queue_backend', 'workers'
    } & keys


def test_new_evaluator_schedule_defaults_match_kwdagger():
    from kwdagger.schedule import ScheduleEvaluationConfig

    magnet_cfg = NewEvaluationConfig()
    kwdagger_cfg = ScheduleEvaluationConfig()
    for key in [
        'backend', 'tmux_workers', 'skip_existing', 'cache', 'max_configs'
    ]:
        assert magnet_cfg[key] == kwdagger_cfg[key]


def test_legacy_evaluator_surface_is_still_present():
    keys = set(EvaluationConfig().keys())
    assert {
        'path', 'output_path', 'override', 'jobs', 'parallel_backend',
    } <= keys
    assert not {'params', 'queue_backend'} & keys
