"""Tests for the faithful-replay CLI ``materialize_helm_run_from_spec``.

These need ``helm`` importable (the module under test imports it), so the whole
file is guarded by ``importorskip``. They cover the two version-drift surfaces
from ``docs/planning/run-from-run-spec-json-plan.md`` section 1:

* **field/shape drift is silent** — exercised by the round-trip key-preservation
  test (a raw JSON key absent after ``from_json`` -> ``to_json`` means cattrs
  silently dropped it);
* **class availability is loud** — exercised by the preflight tests.

The heavy end-to-end replay (a real model executes) is opt-in behind
``MATERIALIZE_FROM_SPEC_INTEGRATION`` so the default suite stays CPU/offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("helm")

from helm.common.object_spec import ObjectSpec  # noqa: E402

from magnet.backends.helm.cli.materialize_helm_run_from_spec import (  # noqa: E402
    MaterializeHelmRunFromSpecConfig,
    collect_class_names,
    preflight_resolve_classes,
)

PUBLIC_ROOT = Path("/data/crfm-helm-public")


def _spec_stub(scenario_spec, metric_specs=None, annotators=None):
    """A minimal stand-in exposing only the attributes the preflight reads.

    ``collect_class_names`` / ``preflight_resolve_classes`` touch only
    ``scenario_spec`` / ``metric_specs`` / ``annotators`` (never ``adapter_spec``,
    whose model fields are deployment names, not class paths), so we avoid having
    to construct a full ``AdapterSpec``.
    """
    return SimpleNamespace(
        scenario_spec=scenario_spec,
        metric_specs=metric_specs or [],
        annotators=annotators,
    )


# --------------------------------------------------------------------------
# Preflight class resolution (loud version-drift surface)
# --------------------------------------------------------------------------


def test_collect_class_names_recurses_into_nested_object_specs():
    # An annotator whose args carry a nested judge model spec (as a plain dict,
    # which is how cattrs leaves Dict[str, Any] values after from_json).
    annotator = ObjectSpec(
        class_name="pkg.mod.MyAnnotator",
        args={"judge": {"class_name": "pkg.mod.JudgeModel", "args": {"temperature": 0}}},
    )
    spec = _spec_stub(
        scenario_spec=ObjectSpec(class_name="pkg.mod.MyScenario", args={}),
        metric_specs=[ObjectSpec(class_name="pkg.mod.MyMetric", args={})],
        annotators=[annotator],
    )
    names = collect_class_names(spec)
    assert "pkg.mod.MyScenario" in names
    assert "pkg.mod.MyMetric" in names
    assert "pkg.mod.MyAnnotator" in names
    # The nested (dict-shaped) judge spec must be reached by the recursion.
    assert "pkg.mod.JudgeModel" in names


def test_preflight_passes_for_resolvable_classes():
    # ``helm.common.object_spec.ObjectSpec`` is guaranteed importable.
    spec = _spec_stub(
        scenario_spec=ObjectSpec(class_name="helm.common.object_spec.ObjectSpec", args={}),
        metric_specs=[ObjectSpec(class_name="helm.common.object_spec.ObjectSpec", args={})],
    )
    preflight_resolve_classes(spec)  # should not raise


def test_preflight_fails_loudly_on_missing_class():
    spec = _spec_stub(
        scenario_spec=ObjectSpec(class_name="helm.benchmark.scenarios.no_such_module.Missing", args={}),
    )
    with pytest.raises(SystemExit) as excinfo:
        preflight_resolve_classes(spec)
    assert "no_such_module" in str(excinfo.value)


def test_preflight_reports_non_import_errors_without_crashing(monkeypatch):
    # Importing a scenario module can raise things other than ImportError /
    # AttributeError — notably helm's ``OptionalDependencyNotInstalled`` for
    # vision-language extras. The preflight must record those as unresolved
    # classes (an environment/recipe filter reason), never propagate the raw
    # exception and crash mid-scan.
    import helm.common.object_spec as ospec

    def boom(_name):
        raise RuntimeError("Optional dependency 'latex' is not installed")

    monkeypatch.setattr(ospec, "get_class_by_name", boom)
    spec = _spec_stub(scenario_spec=ObjectSpec(class_name="pkg.mod.Thing", args={}))
    with pytest.raises(SystemExit) as excinfo:
        preflight_resolve_classes(spec)
    assert "pkg.mod.Thing" in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)


def test_preflight_detects_missing_nested_judge_class():
    # A resolvable annotator wrapping an unresolvable nested judge spec — the
    # recursion must surface the nested failure, not just the top-level class.
    annotator = ObjectSpec(
        class_name="helm.common.object_spec.ObjectSpec",
        args={"judge": {"class_name": "helm.nope.Judge", "args": {}}},
    )
    spec = _spec_stub(
        scenario_spec=ObjectSpec(class_name="helm.common.object_spec.ObjectSpec", args={}),
        annotators=[annotator],
    )
    with pytest.raises(SystemExit) as excinfo:
        preflight_resolve_classes(spec)
    assert "helm.nope.Judge" in str(excinfo.value)


# --------------------------------------------------------------------------
# Schema-drift round-trip guard (silent version-drift surface)
# --------------------------------------------------------------------------


def _missing_key_paths(raw, roundtripped, prefix=""):
    """Return key paths present in ``raw`` but absent from ``roundtripped``.

    Because HELM writes ``run_spec.json`` with ``asdict_without_nones`` (drops
    only None-valued fields), every key in ``raw`` is non-None, and the codec
    re-emits every non-None field — so a path reported here means cattrs silently
    dropped an unknown key on ``from_json`` (the field/shape drift we guard).
    """
    missing: list[str] = []
    if isinstance(raw, dict):
        if not isinstance(roundtripped, dict):
            return [prefix or "<root>"]
        for key, value in raw.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in roundtripped:
                missing.append(path)
            else:
                missing.extend(_missing_key_paths(value, roundtripped[key], path))
    elif isinstance(raw, list):
        if not isinstance(roundtripped, list):
            return [prefix or "<root>"]
        for idx, value in enumerate(raw):
            path = f"{prefix}[{idx}]"
            if idx >= len(roundtripped):
                missing.append(path)
            else:
                missing.extend(_missing_key_paths(value, roundtripped[idx], path))
    return missing


def test_missing_key_paths_helper_flags_drops_and_passes_identity():
    raw = {"a": 1, "b": {"c": 2, "old_field": 3}, "lst": [{"x": 1}, {"y": 2}]}
    roundtripped = {"a": 1, "b": {"c": 2}, "lst": [{"x": 1}]}
    missing = _missing_key_paths(raw, roundtripped)
    assert "b.old_field" in missing
    assert "lst[1]" in missing
    # A faithful round-trip drops nothing.
    assert _missing_key_paths(raw, raw) == []


def _sample_public_run_specs(limit: int = 8) -> list[Path]:
    if not PUBLIC_ROOT.is_dir():
        return []
    found: list[Path] = []
    for path in PUBLIC_ROOT.glob("*/benchmark_output/runs/*/*/run_spec.json"):
        found.append(path)
        if len(found) >= limit:
            break
    return found


def test_public_run_spec_json_roundtrips_without_key_loss():
    from helm.benchmark.run_spec import RunSpec
    from helm.common.codec import from_json, to_json

    specs = _sample_public_run_specs()
    if not specs:
        pytest.skip("no public run_spec.json corpus at /data/crfm-helm-public")

    checked = 0
    for path in specs:
        raw_text = path.read_text()
        raw = json.loads(raw_text)
        try:
            run_spec = from_json(raw_text, RunSpec)
        except Exception:
            # The pinned crfm-helm shape differs from this spec's era badly
            # enough that structuring raises (a since-added required field, etc.).
            # That is the *loud* drift surface, not what this test guards — skip.
            continue
        roundtripped = json.loads(to_json(run_spec))
        missing = _missing_key_paths(raw, roundtripped)
        assert not missing, f"{path} silently dropped keys on from_json: {missing}"
        checked += 1

    if checked == 0:
        pytest.skip("no sampled public spec deserialized under the pinned crfm-helm")


# --------------------------------------------------------------------------
# End-to-end replay (opt-in; a real model executes)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MATERIALIZE_FROM_SPEC_INTEGRATION"),
    reason="set MATERIALIZE_FROM_SPEC_INTEGRATION=1 (and provide a runnable model) to replay",
)
def test_integration_discovery_replay_keeps_official_run_name(tmp_path):
    """Replay a discovered official run_spec.json and assert identity is kept.

    Drive via env so it can target whatever official run + local model the host
    has: ``FROM_SPEC_RUN_ENTRY`` (e.g. an mmlu/openai_gpt2 entry),
    ``FROM_SPEC_PRECOMPUTED_ROOT`` (defaults to /data/crfm-helm-public), and an
    optional ``FROM_SPEC_MODEL_DEPLOYMENTS`` override yaml.
    """
    run_entry = os.environ["FROM_SPEC_RUN_ENTRY"]
    precomputed_root = os.environ.get("FROM_SPEC_PRECOMPUTED_ROOT", str(PUBLIC_ROOT))
    argv = [
        "--run-entry", run_entry,
        "--precomputed-root", precomputed_root,
        "--suite", "from-spec-integration",
        "--max-eval-instances", "2",
        "--out-dpath", str(tmp_path),
    ]
    override = os.environ.get("FROM_SPEC_MODEL_DEPLOYMENTS")
    if override:
        argv += ["--model-deployments-fpath", override]

    manifest = MaterializeHelmRunFromSpecConfig.main(argv)

    assert (tmp_path / "DONE").is_file()
    assert manifest["status"] == "replayed"
    # By-name substitution: produced dir keeps the official run_spec.name.
    assert manifest["replay"]["computed_run_name"] == manifest["recipe"]["run_spec_name"]
