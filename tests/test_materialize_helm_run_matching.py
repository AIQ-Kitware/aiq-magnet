import pytest
pytest.importorskip('helm', reason="needs the helm extra: pip install 'aiq-magnet[helm]'")  # noqa: E402
from magnet.backends.helm.cli.materialize_helm_run import run_dir_matches_requested


def test_run_dir_matches_requested_normalizes_model_deployment() -> None:
    requested = (
        "ifeval:model=qwen/qwen2.5-7b-instruct-turbo,"
        "model_deployment=kubeai/qwen2-5-7b-instruct-turbo-default-local"
    )
    produced = (
        "ifeval:model=qwen_qwen2.5-7b-instruct-turbo,"
        "model_deployment=kubeai_qwen2-5-7b-instruct-turbo-default-local"
    )
    assert run_dir_matches_requested(produced, requested)
