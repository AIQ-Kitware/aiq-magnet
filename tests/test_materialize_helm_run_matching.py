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
