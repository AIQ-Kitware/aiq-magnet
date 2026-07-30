from pathlib import Path

import pytest

from magnet.backends.helm.cli.materialize_helm_run import prepare_local_helm_config


def test_copies_all_three_sidecars(tmp_path: Path) -> None:
    out_dpath = tmp_path / "out"
    out_dpath.mkdir()
    deployments = tmp_path / "my_deployments.yaml"
    deployments.write_text("model_deployments: []\n")
    metadata = tmp_path / "my_metadata.yaml"
    metadata.write_text("models: []\n")
    tokenizers = tmp_path / "my_tokenizers.yaml"
    tokenizers.write_text("tokenizer_configs: []\n")

    local_path = prepare_local_helm_config(
        out_dpath=out_dpath,
        local_path="prod_env",
        model_deployments_fpath=deployments,
        model_metadata_fpath=metadata,
        tokenizer_configs_fpath=tokenizers,
    )

    # Each source lands under its CANONICAL name (what HELM's
    # register_configs_from_directory looks for), not the source basename.
    assert local_path == out_dpath / "prod_env"
    assert (local_path / "model_deployments.yaml").read_text() == deployments.read_text()
    assert (local_path / "model_metadata.yaml").read_text() == metadata.read_text()
    assert (local_path / "tokenizer_configs.yaml").read_text() == tokenizers.read_text()


def test_sidecars_are_optional(tmp_path: Path) -> None:
    out_dpath = tmp_path / "out"
    out_dpath.mkdir()
    local_path = prepare_local_helm_config(out_dpath=out_dpath, local_path="prod_env")
    assert local_path.is_dir()
    assert not (local_path / "model_metadata.yaml").exists()
    assert not (local_path / "tokenizer_configs.yaml").exists()


def test_missing_sidecar_fails_with_param_name(tmp_path: Path) -> None:
    out_dpath = tmp_path / "out"
    out_dpath.mkdir()
    with pytest.raises(FileNotFoundError, match="model_metadata_fpath"):
        prepare_local_helm_config(
            out_dpath=out_dpath,
            local_path="prod_env",
            model_metadata_fpath=tmp_path / "does_not_exist.yaml",
        )
