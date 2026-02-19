#!/usr/bin/env python3
"""
inspect_helm_models.py

Inspect HELM model deployments using pandas + scriptconfig.

Pulls deployments from:
  - helm.benchmark.config_registry.register_builtin_configs_from_helm_package()
  - helm.benchmark.model_deployment_registry.ALL_MODEL_DEPLOYMENTS

Shows:
  - deployment name
  - model_name
  - tokenizer_name
  - max_sequence_length / max_request_length / max_sequence_and_generated_tokens_length
  - deprecated
  - client_spec.class_name
  - client_spec.args (optionally flattened)

Examples:
  python inspect_helm_models.py
  python inspect_helm_models.py --where "deployment.str.startswith('openai/')"
  python inspect_helm_models.py --columns deployment model_name client_class max_sequence_length
  python inspect_helm_models.py --sort model_name
  python inspect_helm_models.py --groupby model_name
  python inspect_helm_models.py --format json
  python inspect_helm_models.py --flatten-client-args 1 --client-args-prefix cs_
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
import scriptconfig as scfg


def _build_deployments_df(
    *,
    include_client_args: bool = True,
    flatten_client_args: bool = False,
    client_args_prefix: str = "client_",
) -> pd.DataFrame:
    # Import HELM registries exactly as requested
    from helm.benchmark import config_registry
    from helm.benchmark import model_deployment_registry

    config_registry.register_builtin_configs_from_helm_package()

    rows = []
    for dep in model_deployment_registry.ALL_MODEL_DEPLOYMENTS:
        cs = getattr(dep, "client_spec", None)

        row: Dict[str, Any] = {
            "deployment": getattr(dep, "name", None),
            "model_name": getattr(dep, "model_name", None),
            "tokenizer_name": getattr(dep, "tokenizer_name", None),
            "max_sequence_length": getattr(dep, "max_sequence_length", None),
            "max_request_length": getattr(dep, "max_request_length", None),
            "max_sequence_and_generated_tokens_length": getattr(
                dep, "max_sequence_and_generated_tokens_length", None
            ),
            "deprecated": getattr(dep, "deprecated", None),
            "client_class": getattr(cs, "class_name", None),
        }

        # Include client args (often contains endpoints / model identifiers / etc.)
        cs_args = getattr(cs, "args", None)
        if include_client_args:
            row["client_args"] = cs_args

        if flatten_client_args and isinstance(cs_args, dict):
            for k, v in cs_args.items():
                row[f"{client_args_prefix}{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    # Helpful default ordering
    if "deployment" in df.columns:
        df = df.sort_values(["deployment"], kind="stable", na_position="last").reset_index(drop=True)
    return df


class InspectHelmModelsConfig(scfg.DataConfig):
    """
    Pandas-based inspection of HELM model deployments.
    """

    # Output / formatting
    format = scfg.Value(
        "table",
        help="Output format",
        choices=["table", "csv", "json", "jsonl", "md"],
    )
    max_rows = scfg.Value(None, help="Max rows to print (None = no limit)")
    columns = scfg.Value(
        None,
        help="Subset of columns to show (space-separated)",
        nargs="*",
    )

    # Selection / filtering / shaping
    where = scfg.Value(
        None,
        help=(
            "Pandas query or python expression evaluated against df. "
            "Example: \"deployment.str.startswith('openai/') and ~deprecated\""
        ),
    )
    query = scfg.Value(
        None,
        help="Pandas DataFrame.query string (uses column names). Example: \"deprecated == False\"",
    )
    sort = scfg.Value(
        None,
        help="Column(s) to sort by",
        nargs="*",
    )
    groupby = scfg.Value(
        None,
        help="If set, group by this column and show deployment counts per group",
    )

    # Client spec options
    include_client_args = scfg.Value(True, help="Include client_spec.args as a dict column")
    flatten_client_args = scfg.Value(False, help="Flatten client_spec.args into individual columns")
    client_args_prefix = scfg.Value("client_", help="Prefix for flattened client args columns")


def _apply_where_expr(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """
    Evaluate a python expression with 'df' in scope and columns accessible as df['col'].
    Also exposes common helpers via locals.
    """
    # Users often want .str filters; they can use df['deployment'].str...
    # But we also allow "deployment" as a variable name (Series) for convenience.
    localns = {"df": df}
    for col in df.columns:
        # Only expose safe identifiers
        if isinstance(col, str) and col.isidentifier():
            localns[col] = df[col]
    try:
        mask = eval(expr, {"__builtins__": {}}, localns)
    except Exception as ex:
        raise ValueError(f"Failed to eval --where expression: {expr!r}. Error: {ex}") from ex
    if isinstance(mask, pd.Series):
        return df[mask].copy()
    if isinstance(mask, pd.DataFrame):
        return mask.copy()
    raise ValueError(f"--where must evaluate to a boolean Series or a DataFrame, got: {type(mask)}")


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # fallback without tabulate
        return df.to_string(index=False)


def main(argv=None) -> int:
    cfg = InspectHelmModelsConfig.cli(argv=argv)

    df = _build_deployments_df(
        include_client_args=cfg["include_client_args"],
        flatten_client_args=cfg["flatten_client_args"],
        client_args_prefix=cfg["client_args_prefix"],
    )

    # Filtering: query first (safe-ish), then where (python expr)
    if cfg["query"]:
        df = df.query(cfg["query"]).copy()

    if cfg["where"]:
        df = _apply_where_expr(df, cfg["where"])

    # Grouping summary
    if cfg["groupby"]:
        gcol = cfg["groupby"]
        if gcol not in df.columns:
            raise SystemExit(f"--groupby column {gcol!r} not found. Available: {list(df.columns)}")
        out = (
            df.groupby(gcol, dropna=False)
            .agg(
                num_deployments=("deployment", "count"),
                any_deprecated=("deprecated", lambda s: bool(pd.Series(s).fillna(False).any())),
                client_classes=("client_class", lambda s: ",".join(sorted(set([x for x in s.dropna().astype(str)])))),
            )
            .reset_index()
        )
        df = out

    # Sort + columns
    if cfg["sort"]:
        for col in cfg["sort"]:
            if col not in df.columns:
                raise SystemExit(f"--sort column {col!r} not found. Available: {list(df.columns)}")
        df = df.sort_values(list(cfg["sort"]), kind="stable", na_position="last")

    if cfg["columns"]:
        for col in cfg["columns"]:
            if col not in df.columns:
                raise SystemExit(f"--columns value {col!r} not found. Available: {list(df.columns)}")
        df = df[list(cfg["columns"])].copy()

    # Limit printing
    if cfg["max_rows"] is not None:
        df = df.head(int(cfg["max_rows"]))

    fmt = cfg["format"]
    if fmt == "table":
        # Best-effort width management for terminals
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))
    elif fmt == "md":
        print(_to_markdown(df))
    elif fmt == "csv":
        print(df.to_csv(index=False))
    elif fmt == "json":
        print(json.dumps(df.to_dict(orient="records"), indent=2, default=str))
    elif fmt == "jsonl":
        for rec in df.to_dict(orient="records"):
            print(json.dumps(rec, default=str))
    else:
        raise SystemExit(f"Unknown format: {fmt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
