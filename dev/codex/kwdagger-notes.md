# kwdagger Notes

Date started: 2026-03-24
Project: `/home/joncrall/code/aiq-magnet`

## Important Points

- `kwdagger schedule` is correctly building the HELM smoke-test DAG as one node per `run_entry`; for the current smoke manifest that means 6 jobs total.
- With `backend=tmux`, `devices=0,1`, and `tmux_workers=2`, the queue is split into two tmux sessions, one per visible GPU, with jobs serialized within each session.
- `kwdagger` may render node defaults into the generated CLI even when the manifest omits them.
  - Observed examples:
    - `--precomputed_root=None`
    - `--model_deployments_fpath=None`
    - bare `--enable_huggingface_models`
    - bare `--enable_local_huggingface_models`
- Because of that behavior, downstream CLIs should normalize null-like placeholders such as `None`, `null`, and empty strings instead of assuming omitted values stay omitted.
- For the HELM audit node, it is better to omit unset optional params when rendering the final command than to rely only on downstream normalization.
  - The custom node `command` property now drops `None` and empty-list values before building the shell command.
- For list-valued params passed through `kwdagger`, prefer a single key/value YAML string over `nargs='*'`.
  - Current audit/HELM example:
    - `helm.enable_huggingface_models: '["repo-a", "repo-b"]'`
    - `helm.enable_local_huggingface_models: '["/models/a"]'`
  - The pipeline-facing script can decode that with `kwutil.Yaml.coerce`, then expand it into the downstream CLI format if needed.
- `scriptconfig` emitted a smartcast warning for comma-separated `devices`; this did not block scheduling, but it is a hint that list parsing behavior may change in a future version.
- Existing `cmd_queue` tmux sessions with the same queue name can trigger an interactive prompt asking whether older sessions should be killed.
  - This is important for automation and unattended runs.
  - Current mitigation in the audit runner: derive `--queue_name` from `experiment_name` instead of reusing the generic `schedule-eval`.
- The audit comparison step needs to discover completed jobs recursively under the kwdagger results root.
  - Real layout is nested like: `<results>/<node_name>/<job_id>/DONE`
  - A shallow glob like `results/*/DONE` will miss all HELM jobs and incorrectly report `missing_kwdg_match`.
