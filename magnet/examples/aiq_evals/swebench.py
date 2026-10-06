"""Prepare tiny coding benchmark recipes for the existing EvaluationNode."""
from __future__ import annotations

import json
from pathlib import Path


def recipe(request: dict, *, name: str, endpoint: str, worker_python: str, select: dict) -> dict:
    algo = {key: value for key, value in request.items() if key != 'schema_version'}
    algo['select'] = select
    return {'name': name, 'title': name, 'description': 'Coding benchmark protocol smoke test',
        'version': '1.0', 'organizations': [],
        'submitter': {'name': 'Example', 'email': 'example@example.invalid'}, 'links': [],
        'tags': ['example', 'aiq-magnet-evals', 'swebench'],
        'claim': {'python': 'assert metrics.evaluate.eligible'},
        'evidence': {'scope': 'requested'},
        'kwdagger': {'result_node': 'evaluate', 'pipeline': {'nodes': {'evaluate': {
            'class': 'magnet.backends.aiq_evals.EvaluationNode', 'algo_params': algo,
            'perf_params': {'endpoint': endpoint, 'worker_python': worker_python,
                            'verify_serving_provenance': True}}}}}}


def main(argv=True):
    import kwconf
    import yaml

    class Config(kwconf.Config):
        kind = kwconf.Value('verified', choices=['verified', 'pro', 'harbor'])
        endpoint = kwconf.Value(None, required=True, help='infer-stack alias')
        served_model = kwconf.Value(None, help='served name; defaults to alias')
        worker_python = kwconf.Value(None, required=True)
        output = kwconf.Value(None, required=True, help='output YAML recipe')
        evals_root = kwconf.Value(None, help='evaluator source checkout')
        pro_root = kwconf.Value(None, help='pinned Pro V2 checkout')
        image_pins = kwconf.Value(None, help='Pro {instance: sha256:image-id} JSON')
        task_path = kwconf.Value(None, help='Harbor task directory')
        agent = kwconf.Value('oracle', help='Harbor built-in or python:module:Class')

    args = Config.cli(argv=argv, strict=True, special_options=False)
    import magnet_evals
    from magnet_evals import EvaluationRequest, ModelBinding

    root = Path(args.evals_root or Path(magnet_evals.__file__).resolve().parents[1])
    binding = ModelBinding(role='primary', model=args.served_model or args.endpoint,
                           provider='openai', provider_options=(
                               {'responses_api': False} if args.kind == 'verified' else {}))
    if args.kind == 'verified':
        request = EvaluationRequest.from_dict(json.loads((root / 'examples/swebench_verified_request.json').read_text()))
        from dataclasses import replace

        request = replace(request, models=(binding,))
        select = {'scorer': 'swe_bench_scorer', 'metric': 'accuracy'}
    elif args.kind == 'pro':
        from magnet_evals.benchmarks.swe_bench_pro import profile_request

        if not args.pro_root or not args.image_pins:
            raise ValueError('Pro smoke needs pro_root and image_pins')
        request = profile_request(args.pro_root, 'smoke', binding,
                                  json.loads(Path(args.image_pins).read_text()))
        select = {'metric': 'mean', 'score': 'reward'}
    else:
        if not args.task_path:
            raise ValueError('Harbor needs task_path')
        request = EvaluationRequest(engine='harbor', task='path:' + str(Path(args.task_path).resolve()),
                                    data_revision='caller-selected', models=(binding,),
                                    task_options={'agent': args.agent})
        select = {'metric': 'mean', 'score': 'reward'}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(recipe(request.to_dict(), name='swebench_' + args.kind + '_smoke',
        endpoint=args.endpoint, worker_python=args.worker_python, select=select), sort_keys=False))
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
