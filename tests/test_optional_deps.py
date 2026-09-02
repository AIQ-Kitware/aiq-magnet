"""
The core of magnet -- cards, claims, theory, kwdagger execution, containers,
leasing, the CLI -- must import without the ``helm`` extra. A regression here
means a team that only writes a kwdagger pipeline over its own artifacts is
made to install torch to run it.
"""
import subprocess
import sys
import textwrap

CORE = [
    'magnet',
    'magnet.evaluation',
    'magnet.evaluation_new',
    'magnet.theory',
    'magnet.containers',
    'magnet.leasing',
    'magnet.cli',
    'magnet.__main__',
]

HELM_ONLY = [
    'magnet.backends.helm.helm_outputs',
    'magnet.predictor',
    'magnet.helm_inference',
    'magnet.instance_predictor',
    'magnet.perturb_instances',
]

PROBE = textwrap.dedent('''
    import importlib, json, sys
    blocked = set(sys.argv[1].split(','))
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in blocked:
                raise ImportError('blocked: ' + name)
    sys.meta_path.insert(0, Block())
    out = {}
    for m in sys.argv[2].split(','):
        try:
            importlib.import_module(m)
            out[m] = 'ok'
        except Exception as ex:
            out[m] = type(ex).__name__ + ': ' + str(ex)[:160]
    print(json.dumps(out))
''')


def _probe(blocked, modules):
    import json
    proc = subprocess.run(
        [sys.executable, '-c', PROBE, ','.join(blocked), ','.join(modules)],
        capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_core_imports_without_helm():
    got = _probe(['helm', 'sklearn', 'gcsfs', 'torch'], CORE)
    bad = {k: v for k, v in got.items() if v != 'ok'}
    assert not bad, bad


def test_helm_modules_name_the_extra():
    got = _probe(['helm'], HELM_ONLY)
    for mod, msg in got.items():
        assert msg.startswith('MissingOptionalDependency'), (mod, msg)
        assert "aiq-magnet[helm]" in msg, (mod, msg)
