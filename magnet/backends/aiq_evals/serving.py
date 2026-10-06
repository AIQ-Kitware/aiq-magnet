"""Bind configured serving facts and verify the realization supplied by a lease."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

SCHEMA = 'infer-stack-serving-provenance/1'


def validate_provenance(value: dict) -> dict:
    payload = value.get('payload')
    if value.get('schema') != SCHEMA or not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('unsupported serving provenance schema')
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = 'sha256:' + hashlib.sha256(encoded.encode()).hexdigest()
    reasons = value.get('unknown_reasons')
    if value.get('digest') != digest:
        raise ValueError('serving provenance digest does not match its payload')
    if not isinstance(reasons, list) or any(not isinstance(r, str) or not r.strip() for r in reasons):
        raise ValueError('serving provenance has invalid uncertainty facts')
    if not isinstance(value.get('immutable'), bool) or value['immutable'] != (not reasons):
        raise ValueError('serving provenance has inconsistent immutability facts')
    return value


def catalog_provenance(roles: dict[str, str]) -> dict:
    """Read scientific facts on the scheduling host, without acquiring a lease."""
    aliases = {}
    for alias in dict.fromkeys(roles.values()):
        process = subprocess.run(['infer-stack', 'catalog', 'endpoint', 'provenance', alias],
                                 capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise ValueError(f'cannot resolve serving provenance for {alias!r}: {process.stderr[-2000:]}')
        aliases[alias] = validate_provenance(json.loads(process.stdout))
    return {role: aliases[alias] for role, alias in roles.items()}


def bind_provenance(request: dict, expected: dict) -> dict:
    request = copy.deepcopy(request)
    bindings = {model['role']: model for model in request['models']}
    for role, value in expected.items():
        value = validate_provenance(value)
        if role not in bindings:
            raise ValueError(f'serving provenance role {role!r} has no model binding')
        model = bindings[role]
        token = model.get('cache_token')
        if token and token != value['digest']:
            raise ValueError(f'model role {role!r} cache_token conflicts with serving provenance; '
                             'remove the token or explicitly disable verify_serving_provenance')
        model['cache_token'] = value['digest']
        reasons = list(model.get('identity_unknown_reasons') or [])
        reasons.extend(value['unknown_reasons'])
        if reasons:
            model['identity_unknown_reasons'] = list(dict.fromkeys(reasons))
    return request


def verify_catalog(roles: dict, expected: dict) -> None:
    if expected and catalog_provenance(roles) != expected:
        raise ValueError('catalog serving provenance changed since scheduling')


def verify_lease(roles: dict, expected: dict, environ=None) -> None:
    if not expected:
        return
    environ = os.environ if environ is None else environ
    path = environ.get('INFER_STACK_ENDPOINT_DESCRIPTOR')
    if not path:
        raise ValueError('lease supplied no public endpoint descriptor')
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get('lease_id') != environ.get('INFER_STACK_LEASE_ID'):
        raise ValueError('endpoint descriptor belongs to a different lease')
    actual = descriptor.get('serving_provenance') or {}
    for role, alias in roles.items():
        if alias not in actual or validate_provenance(actual[alias]) != expected.get(role):
            raise ValueError(f'leased serving provenance for role {role!r} differs from scheduling')
