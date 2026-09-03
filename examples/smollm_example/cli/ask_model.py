"""
Ask one leased endpoint every question in the dummy dataset.

This is the node the example exists to demonstrate. It does not resolve a
model, start a server, or know a URL: it reads the two variables
``infer-stack run`` exports into the command it wraps, and talks to whatever is
on the other end. Which model that is comes from the catalog, so pointing this
at a simulator instead of a GPU is ``INFER_STACK_CATALOG``, not a code change.

CommandLine:
    # Never run bare -- there is no endpoint without a lease.
    infer-stack run --endpoint smol-135 -- \
        python -m smollm_example.cli.ask_model \
            --endpoint=smol-135 --items_fpath=items.json \
            --out_fpath=answers.json
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import TypedDict, cast

import kwconf
import ubelt as ub

from smollm_example.cli._types import (
    AnswerRecord,
    AnswersPayload,
    ItemsPayload,
)

__all__ = ['served_model_name', 'ask_one', 'AskModelCLI']

#: What ``infer-stack run`` exports. Their absence is the one failure worth
#: reporting outright: it means the command was not wrapped in a lease, and
#: every request would otherwise fail one at a time with a connection error.
BASE_URL_ENVVAR = 'OPENAI_BASE_URL'
API_KEY_ENVVAR = 'OPENAI_API_KEY'


class _ChatMessage(TypedDict, total=False):
    content: str


class _ChatChoice(TypedDict, total=False):
    message: _ChatMessage


class _ChatResponse(TypedDict, total=False):
    choices: list[_ChatChoice]


def served_model_name(alias: str) -> str:
    """
    The name to put in the request body for a catalog alias.

    infer-stack exports ``INFER_STACK_ENDPOINT_<SLUG>`` per leased endpoint,
    holding the name the gateway actually serves it under. That is usually the
    alias, but need not be -- a catalog may serve two aliases from one
    deployment -- so the exported value wins where there is one.

    Args:
        alias (str): the catalog alias this node leased.

    Returns:
        str: the served model name.

    Example:
        >>> import os
        >>> from smollm_example.cli.ask_model import (
        ...     served_model_name)
        >>> served_model_name('smol-135')
        'smol-135'
        >>> os.environ['INFER_STACK_ENDPOINT_SMOL_135'] = 'served-name'
        >>> served_model_name('smol-135')
        'served-name'
        >>> del os.environ['INFER_STACK_ENDPOINT_SMOL_135']
    """
    slug = re.sub(r'[^A-Z0-9]+', '_', alias.upper()).strip('_')
    return os.environ.get(f'INFER_STACK_ENDPOINT_{slug}') or alias


def ask_one(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, float, str]:
    """
    One chat completion, returning the text and how long it took.

    Returns:
        tuple[str, float, str]: the answer, seconds elapsed, and an error
            string which is empty when the request succeeded. A failed item is
            recorded rather than raised: one refusal should show up as a gap in
            coverage, not as a dead campaign.
    """
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }).encode('utf-8')
    request = urllib.request.Request(
        f'{base_url.rstrip("/")}/chat/completions',
        data=body,
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key or "none"}',
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = cast(
                _ChatResponse,
                json.loads(response.read().decode('utf-8')),
            )
    except (urllib.error.URLError, TimeoutError, OSError) as ex:
        return '', time.monotonic() - started, f'{type(ex).__name__}: {ex}'
    except json.JSONDecodeError as ex:
        return '', time.monotonic() - started, f'bad JSON: {ex}'
    elapsed = time.monotonic() - started
    choices = payload.get('choices') or []
    if not choices:
        return '', elapsed, 'no choices in response'
    message = choices[0].get('message') or {}
    return (message.get('content') or '').strip(), elapsed, ''


def _normalize(text: str) -> str:
    """The first number in a response, so formatting chatter does not count.

    Example:
        >>> from smollm_example.cli.ask_model import _normalize
        >>> _normalize('The answer is 12.')
        '12'
        >>> _normalize('no digits here')
        ''
    """
    found = re.search(r'-?\d+', text or '')
    return found.group(0) if found else ''


class AskModelCLI(kwconf.Config):
    """Ask one leased endpoint every question in the dataset."""

    endpoint: str | None = kwconf.Value(
        None, help='catalog alias this node leased; also the model asked')
    items_fpath: str | None = kwconf.Value(
        None, help='dataset written by make_items',
        tags=['in_path'])
    max_tokens: int = kwconf.Value(16, help='cap on the reply length')
    temperature: float = kwconf.Value(0.0, help='0 for a repeatable answer')
    timeout: float = kwconf.Value(120.0, help='seconds to wait per request')
    out_fpath: str = kwconf.Value(
        'answers.json', help='where to write the answers',
        tags=['out_path', 'primary'])

    @classmethod
    def main(
        cls, argv: bool | list[str] = True, **kwargs: object
    ) -> None:
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')

        base_url = os.environ.get(BASE_URL_ENVVAR)
        if not base_url:
            raise SystemExit(ub.paragraph(
                f"""
                {BASE_URL_ENVVAR} is unset, so this command is not running
                inside a lease. Schedule the card with --per_node_leasing,
                or wrap this invocation in `infer-stack run --endpoint
                <alias> --`. Without it there is no server to ask.
                """
            ))

        alias = str(config['endpoint'] or '').strip()
        if not alias:
            raise SystemExit(
                '--endpoint is required: it names the alias to ask')
        model = served_model_name(alias)
        api_key = os.environ.get(API_KEY_ENVVAR, '')

        items_payload = cast(
            ItemsPayload,
            json.loads(ub.Path(config['items_fpath']).read_text()),
        )
        items = items_payload['items']

        answers: list[AnswerRecord] = []
        for item in items:
            text, elapsed, error = ask_one(
                base_url, api_key, model, item['prompt'],
                int(config['max_tokens']), float(config['temperature']),
                float(config['timeout']),
            )
            answers.append({
                'id': item['id'],
                'expected': item['expected'],
                'answer': text,
                'normalized': _normalize(text),
                'seconds': round(elapsed, 4),
                'error': error,
            })

        answered = [a for a in answers if a['answer']]
        exact = [a for a in answered if a['normalized'] == a['expected']]
        total = len(answers) or 1
        payload: AnswersPayload = {
            'result': {'metrics': {
                'endpoint': alias,
                'served_model': model,
                'n_items': len(answers),
                'n_answered': len(answered),
                # The one that matters here: did every question come back?
                'answered_rate': len(answered) / total,
                # Descriptive only. Against a simulator the text is random, so
                # this is a number about the plumbing, not about the model.
                'exact_rate': len(exact) / total,
                'mean_seconds': (
                    sum(a['seconds'] for a in answers) / total
                ),
            }},
            'answers': answers,
        }
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    AskModelCLI.main()
