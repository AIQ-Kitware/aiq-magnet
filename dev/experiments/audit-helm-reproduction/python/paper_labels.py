from __future__ import annotations

from pathlib import Path
from typing import Any

import kwutil

from common import audit_root


def paper_label_config_fpath() -> Path:
    return audit_root() / 'configs' / 'paper_label_mappings.yaml'


def _load_config() -> dict[str, Any]:
    fpath = paper_label_config_fpath()
    if not fpath.exists():
        return {}
    data = kwutil.Yaml.load(fpath)
    if not isinstance(data, dict):
        raise TypeError(f'Expected mapping in {fpath}')
    return data


class PaperLabelManager:
    """
    Lightweight relabel helper inspired by kwplot.managers.LabelManager.

    This keeps a single checked-in mapping file for paper-facing labels, while
    still letting reports preserve the raw internal machine codes.
    """

    def __init__(self, style: str = 'paper_short'):
        self.style = style
        self.config = _load_config()
        self.machine_map = self.config.get('machine_host', {}) or {}

    def machine_label(self, machine_host: str | None, *, fallback: str | None = None) -> str:
        raw = str(machine_host or fallback or 'unknown')
        info = self.machine_map.get(raw, {}) or {}
        return str(info.get(self.style) or raw)

    def relabel_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        new_text = str(text)
        keys = sorted(self.machine_map.keys(), key=len, reverse=True)
        for key in keys:
            repl = self.machine_label(key)
            new_text = new_text.replace(str(key), repl)
        return new_text


def load_paper_label_manager(style: str = 'paper_short') -> PaperLabelManager:
    return PaperLabelManager(style=style)
