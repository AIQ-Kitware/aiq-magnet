"""Tests for the HELM ProcessNode command renderer."""

import pytest

from magnet.backends.helm.pipeline import MaterializeHelmRunNode


def test_helm_command_requires_executable(monkeypatch):
    node = MaterializeHelmRunNode()
    # ProcessNode permits an unset executable; the HELM node should reject it
    # explicitly rather than producing an invalid command or returning None.
    # ProcessNode copies the class default onto each instance at creation.
    # Patching the class leaves this node's executable unchanged.
    monkeypatch.setattr(node, 'executable', None)
    with pytest.raises(ValueError, match='requires an executable'):
        _ = node.command
