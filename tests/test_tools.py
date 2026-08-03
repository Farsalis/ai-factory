"""Smoke test for ``calc_tool`` in ``src.inference_with_tools``.

The standalone agent loop and registry live in ``src.tools``; see
``test_tools_module.py`` for those tests.
"""

import pytest

pytest.importorskip("requests")
pytest.importorskip("torch")

from src.inference_with_tools import calc_tool


@pytest.mark.unit
def test_inference_calc_tool_evaluates_expression() -> None:
    """inference_with_tools.calc_tool evaluates a simple expression."""
    result = calc_tool("2 + 3")
    assert result == "5"
