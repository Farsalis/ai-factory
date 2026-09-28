"""Pytest configuration shared by the whole test suite.

On Windows, pyarrow crashes with a fatal access violation inside ``pyarrow.lib``
when it is first imported partway through collection, as happens via the
``transformers`` -> ``sklearn`` -> ``pandas`` -> ``pyarrow`` chain that test
modules pull in. Importing it here — conftest is loaded before any test module —
fixes the DLL load order and lets ``pytest tests`` run.
"""

try:  # pragma: no cover - import-order workaround, not behaviour under test
    import pyarrow  # noqa: F401
except ImportError:  # pyarrow is optional for the pure-Python tests
    pass
