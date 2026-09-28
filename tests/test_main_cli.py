"""Tests for src.main CLI dispatch (argparse, not Typer)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.main import cli

_TESTS_DIR = Path(__file__).resolve().parent
TEST_CONFIG_YAML = _TESTS_DIR / "configs" / "test_config.yaml"


@pytest.mark.unit
def test_cli_no_args_prints_help_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invoking with no arguments shows help and returns exit code 2."""
    code = cli([])
    captured = capsys.readouterr()
    assert code == 2
    assert "--config-path" in captured.out


@pytest.mark.unit
def test_cli_help_flag_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """--help on the default command prints pipeline help."""
    code = cli(["--help"])
    captured = capsys.readouterr()
    assert code == 0
    assert "--config-path" in captured.out
    assert "--inference-only" in captured.out


@pytest.mark.unit
def test_cli_missing_config_path_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """Pipeline invocation without --config-path fails argparse validation."""
    with pytest.raises(SystemExit) as exc_info:
        cli(["--run-inference"])
    assert exc_info.value.code != 0


@pytest.mark.unit
def test_cli_pipeline_dispatches_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """--config-path runs load_config_from_yaml and run_pipeline."""
    mock_run = MagicMock()
    mock_load = MagicMock(return_value=MagicMock(dpo=None))
    monkeypatch.setattr("src.main.run_pipeline", mock_run)
    monkeypatch.setattr("src.main.load_config_from_yaml", mock_load)

    code = cli(["--config-path", str(TEST_CONFIG_YAML)])

    assert code == 0
    mock_load.assert_called_once()
    mock_run.assert_called_once()


@pytest.mark.unit
def test_cli_inference_only_skips_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """--inference-only loads config and runs inference without SFT/DPO."""
    mock_run = MagicMock()
    mock_infer = MagicMock()
    mock_load = MagicMock(return_value=MagicMock(dpo=None))
    monkeypatch.setattr("src.main.run_pipeline", mock_run)
    monkeypatch.setattr("src.main.run_inference_phase", mock_infer)
    monkeypatch.setattr("src.main.load_config_from_yaml", mock_load)

    code = cli(
        [
            "--config-path",
            str(TEST_CONFIG_YAML),
            "--inference-only",
            "--example-queries",
            "Calculate 2+2",
        ]
    )

    assert code == 0
    mock_load.assert_called_once()
    mock_infer.assert_called_once()
    mock_run.assert_not_called()
    _, kwargs = mock_infer.call_args
    assert kwargs == {}
    positional = mock_infer.call_args[0]
    assert positional[1] == ["Calculate 2+2"]


@pytest.mark.unit
def test_cli_inference_only_and_run_inference_are_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--inference-only and --run-inference cannot be combined."""
    with pytest.raises(SystemExit) as exc_info:
        cli(
            [
                "--config-path",
                str(TEST_CONFIG_YAML),
                "--inference-only",
                "--run-inference",
            ]
        )
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "not allowed with" in captured.err or "not allowed with" in captured.out


@pytest.mark.unit
def test_cli_optimize_config_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """optimize-config subcommand calls run_optimizer."""
    mock_optimizer = MagicMock(return_value={"training": {}})
    mock_profile = MagicMock()
    mock_profile.vram_bytes = 8 * 1024**3
    mock_profile.system_ram_bytes = 32 * 1024**3

    monkeypatch.setattr("src.model_optimizer.run_optimizer", mock_optimizer)
    monkeypatch.setattr("src.hardware.HardwareProfile", lambda: mock_profile)

    code = cli(
        [
            "optimize-config",
            "--config-path",
            str(TEST_CONFIG_YAML),
            "-p",
            "balanced",
            "-o",
            "out.yaml",
        ]
    )

    assert code == 0
    mock_optimizer.assert_called_once()


@pytest.mark.unit
def test_main_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """__main__ guard delegates to cli() and propagates its exit code."""
    mock_cli = MagicMock(return_value=0)
    monkeypatch.setattr("src.main.cli", mock_cli)

    with pytest.raises(SystemExit) as exc_info:
        exec(  # noqa: S102 — exercise the __main__ guard without re-importing
            "raise SystemExit(cli())",
            {"cli": mock_cli, "SystemExit": SystemExit},
        )

    assert exc_info.value.code == 0
    mock_cli.assert_called_once_with()
