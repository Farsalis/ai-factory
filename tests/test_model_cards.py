"""Tests for Hugging Face model card and run summary generation.

Covers the artifact layout contract:

* ``final_merged_model/README.md`` — SFT-stage card with HF frontmatter.
* ``dpo_model/checkpoint-<step>/README.md`` — DPO card per checkpoint.
* ``dpo_model/dpo_merged_model/README.md`` — DPO card for the merged model.
* ``training_output/README.md`` — overall run summary.
* ``final_adapter/`` — never receives a model card.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

pytest.importorskip("pydantic")
pytest.importorskip("huggingface_hub")

from src.config import ScriptConfig
from src.model_cards import (
    MODEL_CARD_FILENAME,
    DPOCardContext,
    collect_trainer_stats,
    read_checkpoint_stats,
    summarize_stats,
    write_dpo_model_card,
    write_pipeline_run_summary,
    write_sft_model_card,
)

BASE_MODEL = "Qwen/Qwen3.5-9B"


@pytest.fixture
def sample_config(tmp_path: Path) -> ScriptConfig:
    """Build a minimal validated ScriptConfig rooted in tmp_path."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    val_file.write_text('{"a": 3}\n', encoding="utf-8")
    config_dict: dict[str, Any] = {
        "data": {"train_file": str(train_file), "validation_file": str(val_file)},
        "model": {
            "name": BASE_MODEL,
            "max_length": 256,
            "attn_implementation": "sdpa",
            "license": "apache-2.0",
        },
        "quantization": {"enabled": True, "quant_type": "nf4"},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        "training": {"output_dir": str(tmp_path / "training_output")},
        "dpo": {"output_dir": str(tmp_path / "training_output" / "dpo_model")},
    }
    return ScriptConfig(**config_dict)


def _sample_stats() -> dict[str, Any]:
    """Stats shaped like collect_trainer_stats output."""
    return {
        "metrics": {
            "train_loss": 1.25,
            "train_runtime": 7200.0,
            "train_samples_per_second": 1.5,
            "epoch": 1.0,
        },
        "log_history": [
            {"loss": 2.0, "step": 10},
            {"eval_loss": 1.5, "step": 10},
            {"loss": 1.25, "step": 100},
            {"eval_loss": 1.1, "step": 100},
        ],
    }


def _dpo_context() -> DPOCardContext:
    return DPOCardContext(
        base_model=BASE_MODEL,
        sft_model_path="training_output/final_merged_model",
        train_data="900 chosen/rejected tool-selection pairs.",
        license_id="apache-2.0",
        hyperparameters={"max_steps": 100, "beta": 0.1},
        hardware_type="NVIDIA GeForce RTX 4070",
    )


def _frontmatter(card_text: str) -> dict[str, Any]:
    """Parse the YAML frontmatter block of a rendered card."""
    assert card_text.startswith("---\n")
    block = card_text.split("---\n")[1]
    parsed = yaml.safe_load(block)
    assert isinstance(parsed, dict)
    return parsed


class TestStatsHelpers:
    """collect_trainer_stats and summarize_stats behavior."""

    @pytest.mark.unit
    def test_collect_trainer_stats_from_trainer_like_objects(self) -> None:
        """Real-shaped metrics and log history are captured."""
        train_result = SimpleNamespace(metrics={"train_loss": 0.5})
        trainer = SimpleNamespace(
            state=SimpleNamespace(log_history=[{"loss": 0.5, "step": 5}])
        )
        stats = collect_trainer_stats(train_result, trainer)
        assert stats["metrics"] == {"train_loss": 0.5}
        assert stats["log_history"] == [{"loss": 0.5, "step": 5}]

    @pytest.mark.unit
    def test_collect_trainer_stats_tolerates_mocks(self) -> None:
        """Objects without proper metrics/log_history degrade to empty stats."""
        stats = collect_trainer_stats(object(), object())
        assert stats == {"metrics": {}, "log_history": []}

    @pytest.mark.unit
    def test_summarize_stats_picks_final_losses(self) -> None:
        """The last train/eval losses and step count win."""
        summary = summarize_stats(_sample_stats())
        assert summary["final_train_loss"] == 1.25
        assert summary["final_eval_loss"] == 1.1
        assert summary["total_steps"] == 100
        assert summary["train_runtime_seconds"] == 7200.0

    @pytest.mark.unit
    def test_summarize_stats_empty(self) -> None:
        """None and empty stats produce an empty summary."""
        assert summarize_stats(None) == {}
        assert summarize_stats({"metrics": {}, "log_history": []}) == {}


class TestSFTModelCard:
    """SFT-stage card written next to the merged SFT model."""

    @pytest.mark.unit
    def test_frontmatter_conforms_to_hf_schema(
        self, sample_config: ScriptConfig, tmp_path: Path
    ) -> None:
        """base_model, library_name, license, tags, pipeline_tag are set."""
        merged_dir = sample_config.training.merged_model_path
        card_path = write_sft_model_card(
            sample_config, merged_dir, _sample_stats(), hardware_type="RTX 4070"
        )

        assert card_path == merged_dir / MODEL_CARD_FILENAME
        data = _frontmatter(card_path.read_text(encoding="utf-8"))
        assert data["base_model"] == BASE_MODEL
        assert data["library_name"] == "transformers"
        assert data["license"] == "apache-2.0"
        assert data["pipeline_tag"] == "text-generation"
        assert "sft" in data["tags"] and "qlora" in data["tags"]

    @pytest.mark.unit
    def test_standard_sections_and_stage_content(
        self, sample_config: ScriptConfig
    ) -> None:
        """Official template headings plus SFT-specific details are present."""
        card_path = write_sft_model_card(
            sample_config,
            sample_config.training.merged_model_path,
            _sample_stats(),
            hardware_type="RTX 4070",
        )
        text = card_path.read_text(encoding="utf-8")

        for heading in (
            "## Model Details",
            "### Model Description",
            "## Training Details",
            "### Training Data",
            "#### Training Hyperparameters",
            "## Evaluation",
            "### Results",
            "## Environmental Impact",
        ):
            assert heading in text, f"missing template heading: {heading}"

        assert "train.jsonl" in text
        assert "lora_r | 8" in text
        assert "Final training loss | 1.25" in text
        assert "Final evaluation loss | 1.1" in text
        assert "RTX 4070" in text
        assert "2.00" in text  # hours_used from 7200s runtime

    @pytest.mark.unit
    def test_license_omitted_when_unconfigured(
        self, sample_config: ScriptConfig
    ) -> None:
        """No license key appears in frontmatter when model.license is unset."""
        sample_config.model.license = None
        card_path = write_sft_model_card(
            sample_config, sample_config.training.merged_model_path, None
        )
        data = _frontmatter(card_path.read_text(encoding="utf-8"))
        assert "license" not in data


class TestDPOModelCard:
    """DPO-stage cards for checkpoints and the merged DPO model."""

    @pytest.mark.unit
    def test_checkpoint_card_uses_peft_library(self, tmp_path: Path) -> None:
        """Checkpoint cards are PEFT artifacts based on the original hub id."""
        checkpoint = tmp_path / "dpo_model" / "checkpoint-100"
        card_path = write_dpo_model_card(
            checkpoint,
            _dpo_context(),
            _sample_stats(),
            library_name="peft",
            artifact_note="Checkpoint at step 100.",
        )
        text = card_path.read_text(encoding="utf-8")
        data = _frontmatter(text)
        assert data["library_name"] == "peft"
        assert data["base_model"] == BASE_MODEL
        assert "dpo" in data["tags"]
        assert "Checkpoint at step 100." in text
        assert "final_merged_model" in text
        assert "beta | 0.1" in text

    @pytest.mark.unit
    def test_merged_card_uses_transformers_library(self, tmp_path: Path) -> None:
        """The merged DPO model card is a standalone transformers artifact."""
        merged = tmp_path / "dpo_model" / "dpo_merged_model"
        card_path = write_dpo_model_card(
            merged, _dpo_context(), None, library_name="transformers"
        )
        data = _frontmatter(card_path.read_text(encoding="utf-8"))
        assert data["library_name"] == "transformers"
        assert data["license"] == "apache-2.0"

    @pytest.mark.unit
    def test_read_checkpoint_stats_round_trip(self, tmp_path: Path) -> None:
        """trainer_state.json log history feeds the merged-model card."""
        checkpoint = tmp_path / "checkpoint-100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            '{"log_history": [{"loss": 0.42, "step": 100}]}', encoding="utf-8"
        )
        stats = read_checkpoint_stats(checkpoint)
        assert summarize_stats(stats)["final_train_loss"] == 0.42

    @pytest.mark.unit
    def test_read_checkpoint_stats_missing_file(self, tmp_path: Path) -> None:
        """A checkpoint without trainer_state.json yields empty stats."""
        assert read_checkpoint_stats(tmp_path) == {"metrics": {}, "log_history": []}


class TestRunSummary:
    """Top-level training_output/README.md run report."""

    @pytest.mark.unit
    def test_summary_covers_datasets_hardware_stats_artifacts(
        self, sample_config: ScriptConfig
    ) -> None:
        """The summary reports datasets, hardware, losses, and artifact paths."""
        out = sample_config.training.output_dir
        sample_config.training.adapter_path.mkdir(parents=True)
        sample_config.training.merged_model_path.mkdir(parents=True)
        (sample_config.dpo_output_dir / "checkpoint-100").mkdir(parents=True)
        sample_config.dpo_merged_model_path.mkdir(parents=True)

        summary_path = write_pipeline_run_summary(
            sample_config,
            sft_stats=_sample_stats(),
            dpo_stats={"metrics": {}, "log_history": [{"loss": 0.3, "step": 100}]},
            phase_runtimes={"sft": 7200.0, "merge": 300.0, "dpo": 1800.0},
            hardware={"Device": "RTX 4070", "VRAM (GB)": 12.0},
        )

        assert summary_path == out / MODEL_CARD_FILENAME
        text = summary_path.read_text(encoding="utf-8")

        assert "train.jsonl` (2 records)" in text
        assert "val.jsonl` (1 records)" in text
        assert "RTX 4070" in text
        assert "final_train_loss | 1.25" in text
        assert "final_train_loss | 0.3" in text
        assert "`final_adapter/`" in text and "no model card" in text
        assert "`final_merged_model/`" in text
        assert "`dpo_model/checkpoint-100/`" in text
        assert "`dpo_model/dpo_merged_model/`" in text

    @pytest.mark.unit
    def test_summary_marks_missing_artifacts(self, sample_config: ScriptConfig) -> None:
        """Artifacts that were not produced are flagged, not omitted."""
        summary_path = write_pipeline_run_summary(sample_config)
        text = summary_path.read_text(encoding="utf-8")
        assert "*(not produced by this run)*" in text

    @pytest.mark.unit
    def test_summary_is_not_a_model_card(self, sample_config: ScriptConfig) -> None:
        """The run summary is a plain report, not a frontmattered model card."""
        summary_path = write_pipeline_run_summary(sample_config)
        text = summary_path.read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert text.startswith("# Training run summary")


class TestAdapterHasNoCard:
    """final_adapter must remain card-free."""

    @pytest.mark.unit
    def test_run_summary_never_writes_into_adapter_dir(
        self, sample_config: ScriptConfig
    ) -> None:
        """Generating all cards and the summary leaves final_adapter card-free."""
        adapter_dir = sample_config.training.adapter_path
        adapter_dir.mkdir(parents=True)
        write_sft_model_card(
            sample_config, sample_config.training.merged_model_path, _sample_stats()
        )
        write_pipeline_run_summary(sample_config, sft_stats=_sample_stats())
        assert not (adapter_dir / MODEL_CARD_FILENAME).exists()


class TestConfigArtifactPaths:
    """Config-driven artifact path resolution."""

    @pytest.mark.unit
    def test_default_layout(self, sample_config: ScriptConfig) -> None:
        """Default dirnames produce the documented artifact tree."""
        out = sample_config.training.output_dir
        assert sample_config.training.adapter_path == out / "final_adapter"
        assert sample_config.training.merged_model_path == out / "final_merged_model"
        assert sample_config.dpo_output_dir == out / "dpo_model"
        assert (
            sample_config.dpo_merged_model_path
            == out / "dpo_model" / "dpo_merged_model"
        )

    @pytest.mark.unit
    def test_dpo_defaults_without_dpo_section(
        self, sample_config: ScriptConfig
    ) -> None:
        """Without a dpo config section, DPO paths derive from training output."""
        sample_config.dpo = None
        out = sample_config.training.output_dir
        assert sample_config.dpo_output_dir == out / "dpo_model"
        assert (
            sample_config.dpo_merged_model_path
            == out / "dpo_model" / "dpo_merged_model"
        )

    @pytest.mark.unit
    def test_dirnames_are_configurable(self, sample_config: ScriptConfig) -> None:
        """Custom dirnames flow through every derived path."""
        sample_config.training.adapter_dirname = "adapter"
        sample_config.training.merged_model_dirname = "sft_merged"
        assert sample_config.dpo is not None
        sample_config.dpo.merged_model_dirname = "merged"
        out = sample_config.training.output_dir
        assert sample_config.training.adapter_path == out / "adapter"
        assert sample_config.training.merged_model_path == out / "sft_merged"
        assert sample_config.dpo_merged_model_path == out / "dpo_model" / "merged"


class TestAttribution:
    """Developed by / Model Card Authors / Model Card Contact plumbing."""

    @pytest.mark.unit
    def test_sft_card_renders_attribution_from_config(
        self, sample_config: ScriptConfig
    ) -> None:
        """Config attribution replaces the template placeholders on SFT cards."""
        sample_config.model.developers = "Overture System Solutions (O.S.S.)"
        sample_config.model.card_authors = "Overture System Solutions (O.S.S.)"
        sample_config.model.card_contacts = [
            "Samuel Conrad - samuel.conrad@osscontact.com",
            "Jordan Martens - jordan.martens@osscontact.com",
        ]

        card_path = write_sft_model_card(
            sample_config, sample_config.training.merged_model_path, None
        )
        text = card_path.read_text(encoding="utf-8")

        assert "**Developed by:** Overture System Solutions (O.S.S.)" in text
        authors = text.split("## Model Card Authors")[1]
        assert "Overture System Solutions (O.S.S.)" in authors
        contact = text.split("## Model Card Contact")[1]
        assert "Samuel Conrad - samuel.conrad@osscontact.com" in contact
        assert "Jordan Martens - jordan.martens@osscontact.com" in contact
        assert "[More Information Needed]" not in contact

    @pytest.mark.unit
    def test_contacts_render_on_separate_lines(
        self, sample_config: ScriptConfig
    ) -> None:
        """Each contact is its own markdown block, not a run-on line."""
        sample_config.model.card_contacts = ["First - a@x.com", "Second - b@x.com"]
        card_path = write_sft_model_card(
            sample_config, sample_config.training.merged_model_path, None
        )
        contact = card_path.read_text(encoding="utf-8").split("## Model Card Contact")[
            1
        ]
        assert "First - a@x.com\n\nSecond - b@x.com" in contact

    @pytest.mark.unit
    def test_attribution_placeholders_remain_when_unset(
        self, sample_config: ScriptConfig
    ) -> None:
        """Without config attribution the HF placeholders are left intact."""
        card_path = write_sft_model_card(
            sample_config, sample_config.training.merged_model_path, None
        )
        text = card_path.read_text(encoding="utf-8")
        assert "**Developed by:** [More Information Needed]" in text
        assert "[More Information Needed]" in text.split("## Model Card Contact")[1]

    @pytest.mark.unit
    def test_dpo_card_renders_attribution_from_context(self, tmp_path: Path) -> None:
        """DPO cards carry the attribution supplied on the card context."""
        context = DPOCardContext(
            base_model=BASE_MODEL,
            sft_model_path="training_output/final_merged_model",
            train_data="Preference pairs.",
            developers="Overture System Solutions (O.S.S.)",
            card_authors="Overture System Solutions (O.S.S.)",
            card_contacts=(
                "Samuel Conrad - samuel.conrad@osscontact.com",
                "Jordan Martens - jordan.martens@osscontact.com",
            ),
        )
        card_path = write_dpo_model_card(
            tmp_path / "dpo_merged_model", context, None, library_name="transformers"
        )
        text = card_path.read_text(encoding="utf-8")

        assert "**Developed by:** Overture System Solutions (O.S.S.)" in text
        contact = text.split("## Model Card Contact")[1]
        assert "samuel.conrad@osscontact.com" in contact
        assert "jordan.martens@osscontact.com" in contact

    @pytest.mark.unit
    def test_blank_contacts_are_ignored(self, sample_config: ScriptConfig) -> None:
        """Empty/whitespace contact entries do not create blank lines."""
        sample_config.model.card_contacts = ["  ", "", "Only - only@x.com"]
        card_path = write_sft_model_card(
            sample_config, sample_config.training.merged_model_path, None
        )
        contact = card_path.read_text(encoding="utf-8").split("## Model Card Contact")[
            1
        ]
        assert contact.strip().splitlines()[0] == "Only - only@x.com"

    @pytest.mark.unit
    def test_shipped_config_yaml_carries_attribution(self) -> None:
        """The repo's config.yaml defines the expected O.S.S. attribution."""
        from src.main import load_config_from_yaml

        config = load_config_from_yaml(Path("src/config.yaml"))
        assert config.model.card_authors == "Overture System Solutions (O.S.S.)"
        assert config.model.card_contacts == [
            "Samuel Conrad - samuel.conrad@osscontact.com",
            "Jordan Martens - jordan.martens@osscontact.com",
        ]
