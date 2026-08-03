from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("pydantic")

from src.config import DataConfig
from src.data import (
    VectorizedCompletionOnlyCollator,
    format_icdu_to_chat,
    load_and_prepare_dataset,
)


@dataclass
class DummyTokenizer:
    pad_token_id: int = 0
    eos_token: str = "</s>"
    chat_template: str | None = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        parts = []
        for message in messages:
            parts.append(f"{message['role']}: {message['content']}")
        return "\n".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) % 255 for char in text]


@pytest.mark.unit
def test_format_icdu_to_chat_missing_fields() -> None:
    tokenizer = DummyTokenizer()
    result = format_icdu_to_chat({"application_prompt": "hi"}, tokenizer)
    assert result == {"text": ""}


@pytest.mark.unit
def test_format_icdu_to_chat_with_template() -> None:
    tokenizer = DummyTokenizer(chat_template="dummy")
    example = {
        "application_prompt": "Help me pick a watch.",
        "ideal_response_final": "Consider size.",
        "persona_archetype": "Carla",
        "capability_layer": "Foundational",
        "context_summary": (
            "A busy professional wants a dive watch for her 30th birthday."
        ),
        "user_intent": "Product recommendation",
    }
    result = format_icdu_to_chat(example, tokenizer)
    assert "system:" in result["text"]
    assert "user:" in result["text"]
    assert "assistant:" in result["text"]


@pytest.mark.integration
def test_load_and_prepare_dataset() -> None:
    tokenizer = DummyTokenizer()
    config = DataConfig(
        train_file="tests/fixtures/train.jsonl",
        validation_file="tests/fixtures/validation.jsonl",
    )
    dataset = load_and_prepare_dataset(config, tokenizer)

    assert "train" in dataset
    assert "validation" in dataset
    assert len(dataset["train"]) > 0
    assert len(dataset["validation"]) > 0


@pytest.mark.unit
def test_vectorized_completion_only_collator_masks_prompt() -> None:
    tokenizer = DummyTokenizer()
    collator = VectorizedCompletionOnlyCollator(
        tokenizer=tokenizer, response_template="Assistant: "
    )

    features = [
        {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "attention_mask": [1, 1, 1, 1, 1, 1],
        }
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert torch.all(batch["labels"] == -100)
