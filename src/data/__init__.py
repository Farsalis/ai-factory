"""Data processing package.

This package contains data generation and processing utilities.
"""

import logging
import random
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from src.config import DataConfig

logger = logging.getLogger(__name__)

# Constants
PERTURBATION_PROBABILITY = 0.5
LABEL_PAD_TOKEN_ID = -100
DEFAULT_RESPONSE_TEMPLATE = "Assistant: "

# ICDU required fields
REQUIRED_ICDU_FIELDS = [
    "application_prompt",
    "ideal_response_final",
    "persona_archetype",
    "capability_layer",
    "context_summary",
    "user_intent",
]

# Capability layer system prompts
SYSTEM_PROMPTS = {
    "Foundational": (
        "You are a factual and transparent assistant providing direct, "
        "unambiguous answers to user queries. Focus on clarity and practical "
        "information, ensuring responses are concise and helpful."
    ),
    "Transformational": (
        "You are an empathetic and insightful assistant. Guide users through "
        "complex decisions by reframing problems around their preferences and "
        "asking clarifying questions to ensure clarity in decision-making."
    ),
    "Aspirational": (
        "You are a knowledgeable and proactive partner. Affirm user choices, "
        "provide deep insights, and anticipate unstated needs to foster a "
        "long-term collaborative relationship."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant providing clear and accurate responses."
)

# Context perturbation patterns
PERTURBATION_PATTERNS = {
    "budget": "budget constraints",
    "modern_design": "modern, tech-inspired design",
    "limited_edition": "limited-edition models",
}

# Prompt adjustments for perturbed contexts
PROMPT_ADJUSTMENTS = {
    "budget": " I'm also looking to stay within a reasonable budget.",
    "modern_design": " I prefer something with a modern, techy look.",
    "limited_edition": " I'm curious about limited-edition options.",
}


def _apply_carla_perturbations(context: str) -> str:
    """Apply perturbations specific to Carla persona.

    Args:
        context: Original context string.

    Returns:
        Perturbed context string.
    """
    context = context.replace("busy professional", "retired hobbyist")
    return f"{context} The user is now considering {PERTURBATION_PATTERNS['budget']}."


def _apply_ben_perturbations(context: str) -> str:
    """Apply perturbations specific to Ben persona.

    Args:
        context: Original context string.

    Returns:
        Perturbed context string.
    """
    context = context.replace("30th birthday", "anniversary gift")
    return f"{context} The user prefers a {PERTURBATION_PATTERNS['modern_design']}."


def _apply_alex_perturbations(context: str) -> str:
    """Apply perturbations specific to Alex persona.

    Args:
        context: Original context string.

    Returns:
        Perturbed context string.
    """
    context = context.replace("dive watch", "chronograph watch")
    pat = PERTURBATION_PATTERNS["limited_edition"]
    return f"{context} The user is interested in {pat}."


def perturb_context(context: str, persona: str, intent: str) -> str:
    """Apply the Scenario-Perturbation Method to generate varied contexts.

    Applies persona-specific perturbations to encourage model application
    over simple recitation of training data.

    Args:
        context: Original context summary string.
        persona: Persona archetype name (e.g., "Carla", "Ben", "Alex").
        intent: User intent (currently unused but kept for API compatibility).

    Returns:
        Perturbed context string with persona-specific modifications.
    """
    persona_lower = persona.lower()
    if "carla" in persona_lower:
        return _apply_carla_perturbations(context)
    elif "ben" in persona_lower:
        return _apply_ben_perturbations(context)
    elif "alex" in persona_lower:
        return _apply_alex_perturbations(context)
    return context


def _adjust_prompt_for_perturbation(prompt: str, context: str) -> str:
    """Adjust user prompt to reflect perturbed context.

    Args:
        prompt: Original user prompt.
        context: Perturbed context string.

    Returns:
        Adjusted prompt with additional context-specific information.
    """
    for pattern_key, pattern in PERTURBATION_PATTERNS.items():
        if pattern in context:
            return f"{prompt}{PROMPT_ADJUSTMENTS[pattern_key]}"
    return prompt


def _format_messages_fallback(messages: list[dict[str, str]]) -> str:
    """Format messages using fallback manual formatting.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Formatted string representation of messages.
    """
    return (
        f"System: {messages[0]['content']}\n"
        f"User: {messages[1]['content']}\n"
        f"Assistant: {messages[2]['content']}\n"
    )


def format_icdu_to_chat(
    example: dict[str, Any], tokenizer: PreTrainedTokenizer
) -> dict[str, str]:
    """Convert an ICDU example to a chat-formatted string with optional perturbation.

    Processes ICDU examples by:
    1. Validating required fields
    2. Optionally applying context perturbations (50% probability)
    3. Constructing system prompts based on capability layer
    4. Formatting as chat messages using tokenizer's chat template

    Args:
        example: ICDU example dictionary with required fields.
        tokenizer: Pre-trained tokenizer with optional chat_template support.

    Returns:
        Dictionary with "text" key containing formatted chat string.
        Returns empty text if example is invalid.
    """
    # Validate required fields
    missing_fields = [f for f in REQUIRED_ICDU_FIELDS if f not in example]
    if missing_fields:
        logger.warning(
            f"Skipping invalid ICDU: missing required fields {missing_fields}. "
            f"Found: {list(example.keys())}"
        )
        return {"text": ""}

    # Extract fields
    prompt = example["application_prompt"]
    response = example["ideal_response_final"]
    persona = example["persona_archetype"]
    layer = example["capability_layer"]
    context = example["context_summary"]
    intent = example["user_intent"]

    # Apply perturbation with configured probability
    if random.random() < PERTURBATION_PROBABILITY:  # noqa: S311
        context = perturb_context(context, persona, intent)
        prompt = _adjust_prompt_for_perturbation(prompt, context)

    # Construct system prompt based on capability layer
    system_content = SYSTEM_PROMPTS.get(layer, DEFAULT_SYSTEM_PROMPT)

    # Format as chat messages
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_content}\n"
                f"Context: {context}\n"
                f"User Intent: {intent}\n"
                f"Persona: {persona}"
            ),
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # Apply chat template if available, otherwise use fallback
    try:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = _format_messages_fallback(messages)
    except Exception as err:
        logger.warning(
            f"Chat template application failed ('{err}'). Using fallback formatting."
        )
        text = _format_messages_fallback(messages)

    return {"text": str(text)}


def load_and_prepare_dataset(
    config: DataConfig, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Load, format, and prepare the ICDU dataset for training.

    Loads JSON dataset files, applies chat formatting with optional perturbations,
    and filters out empty or invalid entries.

    Args:
        config: DataConfig containing train_file and validation_file paths.
        tokenizer: Pre-trained tokenizer for chat template formatting.

    Returns:
        Formatted Hugging Face Dataset ready for training.

    Raises:
        FileNotFoundError: If data files specified in config don't exist.
        ValueError: If dataset is empty after formatting and filtering.
    """
    logger.info(
        f"Loading dataset from {config.train_file} and {config.validation_file}"
    )
    data_files = {
        "train": str(config.train_file),
        "validation": str(config.validation_file),
    }

    try:
        dataset = load_dataset("json", data_files=data_files)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    logger.info("Applying ICDU chat formatting to dataset...")
    formatted_dataset = dataset.map(
        lambda x: format_icdu_to_chat(x, tokenizer),
        remove_columns=list(dataset["train"].features),
        desc="Formatting ICDU examples",
    )

    # Filter out empty or invalid entries
    initial_size = len(formatted_dataset["train"]) + len(
        formatted_dataset["validation"]
    )
    formatted_dataset = formatted_dataset.filter(
        lambda x: x["text"].strip() != "", desc="Filtering empty entries"
    )
    final_size = len(formatted_dataset["train"]) + len(formatted_dataset["validation"])

    if final_size == 0:
        raise ValueError("Dataset is empty after formatting and filtering")

    filtered_count = initial_size - final_size
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} empty entries")

    logger.info(
        f"Prepared dataset: {len(formatted_dataset['train'])} train, "
        f"{len(formatted_dataset['validation'])} validation examples"
    )

    return formatted_dataset


class VectorizedCompletionOnlyCollator:
    """Vectorized collator that masks prompt tokens for completion-only fine-tuning.

    This collator pads sequences, finds the response template in each sequence,
    and masks all tokens before the template (including the template itself) so
    that only the assistant's response is used for loss computation.

    Attributes:
        tokenizer: Pre-trained tokenizer for padding and encoding.
        label_pad_token_id: Token ID to use for masked labels (default: -100).
        response_template_ids: Token IDs for the response template string.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        response_template: str = DEFAULT_RESPONSE_TEMPLATE,
        label_pad_token_id: int = LABEL_PAD_TOKEN_ID,
    ) -> None:
        """Initialize the collator.

        Args:
            tokenizer: Pre-trained tokenizer.
            response_template: Template string marking start of assistant response.
            label_pad_token_id: Token ID for masked labels (ignored in loss).
        """
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

        if not self.response_template_ids:
            raise ValueError(
                f"Response template '{response_template}' encoded to empty token list"
            )

        logger.info(
            f"Response template '{response_template}' tokenized to IDs: "
            f"{self.response_template_ids}"
        )

    def _find_template_start(self, padded_input_ids: torch.Tensor) -> torch.Tensor:
        """Find the start index of the response template in each sequence.

        Uses vectorized sliding window matching to efficiently find template
        positions across the batch.

        Args:
            padded_input_ids: Padded input token IDs [batch_size, seq_len].

        Returns:
            Tensor of start indices [batch_size], with seq_len for sequences
            without a match.
        """
        batch_size, seq_len = padded_input_ids.shape
        template_len = len(self.response_template_ids)

        if template_len > seq_len:
            # Template longer than sequence, no match possible
            return torch.full((batch_size,), seq_len, device=padded_input_ids.device)

        # Create sliding windows of template length
        unfolded_ids = padded_input_ids.unfold(
            dimension=1, size=template_len, step=1
        )  # [batch_size, num_windows, template_len]

        # Ensure we have at least one window
        if unfolded_ids.shape[1] == 0:
            return torch.full((batch_size,), seq_len, device=padded_input_ids.device)

        # Create template tensor for comparison with proper shape for broadcasting
        # unfolded_ids is [batch_size, num_windows, template_len]
        # template_tensor should be [1, 1, template_len] to broadcast correctly
        template_tensor = torch.tensor(
            self.response_template_ids,
            device=unfolded_ids.device,
            dtype=unfolded_ids.dtype,
        ).view(1, 1, -1)  # [1, 1, template_len]

        # Find matches: [batch_size, num_windows]
        # Compare each window with template
        matches = (unfolded_ids == template_tensor).all(dim=2)

        # Ensure matches is 2D (should always be, but safety check)
        if matches.dim() != 2:
            # If somehow 1D, reshape it
            if matches.dim() == 1 and batch_size == 1:
                matches = matches.unsqueeze(0)
            else:
                # Unexpected shape, return no match
                return torch.full(
                    (batch_size,), seq_len, device=padded_input_ids.device
                )

        # Find first match index for each sequence
        # argmax returns index of first True, or 0 if all False
        match_indices = torch.argmax(matches.int(), dim=1)  # [batch_size]

        # Handle sequences with no match
        no_match_mask = ~matches.any(dim=1)
        match_indices[no_match_mask] = seq_len

        # For sequences with matches, add template_len to get position after template
        # Only add template_len if there was actually a match
        has_match_mask = ~no_match_mask
        match_indices[has_match_mask] = match_indices[has_match_mask] + template_len

        return match_indices

    def _create_label_mask(
        self, padded_input_ids: torch.Tensor, response_starts: torch.Tensor
    ) -> torch.Tensor:
        """Create mask for tokens to ignore in loss computation.

        Args:
            padded_input_ids: Padded input token IDs [batch_size, seq_len].
            response_starts: Start indices of responses [batch_size].

        Returns:
            Boolean mask [batch_size, seq_len] where True indicates tokens
            to mask (ignore in loss).
        """
        batch_size, seq_len = padded_input_ids.shape

        # Mask everything before response start
        arange = torch.arange(seq_len, device=padded_input_ids.device).expand(
            batch_size, -1
        )
        mask_before_response = arange < response_starts.unsqueeze(1)

        # Also mask padding tokens
        mask_padding = padded_input_ids == self.tokenizer.pad_token_id

        return cast(torch.Tensor, mask_before_response | mask_padding)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of features into padded tensors with masked labels.

        Args:
            features: List of feature dictionaries with "input_ids" and
                "attention_mask" keys.

        Returns:
            Dictionary with:
                - "input_ids": Padded input token IDs [batch_size, seq_len]
                - "attention_mask": Padded attention masks [batch_size, seq_len]
                - "labels": Labels with prompt tokens masked [batch_size, seq_len]
        """
        # Extract and convert to tensors, filtering out invalid entries
        input_ids = []
        attention_masks = []

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        for f in features:
            if "input_ids" not in f:
                continue

            ids = f["input_ids"]
            # Skip if empty or None
            if not ids or len(ids) == 0:
                continue

            # Get or create attention_mask
            if "attention_mask" in f:
                masks = f["attention_mask"]
                # If attention_mask is empty, create it from input_ids
                if not masks or len(masks) == 0:
                    masks = [1 if token_id != pad_token_id else 0 for token_id in ids]
            else:
                # Create attention_mask from input_ids if missing
                masks = [1 if token_id != pad_token_id else 0 for token_id in ids]

            # Ensure lengths match
            if len(ids) != len(masks):
                if len(masks) < len(ids):
                    masks.extend([1] * (len(ids) - len(masks)))
                else:
                    masks = masks[: len(ids)]

            # Convert to tensor, ensuring it's 1D
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            masks_tensor = torch.tensor(masks, dtype=torch.long)

            # Ensure 1D shape
            if ids_tensor.dim() == 0:
                ids_tensor = ids_tensor.unsqueeze(0)
            if masks_tensor.dim() == 0:
                masks_tensor = masks_tensor.unsqueeze(0)

            input_ids.append(ids_tensor)
            attention_masks.append(masks_tensor)

        if not input_ids or not attention_masks:
            # Log details about what went wrong
            invalid_count = len(features) - len(input_ids)
            logger.error(
                f"Empty batch detected: {len(features)} features provided, "
                f"{invalid_count} were invalid, {len(input_ids)} valid. "
                "Sample feature keys: "
                + str(list(features[0].keys()) if features else "no features")
            )
            # Raise an error to prevent training with invalid data
            n_feat, n_valid = len(features), len(input_ids)
            raise ValueError(
                f"Empty batch: {n_feat} features but only {n_valid} had valid "
                "input_ids/attention_mask. Check preprocessing and tokenization."
            )

        if len(input_ids) != len(attention_masks):
            raise ValueError(
                f"Mismatch between input_ids ({len(input_ids)}) and "
                f"attention_mask ({len(attention_masks)}) counts"
            )

        # Ensure we have valid tensors before padding
        if len(input_ids) == 0:
            raise ValueError(
                "Cannot pad empty batch. All features in this batch were invalid. "
                "Check data preprocessing and tokenization."
            )

        padded_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        padded_attention_mask = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        # Ensure output tensors are 2D [batch_size, seq_len]
        if padded_input_ids.dim() != 2:
            nd, shp = padded_input_ids.dim(), padded_input_ids.shape
            raise ValueError(
                f"Expected 2D tensor for input_ids, got {nd}D with shape {shp}"
            )
        if padded_attention_mask.dim() != 2:
            nd = padded_attention_mask.dim()
            shp = padded_attention_mask.shape
            raise ValueError(
                f"Expected 2D tensor for attention_mask, got {nd}D with shape {shp}"
            )

        # Create labels (clone of input_ids)
        labels = padded_input_ids.clone()

        # Find response template start positions
        response_starts = self._find_template_start(padded_input_ids)

        # Create mask for tokens to ignore in loss
        label_mask = self._create_label_mask(padded_input_ids, response_starts)

        # Apply mask to labels
        labels[label_mask] = self.label_pad_token_id

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": labels,
        }


__all__ = [
    "VectorizedCompletionOnlyCollator",
    "format_icdu_to_chat",
    "load_and_prepare_dataset",
]
