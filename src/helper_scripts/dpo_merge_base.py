"""Merge a DPO LoRA checkpoint into its base model and save the result.

Loads the base (already-merged SFT) model and the DPO PEFT adapter from disk,
applies ``merge_and_unload`` to fold the adapter weights into the base, saves
the merged model to the chosen output directory, and writes a DPO-stage
Hugging Face model card next to the merged weights.

By default the merged model is written to ``<adapter parent>/dpo_merged_model``
(i.e. ``training_output/dpo_model/dpo_merged_model`` for a pipeline-produced
checkpoint), matching the pipeline's artifact layout.

:example:
    >>> python -m src.helper_scripts.dpo_merge_base \\
    ...     --base ./training_output/final_merged_model \\
    ...     --adapter ./training_output/dpo_model/checkpoint-100 \\
    ...     --base-model-id Qwen/Qwen3.5-9B
"""

from pathlib import Path

import typer
from peft import PeftModel

from src.config import DEFAULT_DPO_MERGED_DIRNAME
from src.model_cards import (
    DPOCardContext,
    read_checkpoint_stats,
    write_dpo_model_card,
)
from src.model_setup import resolve_model_class

app = typer.Typer(add_completion=False)


@app.command()
def main(
    base: Path = typer.Option(  # noqa: B008
        ..., exists=True, file_okay=False, help="Base model directory."
    ),
    adapter: Path = typer.Option(  # noqa: B008
        ..., exists=True, file_okay=False, help="DPO PEFT adapter directory."
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        file_okay=False,
        help=(
            "Where to write the merged model. Defaults to "
            f"'{DEFAULT_DPO_MERGED_DIRNAME}' beside the adapter checkpoint."
        ),
    ),
    base_model_id: str | None = typer.Option(
        None,
        help=(
            "Hub identifier of the original base model for the model card "
            "frontmatter (e.g. 'Qwen/Qwen3.5-9B'). Defaults to the base path."
        ),
    ),
    license_id: str | None = typer.Option(
        None,
        "--license",
        help="SPDX license identifier for the model card (e.g. 'apache-2.0').",
    ),
    developers: str | None = typer.Option(
        None,
        help="'Developed by' credit for the model card.",
    ),
    card_authors: str | None = typer.Option(
        None,
        help="'Model Card Authors' credit for the model card.",
    ),
    card_contact: list[str] = typer.Option(  # noqa: B008
        [],
        help=(
            "'Model Card Contact' entry, e.g. 'Name - name@example.com'. "
            "Repeat the option for additional contacts, one per rendered line."
        ),
    ),
    preserve_all_tensors: bool = typer.Option(
        True,
        help=(
            "Load the base model's declared architecture so all of its tensors "
            "(e.g. a multimodal vision tower) survive into the merged output."
        ),
    ),
) -> None:
    """Merge ``adapter`` into ``base`` and save the result to ``output``.

    :args:
        base: Path to the base model directory.
        adapter: Path to the DPO PEFT adapter checkpoint.
        output: Destination directory for the merged model (defaulted beside
            the adapter when omitted).
        base_model_id: Hub id of the original base model for the model card.
        license_id: SPDX license identifier for the model card.
        developers: 'Developed by' credit for the model card.
        card_authors: 'Model Card Authors' credit for the model card.
        card_contact: 'Model Card Contact' entries (repeatable).
        preserve_all_tensors: Keep every base-model tensor in the output.
    """
    if output is None:
        output = adapter.parent / DEFAULT_DPO_MERGED_DIRNAME

    model_class = resolve_model_class(
        str(base),
        preserve_all_tensors=preserve_all_tensors,
    )
    base_model = model_class.from_pretrained(str(base))
    dpo_model = PeftModel.from_pretrained(base_model, str(adapter))
    merged = dpo_model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))

    context = DPOCardContext(
        base_model=base_model_id or str(base),
        sft_model_path=str(base),
        train_data=(
            "DPO tool-selection preference pairs; see the training run "
            "summary in the training output root for dataset details."
        ),
        license_id=license_id,
        developers=developers,
        card_authors=card_authors,
        card_contacts=tuple(card_contact),
    )
    write_dpo_model_card(
        output,
        context,
        read_checkpoint_stats(adapter),
        library_name="transformers",
        artifact_note=(
            f"This directory is the standalone merged DPO model produced from "
            f"the adapter checkpoint at `{adapter}`."
        ),
    )
    typer.echo(f"Merged DPO model and model card written to {output}")


if __name__ == "__main__":
    app()
