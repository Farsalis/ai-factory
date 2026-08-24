"""Merge a DPO LoRA checkpoint into its base model and save the result.

Loads the base (already-merged SFT) model and the DPO PEFT adapter from disk,
applies ``merge_and_unload`` to fold the adapter weights into the base, and
saves the merged model to the chosen output directory.

:example:
    >>> python -m src.helper_scripts.dpo_merge_base \\
    ...     --base ./training_output/final_merged_model \\
    ...     --adapter ./training_output/dpo_model/checkpoint-100 \\
    ...     --output ./training_output/dpo_merged_model
"""

from pathlib import Path

import typer
from peft import PeftModel

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
    output: Path = typer.Option(  # noqa: B008
        ..., file_okay=False, help="Where to write the merged model."
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
        output: Destination directory for the merged model.
        preserve_all_tensors: Keep every base-model tensor in the output.
    """
    model_class = resolve_model_class(
        str(base),
        preserve_all_tensors=preserve_all_tensors,
    )
    base_model = model_class.from_pretrained(str(base))
    dpo_model = PeftModel.from_pretrained(base_model, str(adapter))
    merged = dpo_model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))


if __name__ == "__main__":
    app()
