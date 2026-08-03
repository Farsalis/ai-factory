"""Upload a local model artifact to a Hugging Face Hub repository.

The token is read from the ``HF_TOKEN`` environment variable; it is never
read from the source tree. All other parameters are passed on the command
line.

Example:
    >>> # bash
    >>> # export HF_TOKEN="<YOUR_HF_TOKEN>"
    >>> python -m src.helper_scripts.upload.upload_to_hf \\
    ...     --file ./models/MyModel-q4_K_M.gguf \\
    ...     --repo Pragmanic0/MyModel \\
    ...     --path-in-repo MyModel-q4_K_M.gguf
"""

import os
from pathlib import Path

import typer
from huggingface_hub import upload_file

app = typer.Typer(add_completion=False)


@app.command()
def main(
    file: Path = typer.Option(  # noqa: B008
        ..., exists=True, readable=True, help="Local file to upload."
    ),
    repo: str = typer.Option(..., help="Target repo id, e.g. 'user/model-name'."),
    path_in_repo: str = typer.Option(..., help="Destination path inside the repo."),
    repo_type: str = typer.Option("model", help="'model', 'dataset', or 'space'."),
) -> None:
    """Upload ``file`` to ``repo`` at ``path_in_repo`` using ``HF_TOKEN``.

    :args:
        file: Local artifact path; must exist and be readable.
        repo: Hugging Face Hub repository id.
        path_in_repo: Destination path within the repository.
        repo_type: Repository kind passed to the Hub API.

    :raises:
        RuntimeError: If the ``HF_TOKEN`` environment variable is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Export it with a token that has write access to the target repo."
        )

    upload_file(
        path_or_fileobj=str(file),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type=repo_type,
        token=token,
    )


if __name__ == "__main__":
    app()
