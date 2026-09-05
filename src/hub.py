"""Safe, opt-in Hugging Face checkpoint publishing helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from src.checkpoint import load_checkpoint

def verify_checkpoint_files(checkpoint_dir: str | Path) -> bool:
    """Check local checkpoint completeness before a network upload is attempted."""
    directory = Path(checkpoint_dir)
    return all((directory / name).is_file() for name in ("model.pt", "training_state.pt", "metadata.json"))

def upload_verified_checkpoint(checkpoint_dir: str | Path, repo_id: str = "OmilosAISolutions/omilos-legal-ai-10k") -> dict[str, Any]:
    """Upload only a complete checkpoint using standard HF authentication.

    Authentication is delegated to ``HF_TOKEN``/``huggingface-cli login``.  Errors
    are returned to the caller so the local training process can retain its safe
    checkpoint and decide whether to retry.
    """
    directory = Path(checkpoint_dir)
    if not verify_checkpoint_files(directory):
        return {"uploaded": False, "reason": "checkpoint is not locally verified"}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(directory), path_in_repo=directory.name)
    except Exception as exc:  # Network/auth errors must not destroy the local checkpoint.
        return {"uploaded": False, "reason": str(exc)}
    return {"uploaded": True, "repo_id": repo_id, "checkpoint": directory.name}
