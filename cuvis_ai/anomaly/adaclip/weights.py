"""Weight download manager for AdaCLIP pretrained models.

Downloads and caches AdaCLIP pretrained weights from Google Drive.
Weights are stored in ~/.cache/cuvis_ai/adaclip/ by default.

Reference weights from: https://github.com/caoyunkang/AdaCLIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gdown
from loguru import logger

# Google Drive file IDs extracted from the AdaCLIP README
# https://github.com/caoyunkang/AdaCLIP?tab=readme-ov-file
_GDRIVE_FILE_IDS = {
    "pretrained_mvtec_colondb": "1xVXANHGuJBRx59rqPRir7iqbkYzq45W0",
    "pretrained_visa_clinicdb": "1QGmPB0ByPZQ7FucvGODMSz7r5Ke5wx9W",
    "pretrained_all": "1Cgkfx3GAaSYnXPLolx-P7pFqYV0IVzZF",
}

ADACLIP_WEIGHTS: dict[str, dict[str, Any]] = {
    "pretrained_mvtec_colondb": {
        "gdrive_id": _GDRIVE_FILE_IDS["pretrained_mvtec_colondb"],
        "description": "Trained on MVTec AD & ColonDB datasets",
        "filename": "pretrained_mvtec_colondb.pth",
    },
    "pretrained_visa_clinicdb": {
        "gdrive_id": _GDRIVE_FILE_IDS["pretrained_visa_clinicdb"],
        "description": "Trained on VisA & ClinicDB datasets",
        "filename": "pretrained_visa_clinicdb.pth",
    },
    "pretrained_all": {
        "gdrive_id": _GDRIVE_FILE_IDS["pretrained_all"],
        "description": "Trained on all datasets (MVTec, VisA, ColonDB, ClinicDB, etc.)",
        "filename": "pretrained_all.pth",
    },
}


def get_weights_dir() -> Path:
    """Get or create the weights cache directory.

    Returns
    -------
    Path
        Path to ~/.cache/cuvis_ai/adaclip/
    """
    cache_dir = Path.home() / ".cache" / "cuvis_ai" / "adaclip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def list_available_weights() -> list[str]:
    """List available pretrained weight names.

    Returns
    -------
    list[str]
        Names of available pretrained weights.
    """
    return list(ADACLIP_WEIGHTS.keys())


def download_weights(weight_name: str, force: bool = False) -> Path:
    """Download AdaCLIP weights if not cached.

    Parameters
    ----------
    weight_name : str
        Name of the pretrained weights. One of:
        - "pretrained_mvtec_colondb": MVTec AD & ColonDB
        - "pretrained_visa_clinicdb": VisA & ClinicDB
        - "pretrained_all": All datasets combined
    force : bool, default False
        If True, re-download even if cached.

    Returns
    -------
    Path
        Path to the downloaded weights file.

    Raises
    ------
    ValueError
        If weight_name is not recognized.
    RuntimeError
        If download fails.
    """
    if weight_name not in ADACLIP_WEIGHTS:
        available = list_available_weights()
        raise ValueError(f"Unknown weight: {weight_name}. Available: {available}")

    cfg = ADACLIP_WEIGHTS[weight_name]
    cache_dir = get_weights_dir()
    target_path = cache_dir / cfg["filename"]

    if target_path.exists():
        if force:
            target_path.unlink()
        elif _looks_like_html(target_path):
            logger.warning("Cached AdaCLIP weights look invalid (HTML). Re-downloading...")
            target_path.unlink()
        else:
            logger.info(f"Using cached AdaCLIP weights: {target_path}")
            return target_path

    logger.info(f"Downloading AdaCLIP weights: {weight_name}...")
    logger.info(f"Description: {cfg['description']}")

    gdrive_id = cfg["gdrive_id"]
    _download_from_gdrive(gdrive_id, target_path)

    if not target_path.exists():
        raise RuntimeError(f"Download failed for {weight_name}")

    logger.info(f"Successfully downloaded to: {target_path}")
    return target_path


def _download_from_gdrive(file_id: str, target_path: Path) -> None:
    """Download a file from Google Drive.

    For large files, Google Drive shows a confirmation page. This function
    handles both direct downloads and confirmation-required downloads.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    target_path : Path
        Local path to save the file.
    """
    # Ensure parent exists before download
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download(id=file_id, output=str(target_path), quiet=False)
    except Exception as e:  # pragma: no cover - network dependent
        if target_path.exists():
            target_path.unlink()
        raise RuntimeError(f"Failed to download AdaCLIP weights from Google Drive: {e}") from e

    if _looks_like_html(target_path):
        target_path.unlink(missing_ok=True)
        raise RuntimeError("Google Drive returned an HTML confirmation page instead of weights.")


def _looks_like_html(path: Path, bytes_to_check: int = 256) -> bool:
    """Simple heuristic to determine if a downloaded file is HTML instead of binary weights."""
    if not path.exists() or path.stat().st_size == 0:
        return True

    try:
        with path.open("rb") as f:
            head = f.read(bytes_to_check)
    except OSError:
        return True

    lowered = head.lower()
    return lowered.startswith(b"<!doctype html") or b"<html" in lowered


def get_local_weight_path(weight_name: str) -> Path | None:
    """Get path to locally cached weights if they exist.

    Parameters
    ----------
    weight_name : str
        Name of the pretrained weights.

    Returns
    -------
    Path | None
        Path to cached weights, or None if not cached.
    """
    if weight_name not in ADACLIP_WEIGHTS:
        return None

    cfg = ADACLIP_WEIGHTS[weight_name]
    cache_dir = get_weights_dir()
    target_path = cache_dir / cfg["filename"]

    return target_path if target_path.exists() else None


__all__ = [
    "ADACLIP_WEIGHTS",
    "download_weights",
    "get_local_weight_path",
    "get_weights_dir",
    "list_available_weights",
]
