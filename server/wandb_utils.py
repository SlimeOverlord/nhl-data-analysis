import os
import logging
from typing import Optional, Tuple

import joblib
from wandb import Api

logger = logging.getLogger(__name__)


def fetch_and_load_artifact(
    entity: str, project: str, artifact_name: str = "angle_model"
) -> Tuple[Optional[object], Optional[str]]:
    """Try to find `artifact_name` in the specified WandB project, download it,
    and attempt to load a `joblib` serialized `.pkl` file. Returns (model, downloaded_dir).
    Simplified: we only consider `.pkl` files and use `joblib.load`.
    """

    api = Api()
    artifact = None
    # Try fully-qualified artifact name first
    try:
        fq = f"{entity}/{project}/{artifact_name}:latest"
        artifact = api.artifact(fq)
        if artifact is None:
            artifact = api.artifact(f"{artifact_name}:latest")
    except Exception:
        artifact = None

    # If still not found, return immediately (simplified behavior)
    if artifact is None:
        logger.info("Artifact %s not found in %s/%s", artifact_name, entity, project)
        return None, None

    # Download artifact
    try:
        download_dir = artifact.download()
        logger.info("Downloaded artifact to %s", download_dir)
    except Exception as e:
        logger.exception("Failed to download artifact: %s", e)
        return None, None

    # Find candidate .pkl model file
    model_path = None
    for root, dirs, files in os.walk(download_dir):
        for f in files:
            if f.endswith(".pkl"):
                model_path = os.path.join(root, f)
                break
        if model_path:
            break

    if not model_path:
        logger.info("No .pkl model file found in artifact %s", artifact_name)
        return None, download_dir

    # Load model with joblib
    try:
        model = joblib.load(model_path)
        logger.info("Loaded model via joblib from %s", model_path)
        return model, download_dir
    except Exception as e:
        logger.exception("Failed to load model via joblib: %s", e)
        return None, download_dir
