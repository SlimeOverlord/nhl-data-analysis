import os
import logging
from typing import Optional, Tuple
import joblib
from wandb import Api

logger = logging.getLogger(__name__)


def fetch_and_load_artifact(
    entity: str, project: str, artifact_name: str = "angle_model"
) -> Tuple[Optional[object], Optional[object]]:
    """Try to find `artifact_name` in the specified WandB project, download it,
    and attempt to load a `joblib` serialized `.pkl` file. Returns
    `(model, scaler_object)` where `scaler_object` is already loaded via
    `joblib.load` when available.

    The previous implementation returned the download directory as a second
    return value; that is unused by callers, so this helper now returns only
    the loaded artifacts (or `None` when missing).
    """

    # First, prefer a local artifact stored under `server/artifacts/`.
    # Directory names follow the `<artifact_name>:<version>` pattern (e.g. `angle_model:v2`).
    local_artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    download_dir = None
    if os.path.isdir(local_artifacts_dir):
        candidates = [
            d
            for d in os.listdir(local_artifacts_dir)
            if d.startswith(f"{artifact_name}:")
        ]
        if candidates:
            # Prefer highest numeric vN suffix when present (e.g. v2 > v1).
            def parse_version(s: str):
                # s is like 'angle_model:v2' -> return integer 2 when possible
                try:
                    tail = s.split(":", 1)[1]
                except Exception:
                    return (0, s)
                if tail.startswith("v") and tail[1:].isdigit():
                    return (int(tail[1:]), tail)
                # Fallback: use lexicographic order
                return (0, tail)

            candidates.sort(key=parse_version, reverse=True)
            selected = candidates[0]
            download_dir = os.path.join(local_artifacts_dir, selected)
            logger.info(
                "Using local artifact directory %s for artifact %s",
                download_dir,
                artifact_name,
            )

    # If local artifact not found, fall back to WandB API download
    if download_dir is None:
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
            logger.info(
                "Artifact %s not found in %s/%s", artifact_name, entity, project
            )
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
    scaler_path = None
    for root, dirs, files in os.walk(download_dir):
        for f in files:
            if f.endswith("_model.pkl"):
                model_path = os.path.join(root, f)
            if f.endswith("_scaler.pkl"):
                scaler_path = os.path.join(root, f)
        if model_path and scaler_path:
            break

    if not model_path:
        logger.info("No .pkl model file found in artifact %s", artifact_name)
        return None, None

    # Load model with joblib
    model = None
    try:
        model = joblib.load(model_path)
        logger.info("Loaded model via joblib from %s", model_path)
    except Exception as e:
        logger.exception("Failed to load model via joblib: %s", e)

    # Load scaler object if present
    scaler_obj = None
    if scaler_path:
        try:
            scaler_obj = joblib.load(scaler_path)
            logger.info("Loaded scaler via joblib from %s", scaler_path)
        except Exception as e:
            logger.exception("Failed to load scaler via joblib: %s", e)

    return model, scaler_obj
