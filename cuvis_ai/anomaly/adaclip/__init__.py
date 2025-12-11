"""AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection.

This module provides a port-based implementation of AdaCLIP for the cuvis.ai framework,
enabling zero-shot anomaly detection on hyperspectral data via RGB composition.

Reference:
    Cao, Y., et al. "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for
    Zero-Shot Anomaly Detection." ECCV 2024.
    https://github.com/caoyunkang/AdaCLIP

Example
-------
>>> from cuvis_ai.anomaly.adaclip import AdaCLIPModel, download_weights
>>> weight_path = download_weights("pretrained_all")
>>> model = AdaCLIPModel(backbone="ViT-L-14-336")
>>> model.load_weights(weight_path)
>>> model.eval()
>>> anomaly_map, score = model.predict(image_tensor, prompt="candle")
"""

from cuvis_ai.anomaly.adaclip.model import (
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    AdaCLIPModel,
    create_adaclip_model,
)
from cuvis_ai.anomaly.adaclip.weights import (
    ADACLIP_WEIGHTS,
    download_weights,
    get_weights_dir,
    list_available_weights,
)

__all__ = [
    "ADACLIP_WEIGHTS",
    "AdaCLIPModel",
    "OPENAI_DATASET_MEAN",
    "OPENAI_DATASET_STD",
    "create_adaclip_model",
    "download_weights",
    "get_weights_dir",
    "list_available_weights",
]
