"""
Image Preprocessing Pipeline — PneumoEdge
DiagnoAI Global

Standardised chest X-ray preprocessing matching the training pipeline
used for each model. Uses Keras application-specific preprocess_input
rather than generic ImageNet normalisation.
"""

import numpy as np
from PIL import Image


def preprocess_efficientnetb4(image_path: str) -> np.ndarray:
    """
    Preprocess a chest X-ray for EfficientNetB4 inference.
    Input size: 224x224. Uses EfficientNet-specific preprocessing.

    Args:
        image_path: Path to chest X-ray (JPEG or PNG)

    Returns:
        Preprocessed array (1, 224, 224, 3), float32
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array.copy())
    return np.expand_dims(img_array, axis=0)


def preprocess_xception(image_path: str) -> np.ndarray:
    """
    Preprocess a chest X-ray for Xception inference.
    Input size: 299x299. Uses Xception-specific preprocessing.

    Args:
        image_path: Path to chest X-ray (JPEG or PNG)

    Returns:
        Preprocessed array (1, 299, 299, 3), float32
    """
    from tensorflow.keras.applications.xception import preprocess_input

    img = Image.open(image_path).convert("RGB").resize((299, 299))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array.copy())
    return np.expand_dims(img_array, axis=0)


def validate_xray(image_path: str) -> bool:
    """
    Basic validation that a file is a readable image.

    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False
