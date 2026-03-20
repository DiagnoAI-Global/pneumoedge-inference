"""
PneumoEdge TFLite Inference Engine
DiagnoAI Global

Runs quantised TFLite pneumonia detection models on edge devices.
Supports Xception (paediatric, 299x299) and EfficientNetB4 (adult, 224x224).
Uses model-specific preprocessing matching the original training pipeline.
"""

import numpy as np
import time
from PIL import Image


MODEL_CONFIGS = {
    "xception": {
        "input_size": 299,
        "preprocess": "xception",
        "output_type": "softmax",
        "population": "paediatric",
    },
    "efficientnetb4": {
        "input_size": 224,
        "preprocess": "efficientnet",
        "output_type": "sigmoid",
        "population": "adult",
    },
}


class PneumoEdgeInference:
    """
    Edge inference engine for quantised PneumoEdge TFLite models.

    Usage:
        # For paediatric (Xception):
        engine = PneumoEdgeInference(
            model_path="xception_quantized.tflite",
            model_type="xception"
        )

        # For adult (EfficientNetB4):
        engine = PneumoEdgeInference(
            model_path="effnetb4_quantized.tflite",
            model_type="efficientnetb4"
        )

        result = engine.predict("chest_xray.jpg")
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Args:
            model_path: Path to .tflite model file
            model_type: Either "xception" or "efficientnetb4"
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"model_type must be 'xception' or 'efficientnetb4', got '{model_type}'"
            )

        self.config = MODEL_CONFIGS[model_type]
        self.model_type = model_type
        self.input_size = self.config["input_size"]

        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.resize_tensor_input(
                self.interpreter.get_input_details()[0]["index"],
                [1, self.input_size, self.input_size, 3]
            )
            self.interpreter.allocate_tensors()
        except ImportError:
            raise ImportError(
                "TensorFlow not found. Install with: pip install tensorflow"
            )

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Warm-up run for stable timing
        dummy = np.zeros(
            (1, self.input_size, self.input_size, 3), dtype=np.float32
        )
        self.interpreter.set_tensor(self.input_details[0]["index"], dummy)
        self.interpreter.invoke()

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image using model-specific pipeline.
        """
        img = Image.open(image_path).convert("RGB").resize(
            (self.input_size, self.input_size)
        )
        img_array = np.array(img, dtype=np.float32)

        if self.config["preprocess"] == "xception":
            from tensorflow.keras.applications.xception import (
                preprocess_input,
            )
        else:
            from tensorflow.keras.applications.efficientnet import (
                preprocess_input,
            )

        img_array = preprocess_input(img_array.copy())
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: str) -> dict:
        """
        Run pneumonia detection on a chest X-ray.

        Returns:
            Dictionary with prediction, confidence, probabilities, and timing
        """
        preprocessed = self.preprocess(image_path)

        start_time = time.perf_counter()
        self.interpreter.set_tensor(
            self.input_details[0]["index"], preprocessed
        )
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start_time) * 1000

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Handle both sigmoid (1 output) and softmax (2 outputs)
        if self.config["output_type"] == "sigmoid" or output.shape[-1] == 1:
            pneumonia_prob = float(output[0][0])
            normal_prob = 1.0 - pneumonia_prob
        else:
            import tensorflow as tf
            probs = tf.nn.softmax(output[0]).numpy()
            normal_prob = float(probs[0])
            pneumonia_prob = float(probs[1])

        predicted_class = "Pneumonia" if pneumonia_prob > 0.5 else "Normal"
        confidence = (
            pneumonia_prob if predicted_class == "Pneumonia" else normal_prob
        )

        return {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "pneumonia_probability": round(pneumonia_prob * 100, 2),
            "normal_probability": round(normal_prob * 100, 2),
            "inference_time_ms": round(inference_time, 2),
            "model": self.model_type,
            "population": self.config["population"],
        }
