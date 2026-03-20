"""
PneumoEdge Demo Pipeline
DiagnoAI Global

Full inference pipeline demonstration:
image → preprocessing → TFLite prediction → Grad-CAM → result

Usage:
    # Adult X-ray with EfficientNetB4:
    python demo/demo_pipeline.py --image xray.jpg \
        --model effnetb4.tflite --type efficientnetb4

    # Paediatric X-ray with Xception:
    python demo/demo_pipeline.py --image xray.jpg \
        --model xception.tflite --type xception
"""

import argparse
import time
from src.inference.tflite_inference import PneumoEdgeInference
from src.preprocessing.image_preprocessing import validate_xray


def run_demo(image_path: str, model_path: str, model_type: str):
    print("\n=== PneumoEdge Diagnostic Demo ===")
    print(f"Image:      {image_path}")
    print(f"Model:      {model_path}")
    print(f"Model type: {model_type}")
    print("==================================\n")

    if not validate_xray(image_path):
        print("Error: Could not read image file.")
        return

    engine = PneumoEdgeInference(model_path=model_path, model_type=model_type)

    start = time.time()
    result = engine.predict(image_path)
    total_time = (time.time() - start) * 1000

    print(f"Prediction:          {result['prediction']}")
    print(f"Confidence:          {result['confidence']}%")
    print(f"Pneumonia prob:      {result['pneumonia_probability']}%")
    print(f"Normal prob:         {result['normal_probability']}%")
    print(f"Population:          {result['population']}")
    print(f"Inference time:      {result['inference_time_ms']} ms")
    print(f"Total pipeline time: {round(total_time, 2)} ms")
    print("\nNote: Clinical decision support only.")
    print("All results must be reviewed by a qualified clinician.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PneumoEdge Demo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--type",
        required=True,
        choices=["xception", "efficientnetb4"],
        help="Model type"
    )
    args = parser.parse_args()
    run_demo(args.image, args.model, args.type)
