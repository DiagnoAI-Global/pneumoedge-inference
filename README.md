# PneumoEdge Inference

Edge-deployable pneumonia detection inference system.
Part of the DiagnoAI Global diagnostic platform.

Runs on Raspberry Pi 4B and equivalent ARM devices.
Works fully offline. No internet connection required.

## System Architecture
```
X-ray Image
     │
Preprocessing (model-specific)
     │
Quantised TFLite Model
     │
Prediction + Confidence Score
     │
Optimised Grad-CAM
     │
Explainable Diagnosis
     │
Clinician Interface
```

## Models

| Model | Population | Input Size | Accuracy | AUROC |
|-------|-----------|-----------|----------|-------|
| Xception (quantised) | Paediatric (children) | 299×299 | 97.12% | 0.9934 |
| EfficientNetB4 (quantised) | Adult | 224×224 | 80.94% | 0.856 |

## Performance (Raspberry Pi 4B simulation)

| Model | Size | Inference | Grad-CAM | Total pipeline |
|-------|------|-----------|----------|----------------|
| EfficientNetB4 (quantised) | 18.59 MB | ~179–184 ms | ~270 ms | ~450 ms |
| Xception (quantised) | 21.61 MB | ~511–533 ms | ~570 ms | ~1080 ms |

Timing measured on Colab CPU configured to simulate Raspberry Pi 4B
(1.5 GHz clock, 4 threads). Actual hardware validation pending.
Both pipelines are clinically usable for point-of-care screening.

## Requirements
```
tensorflow>=2.10.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
matplotlib>=3.5.0
psutil>=5.9.0
```

Install: `pip install -r requirements.txt`

## Usage
```python
from src.inference.tflite_inference import PneumoEdgeInference

# For adult populations (EfficientNetB4)
engine = PneumoEdgeInference(
    model_path="path/to/effnetb4_quantized.tflite",
    model_type="efficientnetb4"
)

# For paediatric populations (Xception)
engine = PneumoEdgeInference(
    model_path="path/to/xception_quantized.tflite",
    model_type="xception"
)

result = engine.predict("chest_xray.jpg")
print(result)
```

## Repository Structure
```
src/
  inference/         — TFLite inference engine
  preprocessing/     — Model-specific image preprocessing
  explainability/    — Optimised Grad-CAM implementation
demo/
  demo_pipeline.py   — Command-line demo
  gradio_dashboard.py — Interactive web demo
benchmarks/
  edge_performance_results.md — Full timing results
```

## Citation

If you use this system in research, please cite:
```
@mastersthesis{konadu2025pneumonia,
  title={AI-Powered Pneumonia Detection in Low Resource Settings},
  author={Konadu, Evans},
  school={Kingston University},
  year={2025},
  supervisor={Makris, Dimitrios}
}
```

## About

Built by Evans Konadu | DiagnoAI Global
Contact: kofikonadu001@gmail.com
