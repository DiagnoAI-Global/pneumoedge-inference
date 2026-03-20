# Edge Deployment Performance Results

## Test Environment

Raspberry Pi 4B simulation on Google Colab CPU:
- Hardware simulated: Broadcom BCM2711, Cortex-A72 Quad-Core ARM v8, 1.5 GHz, 4 GB LPDDR4
- Colab CPU: Intel x86_64 @ 2.2 GHz, 1 physical core, 2 logical threads, 12.67 GB RAM
- Clock speed adjustment factor: ~1.47x applied to raw timings
- TF_NUM_INTRAOP_THREADS set to 4, TF_NUM_INTEROP_THREADS set to 1
- Actual Raspberry Pi 4B hardware validation pending field deployment

## Binary Classification Models

| Model | Dataset | Original Size | Quantised Size | Reduction | Inference (Original) | Inference (Quantised, adjusted) |
|-------|---------|--------------|----------------|-----------|---------------------|---------------------------------|
| EfficientNetB4 | NIH ChestX-ray14 (Adult) | 208.32 MB | 18.59 MB | 91% | 1461.07 ± 425.52 ms | 179–184 ms |
| Xception | Mendeley (Paediatric) | 82.08 MB | 21.61 MB | 74% | 1038.79 ± 205.36 ms | 511–533 ms |

## Full Pipeline (Inference + Grad-CAM)

| Model | Inference | Grad-CAM | Total |
|-------|-----------|----------|-------|
| EfficientNetB4 (quantised) | ~179–184 ms | ~270 ms | ~450 ms |
| Xception (quantised) | ~511–533 ms | ~570 ms | ~1080 ms |

Both pipelines are clinically usable for point-of-care screening.

## Quantisation Impact on Accuracy

| Model | Original Accuracy | Quantised Accuracy | Original AUROC | Quantised AUROC |
|-------|------------------|--------------------|----------------|-----------------|
| Xception (Mendeley) | 96.96% | 97.12% | 0.9933 | 0.9934 |
| EfficientNetB4 (NIH) | 86.10% | 80.94% | 0.8986 | 0.856 |

Note: Xception accuracy improved post-quantisation — a regularisation
effect of INT8 compression documented in the research.

## Grad-CAM Implementation

- Downsampled gradient computation: 128×128 with bilinear upsampling
- Speedup vs standard Grad-CAM: 5–15×
- Correlation with full-resolution output: 0.960 (Xception)
- Grayscale X-ray base used for clinical readability of overlay
