!pip install -q gradio opencv-python

import gradio as gr
import os
import time
import psutil
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as xception_preprocess

# System Configuration & Pi4 Simulation
def get_pi4_simulation_setup():
    PI_CORES = 4
    PI_CPU_GHZ = 1.5

    os.environ['TF_NUM_INTRAOP_THREADS'] = str(PI_CORES)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'

    try:
        current_ghz = psutil.cpu_freq().current / 1000
        clock_factor = current_ghz / PI_CPU_GHZ
    except:
        clock_factor = 1.5

    return clock_factor

CLOCK_FACTOR = get_pi4_simulation_setup()

# Model Configuration
MODEL_CONFIGS = {
    'efficientnet': {
        'name': 'EfficientNet B4',
        'tflite_path': '/content/drive/MyDrive/NIH_models/Quantized/EffNetB4_quantized_hybrid.tflite',
        'h5_path': '/content/drive/MyDrive/NIH_models/Results/EffNetB4_epoch_15.h5',
        'input_size': (224, 224),
        'gradcam_size': (128, 128),
        'preprocess': efficientnet_preprocess
    },
    'xception': {
        'name': 'Xception',
        'tflite_path': '/content/drive/MyDrive/chest_xray/PTQ_Quantized_Ensemble/base_model_3_int8.tflite',
        'h5_path': '/content/drive/MyDrive/chest_xray/base_model_3.keras',
        'input_size': (299, 299),
        'gradcam_size': (128, 128),
        'preprocess': xception_preprocess
    }
}

# Original OptimisedClinicalGradCAM Class
class OptimisedClinicalGradCAM:
    """
    Minimal, focused Grad-CAM helper for clinical X-ray analysis:
     - Loads H5 model for Grad-CAM computation and classification
     - Compiles optimised tf.function Grad-CAM path with downsampling
     - Provides efficient explanation generation for deployment scenarios
    """

    def __init__(self, h5_path, gradcam_size=(128, 128), target_size=(224, 224), clock_factor=1.0):
        self.clock_factor = clock_factor
        self.h5_model = load_model(h5_path)
        self.target_size = target_size
        self.gradcam_size = gradcam_size

        # Identify last convolutional layer for gradient computation
        self.last_conv_layer = self._find_last_conv_layer()
        self.grad_model = Model(
            inputs=self.h5_model.inputs,
            outputs=[self.h5_model.get_layer(self.last_conv_layer).output, self.h5_model.output]
        )

        # Determine model output type for proper score extraction
        out_shape = self.h5_model.output_shape
        self.model_type = "softmax" if len(out_shape) == 2 and out_shape[1] == 2 else "sigmoid"

        # Compile optimised gradcam function
        self._compile_gradcam_function()

    def _find_last_conv_layer(self):
        for layer in reversed(self.h5_model.layers):
            if 'conv' in layer.name.lower() and len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("No suitable convolutional layer found for Grad-CAM")

    def _compile_gradcam_function(self):
        """Compile efficient Grad-CAM routine with tf.function optimisation."""
        @tf.function
        def _gradcam_compiled(input_tensor):
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                conv_output, preds = self.grad_model(input_tensor)
                if self.model_type == "softmax":
                    score = preds[:, 1]    # pneumonia class
                else:
                    score = preds[:, 0]

            grads = tape.gradient(score, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_output = conv_output[0]
            heatmap = tf.tensordot(conv_output, pooled_grads, axes=[[2], [0]])
            heatmap = tf.nn.relu(heatmap)
            heatmap = tf.image.resize(heatmap[..., tf.newaxis], self.target_size)[..., 0]
            return heatmap, preds

        self._gradcam_compiled = _gradcam_compiled

        # Warm-up compilation
        try:
            dummy = tf.zeros((1, *self.gradcam_size, 3), dtype=tf.float32)
            _ = self._gradcam_compiled(dummy)
        except Exception:
            dummy = tf.zeros((1, *self.target_size, 3), dtype=tf.float32)
            _ = self._gradcam_compiled(dummy)

    def prepare_image_from_pil(self, pil_image):
        """Prepare PIL image for Grad-CAM analysis"""
        img = pil_image.convert('RGB').resize(self.target_size)
        img_arr = np.array(img, dtype=np.float32)
        # Use appropriate preprocessing based on model
        if self.target_size == (224, 224):
            img_pre = efficientnet_preprocess(img_arr.copy())
        else:
            img_pre = xception_preprocess(img_arr.copy())
        img_keras = np.expand_dims(img_pre, 0).astype(np.float32)
        img_display = np.array(img, dtype=np.uint8)
        return img_keras, img_display

    def generate_optimised_gradcam(self, img_keras):
        """Generate Grad-CAM using downsampled input for efficiency."""
        img_small = tf.image.resize(img_keras, self.gradcam_size, method='bilinear')
        start = time.perf_counter()
        try:
            heatmap_tf, preds = self._gradcam_compiled(img_small)
        except Exception:
            heatmap_tf, preds = self._gradcam_compiled(img_keras)
        elapsed = (time.perf_counter() - start) * self.clock_factor

        heatmap = heatmap_tf.numpy()

        # Normalise heatmap values
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap, preds[0], elapsed

    def create_overlay(self, original_img, heatmap, alpha=0.4):
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype('uint8'), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(heatmap_colored, alpha, gray_rgb, 1 - alpha, 0)
        return overlay

# TFLite Inference Functions
def load_tflite_model(tflite_path, input_size):
    """Load and configure TFLite interpreter"""
    interpreter = tf.lite.Interpreter(tflite_path)
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]['index'], [1, *input_size, 3])
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def prepare_image_tflite(pil_image, input_size, preprocess_fn):
    """Prepare PIL image for TFLite inference"""
    img = pil_image.convert('RGB').resize(input_size)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_fn(arr.copy())
    return np.expand_dims(arr, 0)

def run_tflite_inference(interpreter, input_details, output_details, input_data):
    """Run TFLite inference with timing"""
    # Warm-up run
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Timed inference
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    duration_ms = (time.perf_counter() - start) * 1000 * CLOCK_FACTOR

    output = interpreter.get_tensor(output_details[0]['index'])
    return duration_ms, output

# Global variables for model caching
tflite_cache = {}
gradcam_cache = {}

#  Main Inference Function
def run_inference(model_name, image, custom_h5, enable_gradcam):
    """Main inference function using your original implementation"""

    if image is None:
        return None, None, "Please upload a chest X-ray image first", "", "", "", ""

    # Get model configuration
    model_key = None
    for key, config in MODEL_CONFIGS.items():
        if config['name'] == model_name:
            model_key = key
            break

    if not model_key:
        return None, None, "Invalid model selection", "", "", "", ""

    config = MODEL_CONFIGS[model_key]

    try:
        # Load TFLite model (with caching)
        cache_key = f"tflite_{model_key}"
        if cache_key not in tflite_cache:
            tflite_cache[cache_key] = load_tflite_model(config['tflite_path'], config['input_size'])
        interpreter, input_details, output_details = tflite_cache[cache_key]

        # Prepare image for TFLite
        input_data = prepare_image_tflite(image, config['input_size'], config['preprocess'])

        # Run TFLite inference
        inference_time, predictions = run_tflite_inference(interpreter, input_details, output_details, input_data)

        # Process predictions
        pred_array = np.squeeze(predictions)
        if model_key == 'efficientnet':
            # Single sigmoid output
            pneu_prob = float(pred_array)
            norm_prob = 1.0 - pneu_prob
        else:
            # Two-class softmax output [normal, pneumonia]
            norm_prob = float(pred_array[0])
            pneu_prob = float(pred_array[1])

        # Determine diagnosis
        diagnosis = "Pneumonia" if pneu_prob > 0.5 else "Normal"
        confidence = (pneu_prob if diagnosis == "Pneumonia" else norm_prob) * 100.0
        result_text = f"{diagnosis}\nConfidence: {confidence:.1f}%"

        # Generate Optimised Grad-CAM if enabled
        gradcam_overlay = None
        gradcam_time_info = ""

        if enable_gradcam:
            try:
                h5_path = custom_h5.name if custom_h5 else config['h5_path']

                # Load Grad-CAM model (with caching)
                gradcam_cache_key = f"gradcam_{model_key}_{h5_path}"
                if gradcam_cache_key not in gradcam_cache:
                    gradcam_cache[gradcam_cache_key] = OptimisedClinicalGradCAM(
                        h5_path=h5_path,
                        gradcam_size=config['gradcam_size'],
                        target_size=config['input_size'],
                        clock_factor=CLOCK_FACTOR
                    )
                gradcam = gradcam_cache[gradcam_cache_key]

                # Prepare image and generate Optimised Grad-CAM
                img_keras, img_display = gradcam.prepare_image_from_pil(image)
                heat_opt, preds_opt, gradcam_time = gradcam.generate_optimised_gradcam(img_keras)

                # Create overlay using your original method
                overlay = gradcam.create_overlay(img_display, heat_opt)
                gradcam_overlay = Image.fromarray(overlay)
                gradcam_time_info = f"{gradcam_time * 1000:.2f} ms"

            except Exception as e:
                gradcam_time_info = f"Grad-CAM Error: {str(e)}"

        # Performance summary
        perf_text = f"Inference: {inference_time:.2f} ms"
        if gradcam_time_info and not gradcam_time_info.startswith("Grad-CAM Error"):
            perf_text += f" | Optimised Grad-CAM: {gradcam_time_info}"

        # System info
        pi_info = f"Simulated Pi4 clock factor: {CLOCK_FACTOR:.2f}x"

        try:
            model_size = os.path.getsize(config['tflite_path']) / (1024 * 1024)
            model_info = f"{config['name']} TFLite: {model_size:.2f} MB"
        except:
            model_info = f"{config['name']} TFLite: (file not found)"

        # Prepare display images
        display_img = image.resize((250, int(250 * image.size[1] / image.size[0])), Image.LANCZOS)

        return (display_img, gradcam_overlay, result_text,
                f"{inference_time:.2f} ms", perf_text, pi_info, model_info)

    except Exception as e:
        return None, None, f"Error: {str(e)}", "", "", "", ""

# Gradio Interface
model_options = [config['name'] for config in MODEL_CONFIGS.values()]

css = """
.gradio-container { max-width: 1400px !important; }
.gr-button-primary {
    background: linear-gradient(45deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
"""

with gr.Blocks(css=css, title="Pneumonia Detection AI") as demo:

    # Header
    gr.HTML("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%); border-radius: 12px; margin-bottom: 25px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5em; font-weight: 700;'>
            AI-Powered Pneumonia Detection in Low-Resource Settings
        </h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.1em; margin: 8px 0 0 0;'>
            Live Demo
        </p>
    </div>
    """)

    with gr.Row():
        # Left Panel - Controls (smaller)
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("### Model Configuration")

            model_dropdown = gr.Dropdown(
                choices=model_options,
                value=model_options[0],
                label="Select AI Model"
            )

            gr.Markdown("### Upload Chest X-ray")
            image_input = gr.Image(
                label="",
                type="pil",
                height=200
            )

            gr.Markdown("### Advanced Options")
            gradcam_checkbox = gr.Checkbox(
                label="Generate Optimised Grad-CAM Visualization",
                value=True
            )

            custom_model_file = gr.File(
                label="Custom H5/Keras Model (Optional)",
                file_types=[".h5", ".keras", ".hdf5"]
            )

            # Inference Button
            inference_btn = gr.Button(
                "Run Inference",
                variant="primary",
                size="lg"
            )

        # Right Panel - Results (larger)
        with gr.Column(scale=2):
            gr.Markdown("### Analysis Results")

            with gr.Row():
                # Small input display
                input_display = gr.Image(
                    label="Input Image",
                    interactive=False,
                    height=250
                )
                # Large Grad-CAM display
                gradcam_display = gr.Image(
                    label="Optimised Grad-CAM Overlay",
                    interactive=False,
                    height=400
                )

            with gr.Row():
                with gr.Column(scale=2):
                    diagnosis_output = gr.Textbox(
                        label="Diagnosis Result",
                        lines=2,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    timing_output = gr.Textbox(
                        label="Inference Time",
                        interactive=False
                    )

            with gr.Row():
                performance_output = gr.Textbox(
                    label="Performance Summary",
                    interactive=False
                )
                system_output = gr.Textbox(
                    label="Device Simulation",
                    interactive=False
                )
                model_output = gr.Textbox(
                    label="Model Information",
                    interactive=False
                )

    # Connect the interface
    inference_btn.click(
        fn=run_inference,
        inputs=[model_dropdown, image_input, custom_model_file, gradcam_checkbox],
        outputs=[input_display, gradcam_display, diagnosis_output, timing_output,
                performance_output, system_output, model_output]
    )

    # Footer with technical details
    gr.HTML("""
    <div style='margin-top: 20px; padding: 20px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #2563eb;'>
        <h4 style='color: #1e40af; margin-top: 0;'>Technical Implementation</h4>
        <ul style='color: #4a5568; line-height: 1.6; margin-bottom: 0;'>
            <li><strong>Inference:</strong> TensorFlow Lite quantized models for edge deployment</li>
            <li><strong>Grad-CAM:</strong> OptimisedClinicalGradCAM with downsampling (128×128) for faster computation</li>
            <li><strong>Hardware:</strong> Simulated Raspberry Pi 4 environment (1.5GHz, 4-core ARM)</li>
            <li><strong>Performance:</strong> Sub-100ms total processing time optimized for low-resource settings</li>
        </ul>
    </div>
    """)

print("Loading OptimisedClinicalGradCAM system...")
print("Configured for Raspberry Pi 4 simulation with clock factor:", CLOCK_FACTOR)
demo.launch(share=True, inbrowser=True)
