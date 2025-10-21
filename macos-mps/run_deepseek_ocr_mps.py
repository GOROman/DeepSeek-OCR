"""
DeepSeek-OCR Inference Script for MacOS MPS
This script provides a simplified interface for running DeepSeek-OCR on Apple Silicon.
"""

import torch
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')


def check_mps_availability():
    """Check if MPS is available and provide helpful information."""
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("ERROR: MPS not available because PyTorch was not built with MPS enabled.")
            print("Please install PyTorch with MPS support:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        else:
            print("ERROR: MPS not available because the current MacOS version does not support it.")
            print("MPS requires MacOS 12.3+")
        return False
    return True


def setup_device():
    """Set up the appropriate device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"⚠ Using CPU (this will be slow)")

    return device


def load_model_transformers(model_name='deepseek-ai/DeepSeek-OCR', device='mps'):
    """
    Load DeepSeek-OCR using transformers library.

    Note: This requires the model files to have MPS-compatible implementations.
    If this fails with flash_attn errors, you'll need to modify the model files.
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading model: {model_name}")
        print("Note: First run will download the model (~3GB)")

        # Try to load without flash attention
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model with explicit device mapping
        print(f"Loading model to {device}...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Don't use auto device map
        )

        model = model.eval().to(device)
        print("✓ Model loaded successfully")

        return model, tokenizer

    except ImportError as e:
        if 'flash_attn' in str(e):
            print("\n" + "="*60)
            print("ERROR: The model requires flash_attn which is not MPS compatible")
            print("="*60)
            print("\nSolution: You need to modify the model's code to remove flash_attn dependency.")
            print("\nSteps:")
            print("1. Download the model from Hugging Face")
            print("2. Modify modeling_deepseekocr.py to remove flash_attn imports")
            print("3. Replace flash_attn calls with torch.nn.functional.scaled_dot_product_attention")
            print("4. Load from local directory")
            print("\nOr use the custom MPS-compatible implementation in this directory.")
        raise


def run_inference(
    model,
    tokenizer,
    image_path,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    output_path="./output",
    base_size=1024,
    image_size=640,
    crop_mode=True
):
    """Run OCR inference on an image."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_path, exist_ok=True)

    print(f"\nRunning inference on: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Output directory: {output_path}")

    # Run inference
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_path,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=True,
        test_compress=True
    )

    print("\n✓ Inference completed")
    return result


def main():
    """Main function with example usage."""

    print("="*60)
    print("DeepSeek-OCR for MacOS MPS")
    print("="*60)

    # Check MPS availability
    if not check_mps_availability():
        print("\nFalling back to CPU...")

    # Setup device
    device = setup_device()

    # Example configuration
    MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
    IMAGE_PATH = 'your_image.jpg'  # Change this to your image
    OUTPUT_PATH = './output'

    # Load model
    try:
        print("\n" + "="*60)
        print("Loading Model")
        print("="*60)

        model, tokenizer = load_model_transformers(MODEL_NAME, device)

        # Example prompts
        PROMPTS = {
            'document': "<image>\n<|grounding|>Convert the document to markdown.",
            'ocr': "<image>\n<|grounding|>OCR this image.",
            'free_ocr': "<image>\nFree OCR.",
            'figure': "<image>\nParse the figure.",
            'describe': "<image>\nDescribe this image in detail.",
        }

        # Check if image exists
        if not os.path.exists(IMAGE_PATH):
            print(f"\n⚠ Please update IMAGE_PATH variable with your image file path")
            print(f"   Current path: {IMAGE_PATH}")
            print("\nExample usage:")
            print("  python run_deepseek_ocr_mps.py")
            print("\nThen edit this file and set:")
            print("  IMAGE_PATH = 'path/to/your/image.jpg'")
            return

        # Run inference
        result = run_inference(
            model,
            tokenizer,
            IMAGE_PATH,
            prompt=PROMPTS['document'],
            output_path=OUTPUT_PATH,
            base_size=1024,
            image_size=640,
            crop_mode=True
        )

        print("\n" + "="*60)
        print("Result")
        print("="*60)
        print(result)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see flash_attn errors, the model needs to be modified for MPS.")
        print("See the error message above for instructions.")
        raise


if __name__ == "__main__":
    main()
