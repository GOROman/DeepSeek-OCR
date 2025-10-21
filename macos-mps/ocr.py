#!/usr/bin/env python3
"""
DeepSeek-OCR CLI Tool for MacOS
Simple command-line interface for OCR processing

Usage:
    python ocr.py image.jpg                          # Basic OCR
    python ocr.py --markdown document.pdf            # Convert to Markdown
    python ocr.py --batch images/*.jpg               # Process multiple files
    python ocr.py --figure chart.png                 # Parse figures
"""

import argparse
import sys
from pathlib import Path
import torch
from typing import Optional, List


class DeepSeekOCR:
    """Wrapper class for DeepSeek-OCR."""

    def __init__(self, device: Optional[str] = None):
        """Initialize the OCR model."""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading DeepSeek-OCR model (device: {self.device})...")

        try:
            from transformers import AutoModel, AutoTokenizer

            model_name = 'deepseek-ai/DeepSeek-OCR'

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).eval().to(self.device)

            print("✓ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def process(
        self,
        image_path: Path,
        prompt: str,
        output_dir: Path,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> str:
        """Process a single image."""

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {image_path.name}")

        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(output_dir / image_path.stem),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True
        )

        # Save as text file
        output_file = output_dir / f"{image_path.stem}.txt"
        output_file.write_text(result, encoding='utf-8')

        print(f"✓ Saved to: {output_file}")

        return result


def get_prompt(args) -> str:
    """Get the appropriate prompt based on arguments."""

    if args.prompt:
        return args.prompt

    if args.markdown:
        return "<image>\n<|grounding|>Convert the document to markdown."
    elif args.figure:
        return "<image>\nParse the figure."
    elif args.describe:
        return "<image>\nDescribe this image in detail."
    elif args.free:
        return "<image>\nFree OCR."
    else:
        # Default: grounded OCR
        return "<image>\n<|grounding|>OCR this image."


def process_single_file(args, ocr: DeepSeekOCR, image_path: Path) -> bool:
    """Process a single file."""

    try:
        prompt = get_prompt(args)

        # Determine resolution settings
        if args.tiny:
            base_size, image_size, crop_mode = 512, 512, False
        elif args.small:
            base_size, image_size, crop_mode = 640, 640, False
        elif args.large:
            base_size, image_size, crop_mode = 1280, 1280, False
        else:
            # Default: Base/Gundam mode
            base_size, image_size, crop_mode = 1024, 640, True

        result = ocr.process(
            image_path=image_path,
            prompt=prompt,
            output_dir=Path(args.output),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode
        )

        if args.verbose:
            print("\n" + "="*60)
            print("RESULT")
            print("="*60)
            print(result)
            print("="*60)

        return True

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return False


def main():
    """Main CLI function."""

    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR CLI Tool for MacOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic OCR
  python ocr.py image.jpg

  # Convert document to Markdown
  python ocr.py --markdown document.pdf

  # Process multiple files
  python ocr.py --batch images/*.jpg

  # Parse a figure/chart
  python ocr.py --figure chart.png

  # Use custom prompt
  python ocr.py --prompt "Extract all text" image.jpg

  # High-quality mode
  python ocr.py --large document.jpg

  # Fast mode
  python ocr.py --tiny screenshot.png
        """
    )

    # Input
    parser.add_argument('input', nargs='*', help='Input image file(s)')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple files')

    # Prompt selection (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('--markdown', action='store_true',
                              help='Convert document to Markdown')
    prompt_group.add_argument('--figure', action='store_true',
                              help='Parse figure/chart')
    prompt_group.add_argument('--describe', action='store_true',
                              help='Describe image in detail')
    prompt_group.add_argument('--free', action='store_true',
                              help='Free OCR (no layout)')
    prompt_group.add_argument('--prompt', type=str,
                              help='Custom prompt')

    # Resolution modes (mutually exclusive)
    resolution_group = parser.add_mutually_exclusive_group()
    resolution_group.add_argument('--tiny', action='store_true',
                                  help='Tiny mode (512x512, fastest)')
    resolution_group.add_argument('--small', action='store_true',
                                  help='Small mode (640x640)')
    resolution_group.add_argument('--large', action='store_true',
                                  help='Large mode (1280x1280, highest quality)')

    # Output
    parser.add_argument('-o', '--output', default='./output',
                        help='Output directory (default: ./output)')

    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print OCR results to console')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (disable MPS)')

    args = parser.parse_args()

    # Validate input
    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Collect input files
    input_files: List[Path] = []
    for pattern in args.input:
        path = Path(pattern)
        if path.is_file():
            input_files.append(path)
        elif '*' in pattern or args.batch:
            # Handle glob patterns
            from glob import glob
            input_files.extend([Path(f) for f in glob(pattern) if Path(f).is_file()])
        else:
            print(f"⚠ File not found: {pattern}")

    if not input_files:
        print("❌ No valid input files found")
        sys.exit(1)

    print(f"Found {len(input_files)} file(s) to process")

    # Initialize OCR
    device = "cpu" if args.cpu else None
    ocr = DeepSeekOCR(device=device)

    # Process files
    success_count = 0
    for image_path in input_files:
        if process_single_file(args, ocr, image_path):
            success_count += 1

    # Summary
    print("\n" + "="*60)
    print(f"✓ Processed {success_count}/{len(input_files)} files successfully")
    print(f"✓ Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
