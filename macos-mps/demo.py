#!/usr/bin/env python3
"""
Interactive Demo for DeepSeek-OCR on MacOS
Demonstrates various use cases with example images
"""

import sys
from pathlib import Path
import torch


def check_setup():
    """Check if setup is complete."""
    print("Checking setup...")

    try:
        import transformers
        if torch.backends.mps.is_available():
            print("✓ MPS is available")
            return True
        else:
            print("⚠ MPS not available, using CPU")
            return True
    except ImportError:
        print("❌ Dependencies not installed")
        print("\nPlease run setup first:")
        print("  python setup.py")
        return False


def demo_simple_ocr():
    """Demo: Simple OCR on an image."""
    print("\n" + "="*60)
    print("Demo 1: Simple OCR")
    print("="*60)

    print("""
This demo shows basic OCR extraction from an image.

Example command:
    python ocr.py image.jpg

This will:
1. Load the image
2. Extract all visible text
3. Save results to ./output/image.txt
    """)

    if input("\nWant to see the code? [y/N]: ").strip().lower() == 'y':
        print("""
```python
from ocr import DeepSeekOCR
from pathlib import Path

# Initialize OCR
ocr = DeepSeekOCR()

# Process image
result = ocr.process(
    image_path=Path('image.jpg'),
    prompt="<image>\\n<|grounding|>OCR this image.",
    output_dir=Path('./output')
)

print(result)
```
        """)


def demo_document_conversion():
    """Demo: Convert document to Markdown."""
    print("\n" + "="*60)
    print("Demo 2: Document to Markdown Conversion")
    print("="*60)

    print("""
This demo converts a document image or PDF to Markdown format.

Example command:
    python ocr.py --markdown document.pdf

This preserves:
- Headers and sections
- Tables
- Lists
- Formatting
    """)

    if input("\nWant to see the code? [y/N]: ").strip().lower() == 'y':
        print("""
```python
from ocr import DeepSeekOCR
from pathlib import Path

ocr = DeepSeekOCR()

result = ocr.process(
    image_path=Path('document.pdf'),
    prompt="<image>\\n<|grounding|>Convert the document to markdown.",
    output_dir=Path('./output'),
    base_size=1024,
    image_size=640,
    crop_mode=True
)

# Save as Markdown file
Path('output.md').write_text(result)
```
        """)


def demo_figure_parsing():
    """Demo: Parse figures and charts."""
    print("\n" + "="*60)
    print("Demo 3: Figure and Chart Parsing")
    print("="*60)

    print("""
This demo extracts information from figures, charts, and diagrams.

Example command:
    python ocr.py --figure chart.png

Useful for:
- Bar charts
- Line graphs
- Flowcharts
- Diagrams
    """)

    if input("\nWant to see the code? [y/N]: ").strip().lower() == 'y':
        print("""
```python
from ocr import DeepSeekOCR
from pathlib import Path

ocr = DeepSeekOCR()

result = ocr.process(
    image_path=Path('chart.png'),
    prompt="<image>\\nParse the figure.",
    output_dir=Path('./output')
)

# Extract chart data
print("Chart data:", result)
```
        """)


def demo_batch_processing():
    """Demo: Batch process multiple files."""
    print("\n" + "="*60)
    print("Demo 4: Batch Processing")
    print("="*60)

    print("""
This demo processes multiple images at once.

Example command:
    python ocr.py --batch images/*.jpg

Or programmatically:
    """)

    if input("\nWant to see the code? [y/N]: ").strip().lower() == 'y':
        print("""
```python
from ocr import DeepSeekOCR
from pathlib import Path

ocr = DeepSeekOCR()

# Get all images
image_dir = Path('images')
images = list(image_dir.glob('*.jpg'))

print(f"Processing {len(images)} images...")

# Process each image
for image_path in images:
    print(f"Processing: {image_path.name}")

    result = ocr.process(
        image_path=image_path,
        prompt="<image>\\n<|grounding|>OCR this image.",
        output_dir=Path('output')
    )

    print(f"✓ Done: {image_path.name}")

print("All images processed!")
```
        """)


def demo_resolution_modes():
    """Demo: Different resolution modes."""
    print("\n" + "="*60)
    print("Demo 5: Resolution Modes")
    print("="*60)

    print("""
Choose the right mode for your use case:

1. TINY (--tiny)
   - Size: 512×512
   - Speed: Fastest (~1.5s)
   - Use: Screenshots, simple text

2. SMALL (--small)
   - Size: 640×640
   - Speed: Fast (~2s)
   - Use: Small documents

3. BASE (default)
   - Size: 1024×1024
   - Speed: Normal (~3s)
   - Use: Standard documents

4. LARGE (--large)
   - Size: 1280×1280
   - Speed: Slow (~4.5s)
   - Use: High-quality scans

5. GUNDAM (default with crop)
   - Size: Dynamic
   - Speed: Variable (~6s)
   - Use: Large multi-page documents

Examples:
    python ocr.py --tiny screenshot.png      # Fast
    python ocr.py --large important_doc.jpg  # High quality
    """)


def demo_custom_prompts():
    """Demo: Using custom prompts."""
    print("\n" + "="*60)
    print("Demo 6: Custom Prompts")
    print("="*60)

    print("""
Create custom prompts for specific tasks:

Example prompts:
    """)

    prompts = {
        "Table extraction": "<image>\\n<|grounding|>Extract all tables from this document.",
        "Header extraction": "<image>\\n<|grounding|>Extract only the headers and titles.",
        "Footnotes": "<image>\\n<|grounding|>Extract all footnotes and references.",
        "Structured data": "<image>\\n<|grounding|>Extract data in a structured format.",
        "Translation": "<image>\\nExtract and translate the text to English.",
        "Summary": "<image>\\nProvide a summary of the document content.",
    }

    for task, prompt in prompts.items():
        print(f"\n{task}:")
        print(f'    python ocr.py --prompt "{prompt}" image.jpg')

    if input("\nWant to see example code? [y/N]: ").strip().lower() == 'y':
        print("""
```python
from ocr import DeepSeekOCR
from pathlib import Path

ocr = DeepSeekOCR()

# Custom prompt for table extraction
result = ocr.process(
    image_path=Path('document.jpg'),
    prompt="<image>\\n<|grounding|>Extract all tables in Markdown format.",
    output_dir=Path('./output')
)

print("Extracted tables:")
print(result)
```
        """)


def demo_integration_examples():
    """Demo: Integration with other tools."""
    print("\n" + "="*60)
    print("Demo 7: Integration Examples")
    print("="*60)

    print("""
Integrate DeepSeek-OCR with your workflow:

1. Web Scraping + OCR
2. Automated Document Processing
3. Email Attachment Processing
4. Cloud Storage Integration
    """)

    if input("\nWant to see examples? [y/N]: ").strip().lower() == 'y':
        print("""
Example 1: Process screenshots from clipboard

```python
from ocr import DeepSeekOCR
from pathlib import Path
import subprocess

# Get screenshot from clipboard (macOS)
subprocess.run(['screencapture', '-c'])
subprocess.run(['osascript', '-e',
    'set the clipboard to (read (POSIX file "/tmp/screenshot.png") as «class PNGf»)'])

# Save from clipboard
image_path = Path('/tmp/screenshot.png')

# Process
ocr = DeepSeekOCR()
result = ocr.process(
    image_path=image_path,
    prompt="<image>\\n<|grounding|>OCR this image.",
    output_dir=Path('./output')
)

print(result)
```

Example 2: Watch folder for new documents

```python
from ocr import DeepSeekOCR
from pathlib import Path
import time

ocr = DeepSeekOCR()
watch_dir = Path('~/Documents/OCR_Input').expanduser()
output_dir = Path('~/Documents/OCR_Output').expanduser()

processed = set()

print(f"Watching {watch_dir} for new files...")

while True:
    for file in watch_dir.glob('*.{jpg,png,pdf}'):
        if file not in processed:
            print(f"New file detected: {file.name}")

            result = ocr.process(
                image_path=file,
                prompt="<image>\\n<|grounding|>Convert the document to markdown.",
                output_dir=output_dir
            )

            processed.add(file)
            print(f"✓ Processed: {file.name}")

    time.sleep(5)
```
        """)


def main():
    """Main demo function."""
    print("="*60)
    print("DeepSeek-OCR for MacOS - Interactive Demo")
    print("="*60)

    # Check setup
    if not check_setup():
        sys.exit(1)

    demos = [
        ("Simple OCR", demo_simple_ocr),
        ("Document to Markdown", demo_document_conversion),
        ("Figure Parsing", demo_figure_parsing),
        ("Batch Processing", demo_batch_processing),
        ("Resolution Modes", demo_resolution_modes),
        ("Custom Prompts", demo_custom_prompts),
        ("Integration Examples", demo_integration_examples),
    ]

    while True:
        print("\n" + "="*60)
        print("Available Demos:")
        print("="*60)

        for i, (name, _) in enumerate(demos, 1):
            print(f"  {i}. {name}")
        print("  0. Exit")

        try:
            choice = input("\nSelect demo (0-7): ").strip()

            if choice == '0':
                print("\nGoodbye!")
                break

            demo_idx = int(choice) - 1
            if 0 <= demo_idx < len(demos):
                _, demo_func = demos[demo_idx]
                demo_func()
            else:
                print("Invalid choice, please try again")

        except (ValueError, KeyboardInterrupt):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
