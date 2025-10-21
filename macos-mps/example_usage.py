"""
Example Usage of DeepSeek-OCR on MacOS MPS
このファイルには、様々な使用例が含まれています。
"""

import torch
from pathlib import Path


def example_1_simple_ocr():
    """例1: シンプルなOCR処理"""
    from transformers import AutoModel, AutoTokenizer

    # デバイス設定
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # モデルロード
    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    # OCR実行
    result = model.infer(
        tokenizer,
        prompt="<image>\nFree OCR.",
        image_file='sample.jpg',
        output_path='./output',
        base_size=640,
        image_size=640,
        crop_mode=False
    )

    print(result)


def example_2_document_to_markdown():
    """例2: ドキュメントをMarkdownに変換"""
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    # 高解像度のドキュメント処理
    result = model.infer(
        tokenizer,
        prompt="<image>\n<|grounding|>Convert the document to markdown.",
        image_file='document.pdf',  # PDFも対応
        output_path='./output',
        base_size=1024,
        image_size=640,
        crop_mode=True,  # 大きな文書の場合はTrue
        save_results=True
    )

    # 結果をファイルに保存
    with open('./output/result.md', 'w', encoding='utf-8') as f:
        f.write(result)

    print("Markdown saved to: ./output/result.md")


def example_3_batch_processing():
    """例3: 複数画像のバッチ処理"""
    from transformers import AutoModel, AutoTokenizer
    from pathlib import Path

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    # 画像ディレクトリ
    image_dir = Path('./images')
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)

    # すべての画像を処理
    for image_path in image_dir.glob('*.{jpg,jpeg,png,pdf}'):
        print(f"Processing: {image_path.name}")

        result = model.infer(
            tokenizer,
            prompt="<image>\n<|grounding|>OCR this image.",
            image_file=str(image_path),
            output_path=str(output_dir / image_path.stem),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True
        )

        # 結果を個別に保存
        result_file = output_dir / f"{image_path.stem}.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(result)

        print(f"  Saved to: {result_file}")


def example_4_different_resolutions():
    """例4: 異なる解像度モードでの処理"""
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    image_file = 'test.jpg'
    prompt = "<image>\nFree OCR."

    # Tiny モード（最速、低解像度）
    print("Running Tiny mode...")
    result_tiny = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        base_size=512,
        image_size=512,
        crop_mode=False
    )

    # Base モード（標準）
    print("Running Base mode...")
    result_base = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        base_size=1024,
        image_size=1024,
        crop_mode=False
    )

    # Gundam モード（動的解像度、大きな文書用）
    print("Running Gundam mode...")
    result_gundam = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        base_size=1024,
        image_size=640,
        crop_mode=True
    )

    return {
        'tiny': result_tiny,
        'base': result_base,
        'gundam': result_gundam
    }


def example_5_figure_parsing():
    """例5: 図表の解析"""
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    # 図表を含む画像の処理
    result = model.infer(
        tokenizer,
        prompt="<image>\nParse the figure.",
        image_file='chart.png',
        output_path='./output',
        base_size=1024,
        image_size=640,
        crop_mode=True
    )

    print("Figure analysis:")
    print(result)


def example_6_custom_prompt():
    """例6: カスタムプロンプトの使用"""
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    # カスタムプロンプトで詳細な説明を取得
    custom_prompts = [
        "<image>\nDescribe this image in detail.",
        "<image>\n<|grounding|>Extract all text from this image and organize it by sections.",
        "<image>\nIdentify and transcribe all visible text, including headers and footnotes.",
    ]

    for i, prompt in enumerate(custom_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file='document.jpg',
            base_size=1024,
            image_size=640,
            crop_mode=True
        )
        print(f"Result {i+1}:")
        print(result)
        print("-" * 60)


if __name__ == "__main__":
    print("DeepSeek-OCR Examples for MacOS MPS")
    print("=" * 60)

    # MPS可用性チェック
    if torch.backends.mps.is_available():
        print("✓ MPS is available")
    else:
        print("⚠ MPS not available, using CPU")

    print("\nAvailable examples:")
    print("1. Simple OCR")
    print("2. Document to Markdown")
    print("3. Batch Processing")
    print("4. Different Resolutions")
    print("5. Figure Parsing")
    print("6. Custom Prompts")

    print("\nTo run an example, uncomment the corresponding function below:")
    print()

    # 使用例（コメントを外して実行）
    # example_1_simple_ocr()
    # example_2_document_to_markdown()
    # example_3_batch_processing()
    # example_4_different_resolutions()
    # example_5_figure_parsing()
    # example_6_custom_prompt()
