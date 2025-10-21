# DeepSeek-OCR MacOS MPS クイックスタートガイド

## 5分でスタート

### ステップ 1: 環境準備

```bash
# このリポジトリのmacos-mpsディレクトリに移動
cd DeepSeek-OCR/macos-mps

# 仮想環境を作成（推奨）
python3 -m venv venv
source venv/bin/activate
```

### ステップ 2: 依存関係のインストール

```bash
# PyTorch (MPS対応版)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# その他のパッケージ
pip install transformers tokenizers Pillow PyMuPDF einops easydict addict numpy
```

または：

```bash
pip install -r requirements_mps.txt
```

### ステップ 3: 動作確認

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
# 出力: MPS available: True （成功した場合）
```

### ステップ 4: 最初のOCR実行

#### ⚠️ よくあるエラーと対処法

もし以下のようなエラーが出た場合：
```
ImportError: cannot import name 'LlamaFlashAttention2'
```

修正スクリプトを実行してください：
```bash
python fix_model.py
```

その後、もう一度実行：
```bash
python ocr.py your_image.jpg
```

詳しくは [QUICK_FIX.md](QUICK_FIX.md) または [TROUBLESHOOTING.md](TROUBLESHOOTING.md) を参照してください。

#### 正常に動作する場合

```python
from transformers import AutoModel, AutoTokenizer
import torch

# デバイス設定
device = "mps"  # またはMPSが使えない場合は "cpu"

# モデルロード（初回は数分かかります）
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True
)

# OCR実行
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR.",
    image_file='your_image.jpg',  # ← ここに画像のパスを指定
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True
)

print(result)
```

## よくあるユースケース

### ドキュメントのMarkdown変換

```python
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file='document.pdf',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)
```

### 図表の解析

```python
result = model.infer(
    tokenizer,
    prompt="<image>\nParse the figure.",
    image_file='chart.png',
    base_size=1024,
    image_size=640,
    crop_mode=True
)
```

### シンプルなテキスト抽出

```python
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR.",
    image_file='screenshot.png',
    base_size=640,
    image_size=640,
    crop_mode=False  # 小さい画像の場合
)
```

## パフォーマンス設定

### 高速処理（低解像度）

```python
# Tinyモード - 最速
base_size=512, image_size=512, crop_mode=False
```

### 標準処理

```python
# Smallモード - バランス良い
base_size=640, image_size=640, crop_mode=False
```

### 高品質処理

```python
# Baseモード - 高品質
base_size=1024, image_size=1024, crop_mode=False
```

### 大きな文書

```python
# Gundamモード - 動的解像度
base_size=1024, image_size=640, crop_mode=True
```

## トラブルシューティング

### エラー: `flash_attn` が必要

**解決策**: モデルをローカルにダウンロードして修正：

```bash
# モデルをダウンロード
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR

# modeling_deepseekocr.py を編集
# flash_attn のインポートを削除
# flash_attn_qkvpacked_func を torch.nn.functional.scaled_dot_product_attention に置換

# ローカルから読み込み
model = AutoModel.from_pretrained('./DeepSeek-OCR', trust_remote_code=True)
```

### エラー: メモリ不足

**解決策**:

1. より小さい解像度を使用:
```python
base_size=512, image_size=512, crop_mode=False
```

2. dtype を変更:
```python
torch_dtype=torch.float16  # bfloat16の代わりに
```

### MPS が使えない

**確認事項**:
- MacOS 12.3以降か確認
- PyTorchがMPS対応版か確認: `pip list | grep torch`
- 必要に応じて再インストール:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## さらに詳しく

- 詳細なドキュメント: `README_MPS.md`
- サンプルコード: `example_usage.py`
- 実行スクリプト: `run_deepseek_ocr_mps.py`

## 質問・問題

- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions
