# トラブルシューティングガイド

DeepSeek-OCR for MacOS MPSで発生する一般的な問題と解決方法

## ❌ ImportError: cannot import name 'LlamaFlashAttention2'

### エラーメッセージ
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

### 原因
Hugging Faceからダウンロードされたモデルコードが、Flash Attention 2を使おうとしていますが、これはMPSと互換性がありません。

### 解決方法

#### 方法1: 自動修正スクリプトを使用（推奨）

```bash
python fix_model.py
```

このスクリプトは：
1. キャッシュされたモデルファイルを検索
2. Flash Attention関連のコードを自動的にパッチ
3. 元のファイルを `.backup` として保存
4. 修正が完了したら通常通り実行可能

#### 方法2: 環境変数を設定

```bash
# Flash Attentionを無効化
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# SDPAを使用するよう指定
python test_ocr.py your_image.png
```

または、Pythonコード内で：

```python
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# モデルロード時にattn_implementationを明示的に指定
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa'  # SDPAを使用
)
```

#### 方法3: Transformersを最新版に更新

```bash
pip install --upgrade transformers
```

ただし、これだけでは解決しない場合があります。

#### 方法4: キャッシュをクリアして再ダウンロード

```bash
# キャッシュを削除
rm -rf ~/.cache/huggingface/modules/transformers_modules/deepseek*

# 修正後に再実行
python test_ocr.py your_image.png
```

## ❌ MPS not available

### エラーメッセージ
```
MPS not available, using CPU
```

### 原因
- MacOS 12.3未満を使用している
- PyTorchがMPS対応版でインストールされていない

### 解決方法

#### MacOSバージョンを確認
```bash
sw_vers
```

MacOS 12.3以降が必要です。

#### PyTorchを再インストール
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### MPSの可用性を確認
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## ❌ Out of Memory (OOM)

### エラーメッセージ
```
RuntimeError: MPS backend out of memory
```

### 原因
画像サイズが大きすぎる、またはシステムメモリが不足

### 解決方法

#### 方法1: より小さい解像度モードを使用

```bash
# Tinyモード（最小メモリ使用）
python ocr.py --tiny image.jpg

# Smallモード
python ocr.py --small image.jpg
```

#### 方法2: CPUモードを使用

```bash
python ocr.py --cpu image.jpg
```

#### 方法3: 他のアプリケーションを閉じる

アクティビティモニタで他のメモリ集約的なアプリを終了

#### 方法4: 画像を分割して処理

```python
from PIL import Image

# 大きい画像を分割
img = Image.open('large_image.jpg')
width, height = img.size

# 上半分を処理
top_half = img.crop((0, 0, width, height//2))
top_half.save('top.jpg')

# 下半分を処理
bottom_half = img.crop((0, height//2, width, height))
bottom_half.save('bottom.jpg')
```

## ❌ Slow Performance

### 症状
推論が非常に遅い（>10秒/画像）

### 原因と解決方法

#### CPUモードで実行されている

```python
import torch
print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
```

MPSが使われていない場合、`--cpu`フラグを削除

#### 大きすぎる画像サイズ

```bash
# より小さいモードを試す
python ocr.py --small image.jpg
```

#### 他のGPU集約的なアプリが実行中

アクティビティモニタでGPU使用状況を確認

## ❌ Model Download Fails

### エラーメッセージ
```
Connection error / Timeout
```

### 解決方法

#### 方法1: ミラーを使用

```bash
export HF_ENDPOINT=https://hf-mirror.com
python test_ocr.py your_image.png
```

#### 方法2: 手動ダウンロード

```bash
# git-lfsをインストール
brew install git-lfs
git lfs install

# モデルをクローン
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR

# ローカルから読み込み
```

```python
model = AutoModel.from_pretrained(
    './DeepSeek-OCR',
    trust_remote_code=True,
    local_files_only=True
)
```

## ❌ Incorrect OCR Results

### 症状
テキストが正しく認識されない

### 解決方法

#### 方法1: 解像度モードを変更

```bash
# より高品質なモード
python ocr.py --large document.jpg

# 動的解像度（大きいドキュメント用）
python ocr.py document.jpg  # デフォルトでGundamモード
```

#### 方法2: プロンプトを調整

```bash
# レイアウト保持あり
python ocr.py --markdown document.jpg

# レイアウト保持なし
python ocr.py --free document.jpg

# カスタムプロンプト
python ocr.py --prompt "Extract text preserving original layout" document.jpg
```

#### 方法3: 画像前処理

```python
from PIL import Image, ImageEnhance

# コントラストを上げる
img = Image.open('document.jpg')
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(2.0)
img.save('enhanced.jpg')

# 処理
# python ocr.py enhanced.jpg
```

## ❌ Module Not Found Errors

### エラーメッセージ
```
ModuleNotFoundError: No module named 'XXX'
```

### 解決方法

```bash
# すべての依存関係を再インストール
pip install -r requirements_mps.txt

# または個別に
pip install transformers tokenizers Pillow PyMuPDF einops easydict addict numpy
```

## ❌ Permission Errors

### エラーメッセージ
```
PermissionError: [Errno 13] Permission denied
```

### 解決方法

```bash
# キャッシュディレクトリの権限を修正
chmod -R u+w ~/.cache/huggingface

# または異なるキャッシュディレクトリを使用
export HF_HOME=/path/to/writable/directory
```

## ❌ "trust_remote_code" Warning

### 警告メッセージ
```
Warning: You are using untrusted remote code
```

### これは正常です

DeepSeek-OCRはカスタムモデルコードを使用しているため、`trust_remote_code=True`が必要です。これは公式のDeepSeekモデルなので安全です。

警告を抑制するには：

```python
import warnings
warnings.filterwarnings('ignore')
```

## 🔧 一般的なデバッグ手順

### 1. 環境情報を収集

```python
import torch
import transformers
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### 2. 詳細なエラー情報を有効化

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. ミニマルな再現コード

```python
import torch
from transformers import AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True
)
print("✓ Tokenizer loaded")

# モデルのロードはコメントアウトしてテスト
# model = AutoModel.from_pretrained(...)
```

## 📞 サポート

それでも問題が解決しない場合：

1. **GitHub Issues**: https://github.com/deepseek-ai/DeepSeek-OCR/issues
2. **Hugging Face Discussions**: https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions

問題を報告する際は、以下の情報を含めてください：

```bash
# システム情報
sw_vers
python --version
pip list | grep -E "torch|transformers"

# エラーログ全文
python test_ocr.py your_image.png 2>&1 | tee error.log
```
