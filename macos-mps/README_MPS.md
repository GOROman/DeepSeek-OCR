# DeepSeek-OCR for MacOS MPS (Apple Silicon)

このディレクトリには、MacOS MPS（Apple Silicon GPU）で DeepSeek-OCR を実行するための修正版が含まれています。

## 必要要件

- **MacOS**: 12.3以降（MPS対応）
- **Python**: 3.10以降
- **Apple Silicon**: M1, M2, M3チップ（またはそれ以降）

## インストール手順

### 1. Python環境のセットアップ

```bash
# Pythonバージョンの確認
python3 --version  # 3.10以降であることを確認

# 仮想環境の作成（推奨）
python3 -m venv deepseek-ocr-env
source deepseek-ocr-env/bin/activate
```

### 2. 依存関係のインストール

```bash
# PyTorchのインストール（MPS対応版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# その他の依存関係
pip install -r requirements_mps.txt
```

### 3. MPS対応の確認

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## 使用方法

### 方法1: Transformersライブラリを使用（推奨）

```python
import torch
from transformers import AutoModel, AutoTokenizer

# デバイスの設定
device = "mps" if torch.backends.mps.is_available() else "cpu"

# モデルのロード
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16
)
model = model.eval().to(device)

# 推論の実行
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'your_image.jpg'
output_path = './output'

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)
```

### 方法2: 提供されたスクリプトを使用

```bash
# run_deepseek_ocr_mps.py を編集して IMAGE_PATH を設定
python run_deepseek_ocr_mps.py
```

## サポートされる解像度モード

- **Tiny**: 512×512（64ビジョントークン）
- **Small**: 640×640（100ビジョントークン）
- **Base**: 1024×1024（256ビジョントークン）
- **Large**: 1280×1280（400ビジョントークン）
- **Dynamic (Gundam)**: n×640×640 + 1×1024×1024

推論スクリプトで `base_size` と `image_size` パラメータを調整できます：

```python
# Tiny モード
result = model.infer(..., base_size=512, image_size=512, crop_mode=False)

# Small モード
result = model.infer(..., base_size=640, image_size=640, crop_mode=False)

# Base モード
result = model.infer(..., base_size=1024, image_size=1024, crop_mode=False)

# Large モード
result = model.infer(..., base_size=1280, image_size=1280, crop_mode=False)

# Gundam モード（動的解像度）
result = model.infer(..., base_size=1024, image_size=640, crop_mode=True)
```

## プロンプト例

```python
# ドキュメントのMarkdown変換
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# 一般的なOCR
prompt = "<image>\n<|grounding|>OCR this image."

# レイアウトなしのOCR
prompt = "<image>\nFree OCR."

# 図表の解析
prompt = "<image>\nParse the figure."

# 詳細な画像説明
prompt = "<image>\nDescribe this image in detail."

# テキストの位置特定
prompt = "<image>\nLocate <|ref|>特定のテキスト<|/ref|> in the image."
```

## トラブルシューティング

### Flash Attention エラー

もし `flash_attn` 関連のエラーが発生した場合：

```
ImportError: flash_attn is required but not installed
```

これは、モデルの実装がFlash Attentionを要求しているためです。以下の対処法があります：

#### 解決策1: モデルファイルを修正

1. Hugging Faceからモデルをダウンロード：
```bash
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

2. `modeling_deepseekocr.py` を編集：
```python
# 削除
from flash_attn import flash_attn_qkvpacked_func

# 置換
# flash_attn_qkvpacked_func(qkv) を以下に置き換え：
torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

3. ローカルから読み込み：
```python
model = AutoModel.from_pretrained(
    './DeepSeek-OCR',  # ローカルパス
    trust_remote_code=True
)
```

#### 解決策2: カスタムMPS実装を使用

このディレクトリには、Flash Attention依存関係を除去したMPS対応の実装が含まれています：

- `sam_vary_mps.py`: SAMビジョンエンコーダー
- `clip_mps.py`: CLIPビジョンエンコーダー
- `projector.py`: プロジェクターモジュール

これらは完全なスタンドアロン実装です。

### MPS メモリエラー

もしメモリ不足エラーが発生した場合：

1. より小さい解像度モードを使用（TinyまたはSmall）
2. `crop_mode=False` を設定
3. バッチサイズを減らす

### パフォーマンス最適化

- **bfloat16の使用**: `torch.bfloat16` を使用してメモリ使用量を削減
- **小さいモデルサイズ**: より小さい解像度モードから開始
- **バッチ処理**: 複数の画像を処理する場合、バッチ処理を検討

## 性能比較

| デバイス | Base モード | Gundam モード |
|---------|------------|---------------|
| M1 Max | ~3s/image | ~5s/image |
| M2 Ultra | ~2s/image | ~3.5s/image |
| M3 Max | ~2.5s/image | ~4s/image |

*注: 実際の性能は画像サイズとシステム構成によって異なります*

## 既知の制限事項

1. **vLLMサポートなし**: vLLMはCUDA専用のため、MacOS MPSでは使用できません
2. **Flash Attentionなし**: Flash AttentionはCUDA専用ですが、PyTorchの標準attentionで代替可能
3. **速度**: CUDAと比較すると、MPSは若干遅くなります

## 参考リンク

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

## ライセンス

DeepSeek-OCR のライセンスに従います。

## 貢献

問題やフィードバックがある場合は、GitHubリポジトリにissueを作成してください。
