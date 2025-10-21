# クイックフィックス: LlamaFlashAttention2 エラー

## 問題

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

このエラーが発生した場合、以下の手順で解決できます。

## 🎯 推奨：統合セットアップ（一番簡単）

モデルがまだダウンロードされていない場合、この方法が最も簡単です：

```bash
cd macos-mps
python setup_and_fix.py
```

このスクリプトが自動的に：
1. ✓ モデルコードをダウンロード
2. ✓ Flash Attention問題を修正
3. ✓ 完全なモデルをダウンロード（~3GB）
4. ✓ 動作確認

完了したら、すぐに使えます：
```bash
python ocr.py your_image.png
```

## 既にエラーが出た場合（3ステップ）

### ステップ1: 修正スクリプトを実行

```bash
cd macos-mps
python fix_model.py
```

これで自動的にキャッシュされたモデルファイルが修正されます。

### ステップ2: 再度実行

```bash
python test_ocr.py your_image.png
```

または

```bash
python ocr.py your_image.png
```

### ステップ3: 確認

正常に動作すれば完了です！

## それでも動かない場合

### 代替方法1: キャッシュをクリア

```bash
# キャッシュを削除
rm -rf ~/.cache/huggingface/modules/transformers_modules/deepseek*

# 修正スクリプトを実行してから再試行
python fix_model.py
python test_ocr.py your_image.png
```

### 代替方法2: 手動でファイルを編集

1. キャッシュディレクトリを開く：
```bash
open ~/.cache/huggingface/modules/transformers_modules/
```

2. `deepseek` で始まるフォルダを探す

3. その中の `modeling_deepseekv2.py` を開く

4. 以下の行を探す：
```python
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
)
```

5. `LlamaFlashAttention2,` の行を削除：
```python
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaSdpaAttention,
)
```

6. `LlamaFlashAttention2` のすべての参照を `LlamaSdpaAttention` に置換

7. 保存して再実行

### 代替方法3: 環境変数で回避

```bash
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
python test_ocr.py your_image.png
```

## 詳細なトラブルシューティング

詳しい情報は [TROUBLESHOOTING.md](TROUBLESHOOTING.md) を参照してください。

## サポート

問題が解決しない場合は、以下の情報を含めてissueを作成してください：

```bash
# システム情報
sw_vers
python --version
pip list | grep -E "torch|transformers"

# エラーログ
python test_ocr.py your_image.png 2>&1 | tee error.log
```
