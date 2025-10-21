# DeepSeek-OCR MacOS MPS セットアップ完全ガイド

このガイドは、MacOS上でDeepSeek-OCRをMPS（Metal Performance Shaders）またはCPUで動作させるための完全な手順を提供します。

## 📋 前提条件

- MacOS 12.3以降（MPS使用の場合）
- Python 3.8以降
- Apple Silicon (M1/M2/M3) または Intel Mac
- 最低16GB RAM推奨

## 🚀 クイックスタート

### 1. 初期セットアップ

```bash
cd macos-mps
python setup.py
```

これにより：
- 必要な依存パッケージをインストール
- DeepSeek-OCRモデルをダウンロード
- 環境を確認

### 2. 互換性問題の修正

```bash
python fix_all.py
```

これにより自動的に以下を修正：
- Flash Attention参照の削除（MPS非互換）
- CUDA固有のコードの削除
- デバイスタイプの調整
- Pythonキャッシュのクリア

### 3. OCR実行

```bash
# MPSモード（速い、Apple Silicon推奨）
python ocr.py image.png

# CPUモード（遅いが確実に動作）
python ocr.py --cpu image.png
```

## 🔧 詳細セットアップ手順

### ステップ1: 依存関係のインストール

```bash
# 必須パッケージ
pip install torch torchvision
pip install transformers==4.46.3  # 推奨バージョン
pip install tokenizers Pillow PyMuPDF einops easydict addict numpy
```

**重要:** transformers 4.46.3を使用することを強く推奨します。より新しいバージョン（4.50+）では`DynamicCache.seen_tokens`が削除されており、互換性問題が発生します。

### ステップ2: モデルのダウンロード

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)"
```

または`setup.py`スクリプトを使用：

```bash
python setup.py
```

### ステップ3: モデルファイルの修正

ダウンロード後、キャッシュされたモデルファイルを修正する必要があります：

```bash
python fix_all.py
```

このスクリプトは以下を実行します：

1. **Flash Attention削除**: CUDA専用のFlash Attention 2参照を削除
2. **CUDA参照削除**: `.cuda()`呼び出しと`device_type='cuda'`を削除
3. **標準Attention使用**: すべてのAttentionメカニズムを標準`LlamaAttention`に統一
4. **キャッシュクリア**: Pythonバイトコードキャッシュを削除

### ステップ4: 動作確認

```bash
# 環境チェック
python check_env.py

# キャッシュ構造確認
python debug_cache.py
```

## 📝 使用方法

### 基本的な使い方

```bash
# 単一画像のOCR
python ocr.py image.png

# マークダウン形式で出力
python ocr.py --markdown document.png

# 複数画像のバッチ処理
python ocr.py --batch image1.png image2.png image3.png
```

### 解像度モード

```bash
# Tiny（最小メモリ）
python ocr.py --tiny image.png

# Small
python ocr.py --small image.png

# Large（最高品質）
python ocr.py --large image.png

# Gundam（動的解像度、デフォルト）
python ocr.py image.png
```

### デバイス選択

```bash
# MPS（Apple Silicon GPU）
python ocr.py image.png

# CPU（確実だが遅い）
python ocr.py --cpu image.png
```

## ❗ よくある問題と解決方法

### 問題1: LlamaFlashAttention2 ImportError

```bash
python fix_all.py
```

### 問題2: LlamaSdpaAttention is not defined

```bash
python fix_correct.py
```

または

```bash
python fix_all.py
```

### 問題3: CUDA not available

```bash
python fix_cuda_aggressive.py
```

または

```bash
python fix_all.py
```

### 問題4: DynamicCache.seen_tokens エラー

```bash
# transformersをダウングレード
pip install transformers==4.46.3
```

### 問題5: MPS Device Placement エラー

```
UserWarning: input_ids is on cpu, whereas the model is on mps
RuntimeError: Placeholder storage has not been allocated on MPS device
```

**推奨解決策:** CPUモードを使用

```bash
python ocr.py --cpu image.png
```

**実験的修正:**

```bash
python fix_mps_device.py
```

ただし、完全には解決できない可能性があります。現時点ではCPUモードが最も確実です。

## 🔍 トラブルシューティング

詳細なトラブルシューティングガイドは以下を参照：

```bash
cat TROUBLESHOOTING.md
```

または、デバッグ情報を収集：

```bash
# 環境情報
python check_env.py

# キャッシュ構造
python debug_cache.py
```

## 📂 修正スクリプト一覧

| スクリプト | 目的 | 推奨度 |
|----------|------|-------|
| `fix_all.py` | すべての互換性問題を一度に修正 | ⭐⭐⭐⭐⭐ |
| `fix_correct.py` | Flash Attention問題のみ修正 | ⭐⭐⭐ |
| `fix_cuda_aggressive.py` | CUDA参照のみ削除 | ⭐⭐⭐ |
| `fix_mps_device.py` | MPS device placement修正（実験的） | ⭐⭐ |
| `fix_cache.py` | DynamicCache互換性修正 | ⭐⭐ |

**推奨:** まず`fix_all.py`を実行してください。これがすべての一般的な問題をカバーします。

## 🎯 推奨ワークフロー

1. **初回セットアップ:**
   ```bash
   python setup.py
   python fix_all.py
   ```

2. **動作確認:**
   ```bash
   python check_env.py
   ```

3. **テスト実行:**
   ```bash
   # まずCPUモードで確認
   python ocr.py --cpu test_image.png

   # 成功したらMPSモードを試す
   python ocr.py test_image.png
   ```

4. **問題が発生した場合:**
   - `TROUBLESHOOTING.md`を確認
   - `python fix_all.py`を再実行
   - それでも解決しない場合は`--cpu`フラグを使用

## 📊 パフォーマンス比較

| モード | 速度 | メモリ使用量 | 安定性 |
|------|------|------------|-------|
| MPS | ⚡⚡⚡ | 中 | ⚠️ 問題あり |
| CPU | 🐢 | 低 | ✅ 安定 |

**現在の推奨:** CPUモードでの使用を推奨します。MPSモードは開発中です。

## 🔄 アップデート時の注意

モデルを再ダウンロードした場合、修正を再適用する必要があります：

```bash
# モデル再ダウンロード後
python fix_all.py
```

## 🆘 サポート

問題が解決しない場合：

1. **環境情報を収集:**
   ```bash
   python check_env.py > env_info.txt
   python debug_cache.py > cache_info.txt
   ```

2. **エラーログを保存:**
   ```bash
   python ocr.py image.png 2>&1 | tee error.log
   ```

3. **報告先:**
   - GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
   - Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions

## 📖 追加リソース

- `README_MPS.md` - MPS実装の技術詳細
- `QUICK_START_JP.md` - クイックスタートガイド
- `TROUBLESHOOTING.md` - 詳細なトラブルシューティング
- `example_usage.py` - Pythonコード例

## ✅ 動作確認チェックリスト

- [ ] Python 3.8以降がインストールされている
- [ ] `pip list | grep transformers`で4.46.3が表示される
- [ ] `python check_env.py`でMPSまたはCPUが利用可能
- [ ] `python setup.py`が正常に完了
- [ ] `python fix_all.py`が成功
- [ ] `python ocr.py --cpu test.png`が動作

すべてチェックできれば、セットアップは完了です！🎉

## 🔮 今後の改善予定

- [ ] MPSデバイス配置問題の完全解決
- [ ] より新しいtransformersバージョンへの対応
- [ ] パフォーマンスの最適化
- [ ] バッチ処理の改善

---

**最終更新:** 2025-10-21
**動作確認済み環境:** MacOS 13+, Python 3.10, transformers 4.46.3
