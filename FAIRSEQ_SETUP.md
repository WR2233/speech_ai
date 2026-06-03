# Fairseq 環境セットアップ

fairseq の HiFiGAN Vocoder を使った音声生成は、別の conda 環境で実行します。

## セットアップ（1回のみ）

### 1. fairseq 専用環境を作成

```bash
conda create -n fairseq_env python=3.9 -y
conda activate fairseq_env

pip install fairseq==0.12.2
pip install soundfile torch
```

### 2. 動作確認

```bash
python -c "from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder; print('✓ fairseq OK')"
```

## 使用方法

### ステップ 1: main 環境で inference

メイン環境（speech_ai）で実行：

```bash
conda activate speech_ai
cd ~/speech_ai

python speechgpt/src/infer/simple_infer.py \
  --model-name-or-path output/stage1_trial_gemma2 \
  --prompt "こんにちは" \
  --output-dir inference_output
```

出力ファイル：`inference_output/inference_log_YYYYMMDD_HHMMSS.json`

### ステップ 2: fairseq 環境で音声生成

fairseq 環境で実行：

```bash
conda activate fairseq_env
cd ~/speech_ai

python speechgpt/src/infer/fairseq_generate_wav.py \
  --units "[194, 56, 402, 63, 497, 497, 780, 991, ...]" \
  --output inference_output/speech_output.wav
```

または JSON ファイルから自動抽出：

```bash
# inference_log の音声ユニットを JSON から取得してコマンドを生成
UNITS=$(python -c "import json; print(json.load(open('inference_output/inference_log_*.json'))['speech_units'])")

python speechgpt/src/infer/fairseq_generate_wav.py \
  --units "$UNITS" \
  --output inference_output/speech_output.wav
```

## トラブルシューティング

**fairseq インストール時にエラー**
```bash
# pip をダウングレード
pip install 'pip<24.1'
pip install fairseq==0.12.2
```

**omegaconf エラー**
```bash
# fairseq_env では omegaconf 競合がない（Python 3.9だから）
# main の speech_ai 環境が競合する場合は fairseq_env を使う
```

## 環境の切り替え

```bash
# main 環境（訓練・inference）
conda activate speech_ai

# fairseq 環境（音声生成）
conda activate fairseq_env
```
