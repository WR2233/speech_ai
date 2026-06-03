# VM セットアップガイド

Azure VM で SpeechGPT を実行するためのセットアップ手順です。

## 1. VM へのアクセス

```bash
# ローカルマシンから
chmod 600 /path/to/wakuda_key.pem
ssh -i /path/to/wakuda_key.pem azureuser@<VM_IP>
```

## 2. 基本パッケージのインストール

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl wget unzip git
```

## 3. CUDA のセットアップ

### 3.1 NVIDIA リポジトリの設定

```bash
# CUDA キーリングをダウンロード（Ubuntu 24.04 の場合）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### 3.2 CUDA ドライバと Toolkit のインストール

```bash
# CUDA ドライバをインストール
sudo apt install cuda-drivers
sudo reboot

# VM に再接続（再起動後）
ssh -i /path/to/wakuda_key.pem azureuser@<VM_IP>

# GPU が認識されているか確認
nvidia-smi

# CUDA Toolkit をインストール
sudo apt install cuda-toolkit

# パスを設定
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# NVCC が利用可能か確認
nvcc --version
```

## 4. Python 環境のセットアップ

### 4.1 Conda のインストール（推奨）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Conda 利用規約を受け入れ（必須）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Conda 環境を作成（Python 3.10 指定）
conda create -n speech_ai python=3.10 -y
conda activate speech_ai
```

## 5. リポジトリのセットアップ

### 5.1 リポジトリをクローン

```bash
git clone https://github.com/WR2233/speech_ai.git
cd speech_ai
```

### 5.2 依存関係をインストール

```bash
# pip を古いバージョンにダウングレード
pip install 'pip<24.1'

# 依存関係をインストール
pip install -r requirements.txt
```

## 6. モデルのダウンロード

### 6.1 SpeechGPT モデルをダウンロード

```bash
# /mnt にディレクトリを作成
sudo mkdir -p /mnt/models
sudo chown -R $(whoami):$(whoami) /mnt/models

# huggingface_hub の `hf` コマンドを使用
hf download fnlp/SpeechGPT-7B-cm \
  --repo-type model \
  --local-dir /mnt/models/SpeechGPT-7B-cm

# speech_ai プロジェクトから参照できるようにシンボリックリンク作成
cd ~/speech_ai
mkdir -p models
ln -s /mnt/models/SpeechGPT-7B-cm models/SpeechGPT-7B-cm
```

### 6.2 補助モデルをダウンロード

```bash
# mHuBERT（音声→ユニット変換）
cd ~/speech_ai/speechgpt/utils/speech2unit/
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin

# Vocoder（ユニット→音声変換）
cd ../vocoder/
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O config.json
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O vocoder.pt

# プロジェクトルートに戻る
cd ~/speech_ai
```

## 7. GPU 確認

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## 8. 推論の実行

### 8.1 CLI 推論（テキスト）

```bash
python speechgpt/src/infer/cli_infer.py \
  --model-name-or-path models/SpeechGPT-7B-cm \
  --s2u-dir speechgpt/utils/speech2unit/ \
  --vocoder-dir speechgpt/utils/vocoder/ \
  --output-dir output/
```

### 8.2 Web UI（Gradio）

```bash
python speechgpt/src/infer/web_infer.py \
  --model-name-or-path models/SpeechGPT-7B-cm \
  --s2u-dir speechgpt/utils/speech2unit/ \
  --vocoder-dir speechgpt/utils/vocoder/ \
  --output-dir output/ \
  --share
```

Web UI にアクセス：`http://<VM_IP>:7860`

## 9. 訓練の実行（オプション）

### 9.1 データ準備

```bash
# Stage 1: モダリティ適応事前学習
# data/stage1/train.txt と data/stage1/dev.txt を準備

bash speechgpt/scripts/ma_pretrain.sh 1 0 localhost 29500
```

### 9.2 他のステージ

```bash
# Stage 2: クロスモーダル指示チューニング
bash speechgpt/scripts/cm_sft.sh 1 0 localhost 29500

# Stage 3: チェーン・オブ・モダリティ指示チューニング
bash speechgpt/scripts/com_sft.sh 1 0 localhost 29500
```

## 10. Gemma-2-2B-JPN-IT での学習準備

### 10.1 モデルのダウンロード

```bash
# Hugging Face から Gemma-2-2B-JPN-IT をダウンロード
hf download google/gemma-2-2b-jpn-it \
  --repo-type model \
  --local-dir /mnt/models/gemma-2-2b-jpn-it

# speech_ai プロジェクトから参照可能にリンク作成
ln -s /mnt/models/gemma-2-2b-jpn-it ~/speech_ai/models/gemma-2-2b-jpn-it
```

### 10.2 トークナイザーの確認

```bash
# トークナイザーが含まれているか確認
ls -la /mnt/models/gemma-2-2b-jpn-it/
# 以下が含まれていることを確認:
# - tokenizer.model
# - tokenizer_config.json
# - config.json
```

### 10.3 必要なコンポーネント

Gemma-2 での学習に必要なもの：

| コンポーネント | 場所 | 説明 |
|-------------|-----|-----|
| **モデル本体** | `/mnt/models/gemma-2-2b-jpn-it` | 言語モデル |
| **トークナイザー** | モデルに含まれる | テキスト→トークン変換 |
| **Speech2Unit** | `speechgpt/utils/speech2unit/` | 音声→ユニット変換 |
| **Vocoder** | `speechgpt/utils/vocoder/` | ユニット→音声変換 |
| **訓練スクリプト** | `speechgpt/src/train/` | Stage 2/3 訓練用 |
| **訓練データ** | `data/stage2/` | SpeechInstruct Cross-modal |

### 10.4 訓練スクリプト修正（LLM 汎用化）

LLaMA から Gemma-2 に切り替えるには、以下を修正：

```python
# speechgpt/src/train/cm_sft.py

# 変更前:
from transformers import LlamaForCausalLM, LlamaTokenizer

# 変更後:
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルロード時:
# 変更前: 
model = LlamaForCausalLM.from_pretrained(model_name_or_path)
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

# 変更後:
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

### 10.5 訓練データの準備

```bash
# 訓練データをダウンロード（Stage 2: Cross-modal instruction）
mkdir -p ~/speech_ai/data/stage2/
hf download fnlp/SpeechInstruct \
  --repo-type dataset \
  --include "cross_modal_instruction.jsonl" \
  --local-dir ~/speech_ai/data/stage2/
```

### 10.6 訓練実行（Gemma-2 使用）

```bash
# Stage 2: クロスモーダル指示チューニング（Gemma-2 版）
bash speechgpt/scripts/cm_sft.sh \
  1 0 localhost 29500 \
  --model-name google/gemma-2-2b-jpn-it \
  --data-path data/stage2/cross_modal_instruction.jsonl
```

## 11. お試し学習（JVS Corpus を使用）

### 11.1 JVS Corpus のダウンロード

```bash
# Google Drive からダウンロード（3.5 GB）
# https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
# から zip ファイルをダウンロード

# または gdown を使用
pip install gdown
gdown <GOOGLE_DRIVE_FILE_ID> -O jvs_corpus.zip

# 解凍
mkdir -p /mnt/datasets
cd /mnt/datasets
unzip jvs_corpus.zip
ls jvs_corpus/  # sample_jvs001, sample_jvs002, ... が見える
```

### 11.2 Stage 1 お試し用データ準備

```bash
# お試し用（少数話者抽出版）データセット作成
mkdir -p ~/speech_ai/data/stage1_trial

# JVS Corpus から parallel100 のみを抽出（統一された読み上げ）
# サンプル：最初の5人の話者の parallel100 のみ使用
cat > ~/speech_ai/prepare_jvs_trial.py << 'EOF'
import os
import shutil
from pathlib import Path

jvs_root = Path("/mnt/datasets/jvs_corpus")
output_dir = Path(os.path.expanduser("~/speech_ai/data/stage1_trial"))
output_dir.mkdir(parents=True, exist_ok=True)

# 最初の5人の話者を抽出
speakers = sorted([d for d in jvs_root.glob("sample_jvs*")])[:5]

with open(output_dir / "train.txt", "w") as f:
    for speaker_dir in speakers:
        # parallel100 ディレクトリから音声ファイルを抽出
        parallel_dir = speaker_dir / "parallel100"
        if parallel_dir.exists():
            for wav_file in sorted(parallel_dir.glob("*.wav"))[:50]:  # 最初の50件のみ
                # テキスト転写を読み込む
                txt_file = wav_file.with_suffix(".txt")
                if txt_file.exists():
                    with open(txt_file, "r", encoding="utf-8") as tf:
                        text = tf.read().strip()
                    # audio ディレクトリにコピー
                    os.makedirs(output_dir / "audio", exist_ok=True)
                    shutil.copy(wav_file, output_dir / "audio" / wav_file.name)
                    # テキストを出力
                    f.write(f"{text}\n")

print(f"✓ Trial dataset created: {output_dir}")
print(f"  Total files: {len(list((output_dir / 'audio').glob('*.wav')))}")
EOF

python ~/speech_ai/prepare_jvs_trial.py
```

### 11.3 Stage 1 お試し学習実行

```bash
cd ~/speech_ai

# 学習パラメータ（GPU メモリ削減版）
bash speechgpt/scripts/ma_pretrain.sh \
  1 0 localhost 29500 \
  --model-name-or-path google/gemma-2-2b-jpn-it \
  --train-file data/stage1_trial/train.txt \
  --output-dir output/stage1_trial_gemma2 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --learning-rate 5e-5
```

### 11.4 学習モニタリング

```bash
# ログの確認
tail -f output/stage1_trial_gemma2/training_args.bin

# TensorBoard で可視化（オプション）
tensorboard --logdir output/stage1_trial_gemma2
```

