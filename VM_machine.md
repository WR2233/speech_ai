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
mkdir -p ~/datasets
cd ~/datasets

gdown https://drive.google.com/uc?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt

# 解凍
unzip jvs_ver1.zip
ls jvs_ver1/  # sample_jvs001, sample_jvs002, ... が見える
```


```
pip install wandb
wandb login
```
### 11.2 Stage 1 お試し用データ準備（mHuBERT でユニット列に変換）

JVS Corpus の音声を mHuBERT で処理してユニット列に変換します：

#### 11.2.1 mHuBERT モデルダウンロード確認

```bash
# 既に準備済みか確認
ls -lh ~/speech_ai/speechgpt/utils/speech2unit/mhubert*.{pt,bin}

# なければダウンロード
cd ~/speech_ai/speechgpt/utils/speech2unit/
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
```

#### 11.2.3 データ確認

```bash
# 生成ファイル確認
ls -lh ~/speech_ai/data/stage1_trial/

# ユニット列の内容確認
head -3 ~/speech_ai/data/stage1_trial/train.txt

# 行数確認
wc -l ~/speech_ai/data/stage1_trial/train.txt
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

### 11.5 WandB による可視化（推奨）

#### 11.5.1 WandB セットアップ

```bash
# WandB CLI をインストール（既にインストール済みの場合はスキップ）
pip install wandb

# WandB にログイン
wandb login
# → https://wandb.ai/authorize で API キーを取得
# → コンソールでキーをペースト
```

#### 11.5.2 Stage 1 訓練で WandB を有効化

```bash
cd ~/speech_ai

# WandB 有効で Stage 1 学習実行
bash speechgpt/scripts/ma_pretrain.sh \
  1 0 localhost 29500 \
  --model-name-or-path google/gemma-2-2b-jpn-it \
  --train-file data/stage1_trial/train.txt \
  --output-dir output/stage1_trial_gemma2 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --learning-rate 5e-5 \
  --report_to wandb \
  --wandb_project speech-ai-stage1 \
  --wandb_entity <your-wandb-username>  # 省略時: personal workspace
```

#### 11.5.3 WandB ダッシュボードで確認

訓練開始後、以下にアクセス：

```
https://wandb.ai/<your-wandb-username>/speech-ai-stage1
```

**ダッシュボードで表示される項目**:
- Loss (訓練損失、検証損失)
- Learning Rate
- GPU メモリ使用率
- 訓練時間
- Epoch プログレス
- モデル設定（model, batch size など）

#### 11.5.4 複数実験の比較

複数の設定でお試し学習を実行する場合：

```bash
# 実験1: batch_size=4
bash speechgpt/scripts/ma_pretrain.sh 1 0 localhost 29500 \
  --model-name-or-path google/gemma-2-2b-jpn-it \
  --train-file data/stage1_trial/train.txt \
  --output-dir output/stage1_trial_bs4 \
  --per-device-train-batch-size 4 \
  --report_to wandb

# 実験2: batch_size=8
bash speechgpt/scripts/ma_pretrain.sh 1 0 localhost 29500 \
  --model-name-or-path google/gemma-2-2b-jpn-it \
  --train-file data/stage1_trial/train.txt \
  --output-dir output/stage1_trial_bs8 \
  --per-device-train-batch-size 8 \
  --report_to wandb
```

### 12.3 Fairseq 環境（音声生成用）

別環境で fairseq をインストール：
```bash
conda create -n fairseq_env python=3.9 -y
conda activate fairseq_env
pip install 'pip<24.1'
pip install omegaconf==2.0.5
pip install fairseq==0.12.2 soundfile
```

simple_infer.py が自動的に subprocess で fairseq を呼び出して .wav 生成。(現状動かず)
