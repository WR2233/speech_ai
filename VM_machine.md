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

# Conda 環境を作成
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
pip install -r requirements.txt
```

## 6. モデルのダウンロード

### 6.1 Git LFS をインストール

```bash
sudo apt install git-lfs
git lfs install
```

### 6.2 SpeechGPT モデルをダウンロード

```bash
git clone https://huggingface.co/fnlp/SpeechGPT-7B-cm models/SpeechGPT-7B-cm
```

### 6.3 補助モデルをダウンロード

```bash
# mHuBERT（音声→ユニット変換）
cd speechgpt/utils/speech2unit/
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin

# Vocoder（ユニット→音声変換）
cd ../vocoder/
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O config.json
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O vocoder.pt

# プロジェクトルートに戻る
cd ../../..
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
