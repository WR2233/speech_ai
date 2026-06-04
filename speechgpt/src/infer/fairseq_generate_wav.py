"""
fairseq 環境で実行する専用スクリプト
音声ユニット（JSON形式）から .wav を生成
"""

import torch
import json
import argparse
import os
import soundfile as sf
from pathlib import Path

from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder


def generate_wav_from_units(units, vocoder_path, config_path, output_path, sr=16000):
    """音声ユニットから .wav を生成"""

    if not os.path.exists(vocoder_path):
        raise FileNotFoundError(f"Vocoder not found: {vocoder_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ボコーダーを読み込み
    print(f"Loading vocoder from {vocoder_path}...")
    with open(config_path) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(vocoder_path, vocoder_cfg).to(device)

    # 音声ユニットをテンソルに変換
    units_tensor = torch.LongTensor(units).view(1, -1).to(device)
    code_dict = {"code": units_tensor}

    # 音声を生成
    print(f"Generating waveform from {len(units)} units...")
    with torch.no_grad():
        wav = vocoder(code_dict, False)

    # .wav を保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wav_np = wav.detach().cpu().numpy()
    sf.write(output_path, wav_np, sr)

    print(f"✓ WAV saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=str, required=True, help="Speech units (JSON list or comma-separated)")
    parser.add_argument("--output", type=str, required=True, help="Output .wav file path")
    parser.add_argument("--vocoder", type=str, default="speechgpt/utils/vocoder/vocoder.pt")
    parser.add_argument("--config", type=str, default="speechgpt/utils/vocoder/config.json")
    args = parser.parse_args()

    # ユニットをパース
    if args.units.startswith("["):
        # JSON形式
        units = json.loads(args.units)
    elif args.units.startswith("<sosp>"):
        # <sosp><245><166>...<eosp> 形式
        import re
        units = [int(x) for x in re.findall(r"<(\d+)>", args.units)]
    else:
        # カンマ区切り形式
        units = [int(u.strip()) for u in args.units.split(",")]

    print(f"Units: {units[:10]}... ({len(units)} total)")

    # .wav を生成
    generate_wav_from_units(units, args.vocoder, args.config, args.output)


if __name__ == "__main__":
    main()
