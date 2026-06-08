#!/usr/bin/env python3
"""
ReazonSpeech データセットから train.txt を生成
WAV → discrete units に変換（mHuBERT 使用）
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/speech_ai"))

from datasets import load_dataset
import soundfile as sf
from speechgpt.utils.speech2unit.speech2unit import Speech2Unit


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReazonSpeech データセットから train.txt を生成"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/speech_ai/data/stage1_reazonspeech",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="テストモード：10サンプルのみ処理"
    )
    parser.add_argument(
        "--train-hours",
        type=float,
        default=10000.0,
        help="訓練データの時間数（デフォルト: 10000）"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/tmp/speech_ai_temp",
        help="一時ファイルディレクトリ"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # テストモード
    if args.test:
        train_hours = 10.0
        max_samples = 10
        print("=" * 60)
        print(f"テストモード: {max_samples}サンプルを処理")
        print("=" * 60)
    else:
        train_hours = args.train_hours
        max_samples = None
        print("=" * 60)
        print(f"フル訓練モード: Train {train_hours:.0f}h")
        print("=" * 60)

    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Speech2Unit ロード
    print("Loading Speech2Unit model...")
    s2u = Speech2Unit(ckpt_dir=os.path.expanduser("~/speech_ai/speechgpt/utils/speech2unit/"))
    print("✓ Model loaded\n")

    # ReazonSpeech データセットをロード
    print("Loading ReazonSpeech dataset...")
    dataset = load_dataset(
        "reazon-research/reazonspeech",
        "small-v1",
        trust_remote_code=True,
        verification_mode="no_checks"
    )
    train_data = dataset['train']
    print(f"✓ Dataset loaded\n")

    # Temp ディレクトリを作成
    temp_dir = Path(args.temp_dir).expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temp directory: {temp_dir}\n")

    # ファイルを開く（追記モード）
    train_file = output_dir / "train.txt"
    train_f = open(train_file, "a", encoding="utf-8")

    total_duration_hours = 0.0
    train_count = 0
    idx = 0
    file_counter = 0

    print(f"Processing samples...")
    if not args.test:
        print(f"Train: {train_hours:.0f}h\n")

    try:
        try:
            for sample in train_data:
                # テストモード：サンプル数制限
                if max_samples and idx >= max_samples:
                    print(f"  Reached {max_samples} samples. Stopping...")
                    break

                temp_wav_path = None
                try:
                    # オーディオを取得
                    audio = sample['audio']
                    wav = audio['array']
                    sr = audio['sampling_rate']

                    # 音声の長さ（秒）
                    duration_sec = len(wav) / sr
                    total_duration_hours += duration_sec / 3600

                    # 一時ファイルに保存
                    temp_wav_path = temp_dir / f"temp_{file_counter}.wav"
                    sf.write(str(temp_wav_path), wav, sr)
                    file_counter += 1

                    # units に変換
                    units = s2u(str(temp_wav_path), merged=True)

                    # 都度書き込み
                    if total_duration_hours <= train_hours:
                        train_f.write(f"{units}\n")
                        train_f.flush()
                        train_count += 1

                    # 進捗表示
                    if (idx + 1) % 10 == 0 or args.test:
                        print(f"  [{idx + 1:6d}] {total_duration_hours:7.2f}h | Train: {train_count:6d}")

                    # フル実行時：目標時間に達したら終了
                    if not args.test and total_duration_hours >= train_hours:
                        print(f"  Reached {train_hours:.0f} hours. Stopping...")
                        break

                except Exception as e:
                    print(f"  [{idx}] Skipped: {e}")

                finally:
                    # 一時ファイルを削除
                    if temp_wav_path and temp_wav_path.exists():
                        temp_wav_path.unlink()
                    idx += 1

        except Exception as e:
            print(f"\n  ⚠️  Dataset iteration error: {e}")
            print(f"  → Continuing with {train_count} samples collected\n")

    finally:
        # ファイルを閉じる
        train_f.close()

    # 統計情報
    print(f"\n{'='*60}")
    print(f"✓ Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration_hours:.2f} hours")
    print(f"Train samples: {train_count}")
    print(f"Output: {train_file}")


if __name__ == "__main__":
    main()