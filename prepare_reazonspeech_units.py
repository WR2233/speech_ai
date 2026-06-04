#!/usr/bin/env python3
"""
ReazonSpeech データセットから train.txt と valid.txt を生成
WAV → discrete units に変換（mHuBERT 使用）

使用例:
  # テストモード（10サンプルのみ）
  python prepare_reazonspeech_units.py --test

  # フル訓練（10,000時間）
  python prepare_reazonspeech_units.py
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
        description="ReazonSpeech データセットから train.txt / valid.txt を生成"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/speech_ai/data/stage1_reazonspeech",
        help="出力ディレクトリ（デフォルト: ~/speech_ai/data/stage1_reazonspeech）"
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
        "--valid-hours",
        type=float,
        default=2000.0,
        help="検証データの時間数（デフォルト: 2000）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # テストモード
    if args.test:
        train_hours = 999999.0
        valid_hours = 0.0
        max_samples = 10
        print("=" * 60)
        print(f"テストモード: {max_samples}サンプルを処理")
        print("=" * 60)
    else:
        train_hours = args.train_hours
        valid_hours = args.valid_hours
        max_samples = None
        print("=" * 60)
        print(f"フル訓練モード: Train {train_hours:.0f}h / Valid {valid_hours:.0f}h")
        print("=" * 60)

    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Speech2Unit ロード
    print("Loading Speech2Unit model...")
    s2u = Speech2Unit(ckpt_dir=os.path.expanduser("~/speech_ai/speechgpt/utils/speech2unit/"))
    print("✓ Model loaded\n")

    # ReazonSpeech データセットをロード（test.py の方法）
    print("Loading ReazonSpeech dataset...")
    dataset = load_dataset(
        "reazon-research/reazonspeech",
        "small-v1",
        trust_remote_code=True
    )
    train_data = dataset['train']
    print(f"✓ Dataset loaded\n")

    total_duration_hours = 0.0
    train_lines = []
    valid_lines = []
    target_hours = train_hours + valid_hours

    print(f"Processing samples...")
    if not args.test:
        print(f"Target: {target_hours:.0f} hours (Train: {train_hours:.0f}h / Valid: {valid_hours:.0f}h)\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            for idx, sample in enumerate(train_data):
                # テストモード：サンプル数制限
                if max_samples and idx >= max_samples:
                    print(f"  Reached {max_samples} samples. Stopping...")
                    break

                temp_wav_path = None
                try:
                # オーディオを取得
                try:
                    audio = sample['audio']
                    wav = audio['array']
                    sr = audio['sampling_rate']
                except Exception as audio_err:
                    print(f"  [{idx}] Audio decode error: {audio_err}")
                    continue

                # 音声の長さ（秒）
                duration_sec = len(wav) / sr
                total_duration_hours += duration_sec / 3600

                # 一時ファイルに保存
                temp_wav_path = Path(temp_dir) / f"temp_{idx}.wav"
                sf.write(str(temp_wav_path), wav, sr)

                # units に変換
                units = s2u(str(temp_wav_path), merged=True)

                # train / valid に分割
                if total_duration_hours <= train_hours:
                    train_lines.append(units)
                elif total_duration_hours <= target_hours:
                    valid_lines.append(units)

                # 進捗表示
                if (idx + 1) % 10 == 0 or args.test:
                    print(f"  [{idx + 1:6d}] {total_duration_hours:7.2f}h | Train: {len(train_lines):6d} | Valid: {len(valid_lines):6d}")

                # フル実行時：目標時間に達したら終了
                if not args.test and total_duration_hours >= target_hours:
                    print(f"  Reached {target_hours:.0f} hours. Stopping...")
                    break

            except Exception as e:
                print(f"  [{idx}] Warning: {e}")
                continue
                finally:
                    # 一時ファイルを削除
                    if temp_wav_path and temp_wav_path.exists():
                        temp_wav_path.unlink()
        except Exception as e:
            print(f"\n  ⚠️  Dataset iteration error (sample ~{idx}): {e}")
            print(f"  → Processing stopped, but will save {len(train_lines)} train samples\n")

    # train.txt に書き込み
    if train_lines:
        train_file = output_dir / "train.txt"
        print(f"\nWriting {train_file} ({len(train_lines)} lines)...")
        with open(train_file, "w", encoding="utf-8") as f:
            for line in train_lines:
                f.write(f"{line}\n")
    else:
        print("\n警告: train.txt にデータがありません")

    # valid.txt に書き込み
    if valid_lines:
        valid_file = output_dir / "valid.txt"
        print(f"Writing {valid_file} ({len(valid_lines)} lines)...")
        with open(valid_file, "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(f"{line}\n")
    else:
        print("警告: valid.txt にデータがありません")

    # 統計情報
    print(f"\n{'='*60}")
    print(f"✓ Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration_hours:.2f} hours")
    print(f"Train samples: {len(train_lines)}")
    print(f"Valid samples: {len(valid_lines)}")
    print(f"Output directory: {output_dir}")

    # サンプルプレビュー
    if train_lines:
        print(f"\n{'='*60}")
        print(f"Train.txt プレビュー (最初の3行):")
        print(f"{'='*60}")
        with open(output_dir / "train.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 3:
                    preview = line.strip()[:80] + "..." if len(line.strip()) > 80 else line.strip()
                    print(f"{i+1}: {preview}")
                else:
                    break


if __name__ == "__main__":
    main()
