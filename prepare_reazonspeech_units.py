#!/usr/bin/env python3
"""
ReazonSpeech データセットをダウンロードして、train.txt と valid.txt を生成するスクリプト
WAV → discrete units に変換（mHuBERT 使用）

使用例:
  # フル訓練（10,000時間）
  python prepare_reazonspeech_units.py

  # テストモード（1時間のみ）
  python prepare_reazonspeech_units.py --test

  # カスタム出力ディレクトリ
  python prepare_reazonspeech_units.py --output-dir ~/speech_ai/data/stage1_reazonspeech

  # テストモード + カスタムディレクトリ
  python prepare_reazonspeech_units.py --test --output-dir ~/speech_ai/data/stage1_test
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/speech_ai"))

try:
    from datasets import load_dataset
    import soundfile as sf
    from tqdm import tqdm
    from speechgpt.utils.speech2unit.speech2unit import Speech2Unit
except ImportError as e:
    print(f"依存ライブラリのインポートエラー: {e}")
    print("以下をインストールしてください:")
    print("  pip install datasets soundfile tqdm")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReazonSpeech データセットをダウンロードして train.txt / valid.txt を生成"
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
        help="テストモード：1時間分のデータのみ処理"
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="small-v1",
        choices=["small-v1", "medium-v1", "large-v1"],
        help="ReazonSpeech のバージョン（デフォルト: small-v1）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # テストモード の場合は小さい値に設定
    if args.test:
        train_hours = 1.0
        valid_hours = 0.0
        print("=" * 60)
        print("テストモード: 1時間分のデータを処理")
        print("=" * 60)
    else:
        train_hours = args.train_hours
        valid_hours = args.valid_hours
        print("=" * 60)
        print(f"フル訓練モード: Train {train_hours:.0f}h / Valid {valid_hours:.0f}h")
        print("=" * 60)

    # 出力ディレクトリを展開・作成
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Speech2Unit ロード（mHuBERT ベース）
    print("Loading Speech2Unit model...")
    s2u = Speech2Unit(ckpt_dir=os.path.expanduser("~/speech_ai/speechgpt/utils/speech2unit/"))

    # ReazonSpeech データセットをロード（ダウンロードモード）
    print(f"Loading ReazonSpeech dataset ({args.dataset}) from HuggingFace...")
    print("  → ダウンロード中...")
    try:
        dataset = load_dataset(
            "reazon-research/reazonspeech",
            args.dataset,
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_duration_hours = 0.0
    train_lines = []
    valid_lines = []
    target_hours = train_hours + valid_hours

    print(f"\nProcessing samples from ReazonSpeech (streaming)...")
    print(f"Target: {target_hours:.0f} hours (Train: {train_hours:.0f}h / Valid: {valid_hours:.0f}h)\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, sample in enumerate(dataset):
            # audio は {"array": ndarray, "sampling_rate": int} 形式
            audio = sample["audio"]
            wav = audio["array"]
            sr = audio["sampling_rate"]

            # 音声の長さ（秒）
            duration_sec = len(wav) / sr
            total_duration_hours += duration_sec / 3600

            temp_wav_path = None
            try:
                # 一時ファイルに保存してから Speech2Unit で処理
                temp_wav_path = Path(temp_dir) / f"temp_{idx}.wav"
                sf.write(str(temp_wav_path), wav, sr)

                # units に変換
                units = s2u(str(temp_wav_path), merged=True)

                # train / valid に分割
                if total_duration_hours <= train_hours:
                    train_lines.append(units)
                elif total_duration_hours <= target_hours:
                    valid_lines.append(units)

                # 進捗表示（100サンプルごと）
                if (idx + 1) % 100 == 0:
                    train_ratio = (len(train_lines) / max(1, idx + 1)) * 100
                    valid_ratio = (len(valid_lines) / max(1, idx + 1)) * 100
                    print(f"  [{idx + 1:6d} samples] {total_duration_hours:7.1f}h | Train: {len(train_lines):6d} | Valid: {len(valid_lines):6d}")

                if total_duration_hours >= target_hours:
                    print(f"  Reached {target_hours:.0f} hours. Stopping...")
                    break

            except Exception as e:
                # エラーは無視して続行
                print(f"  Warning: Failed to process sample {idx}: {e}")
                continue
            finally:
                # 一時ファイルを削除
                if temp_wav_path and temp_wav_path.exists():
                    temp_wav_path.unlink()

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
        if not args.test:
            print("警告: valid.txt にデータがありません")

    # 統計情報
    print(f"\n{'='*60}")
    print(f"✓ Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration_hours:.1f} hours")
    print(f"Train samples: {len(train_lines)}")
    print(f"Valid samples: {len(valid_lines)}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles generated:")
    if train_lines:
        print(f"  - {output_dir / 'train.txt'}")
    if valid_lines:
        print(f"  - {output_dir / 'valid.txt'}")

    # サンプルプレビュー
    print(f"\n{'='*60}")
    print(f"Train.txt プレビュー (最初の3行):")
    print(f"{'='*60}")
    if train_lines:
        with open(output_dir / "train.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 3:
                    preview = line.strip()[:100] + "..." if len(line.strip()) > 100 else line.strip()
                    print(f"{i+1}: {preview}")
                else:
                    break
    else:
        print("(データなし)")


if __name__ == "__main__":
    main()
