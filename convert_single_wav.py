"""
単一の .wav ファイルを音声ユニットに変換するスクリプト
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/speech_ai"))

from speechgpt.utils.speech2unit.speech2unit import Speech2Unit


def main():
    parser = argparse.ArgumentParser(description="Convert single WAV file to speech units")
    parser.add_argument("--input", type=str, required=True, help="Input WAV file path")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: input.txt)")
    args = parser.parse_args()

    input_file = Path(os.path.expanduser(args.input))

    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        sys.exit(1)

    if not input_file.suffix.lower() == ".wav":
        print(f"✗ Input must be a .wav file: {input_file}")
        sys.exit(1)

    # 出力ファイル決定
    if args.output:
        output_file = Path(os.path.expanduser(args.output))
    else:
        output_file = input_file.with_suffix(".txt")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"\nLoading Speech2Unit...")

    try:
        # Speech2Unit ロード
        s2u = Speech2Unit(ckpt_dir=os.path.expanduser("~/speech_ai/speechgpt/utils/speech2unit/"))

        # 変換
        print(f"Converting {input_file.name}...")
        units = s2u(str(input_file), merged=True)

        # 出力
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{units}\n")

        print(f"\n✓ Conversion complete!")
        print(f"  Output: {output_file}")
        print(f"  Units: {len(units.split())} tokens")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
