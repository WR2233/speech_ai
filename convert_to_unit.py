
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/speech_ai"))

from speechgpt.utils.speech2unit.speech2unit import Speech2Unit

jvs_root = Path("~/datasets/jvs_ver1")
output_dir = Path(os.path.expanduser("~/speech_ai/data/stage1_trial"))
output_dir.mkdir(parents=True, exist_ok=True)

# Speech2Unit ロード
s2u = Speech2Unit(ckpt_dir=os.path.expanduser("~/speech_ai/speechgpt/utils/speech2unit/"))

speakers = sorted([d for d in jvs_root.glob("jvs*")])  # 全話者
print(f"Converting {len(speakers)} speakers to units...")

with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
    total_lines = 0

    for speaker_dir in speakers:
        print(f"  {speaker_dir.name}...", end=" ")
        speaker_count = 0

        # 複数の subset を処理（parallel100 と nonpara30）
        for subset in ["parallel100", "nonpara30"]:
            wav_dir = speaker_dir / subset / "wav24kHz16bit"
            if not wav_dir.exists():
                continue

            for wav_file in sorted(wav_dir.glob("*.wav")):  # 全ファイル
                try:
                    # 音声→ユニット列変換
                    units = s2u(str(wav_file), merged=True)
                    f.write(f"{units}\n")
                    total_lines += 1
                    speaker_count += 1
                except Exception as e:
                    pass  # エラーは無視して続行

        print(f"({speaker_count} files)")

print(f"\n✓ Conversion complete!")
print(f"  Total lines: {total_lines}")
print(f"  Output: {output_dir / 'train.txt'}")

# データプレビュー
print(f"\n=== Train.txt プレビュー ===")
with open(output_dir / "train.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 3:
            print(f"{i+1}: {line.strip()[:80]}...")
        else:
            break