import os
from pathlib import Path

jvs_root = Path("/mnt/datasets/jvs_ver1")
output_dir = Path(os.path.expanduser("~/speech_ai/data/stage1_trial"))
output_dir.mkdir(parents=True, exist_ok=True)

# 最初の5人の話者を抽出
speakers = sorted([d for d in jvs_root.glob("jvs*")])[:5]
print(f"Processing {len(speakers)} speakers...")

with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
    total_lines = 0
    
    for speaker_dir in speakers:
        # parallel100 セットのテキストファイル
        # JVS Corpus 構造: jvs001/parallel100/transcripts_utf8.txt
        transcript_path = speaker_dir / "parallel100" / "transcripts_utf8.txt"
        
        if transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as tf:
                transcripts = [line.strip() for line in tf.readlines()]
            
            print(f"  {speaker_dir.name}: Found {len(transcripts)} transcripts")
            
            # テキストを出力（最初の100件）
            for transcript in transcripts[:100]:
                if transcript:  # 空行をスキップ
                    f.write(f"{transcript}\n")
                    total_lines += 1
        else:
            print(f"  {speaker_dir.name}: transcript not found at {transcript_path}")

print(f"\n✓ Created {total_lines} lines in {output_dir / 'train.txt'}")

# データ確認
print(f"\n=== Train.txt プレビュー ===")
with open(output_dir / "train.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"{i+1}: {line.strip()}")
        else:
            break