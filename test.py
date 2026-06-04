from datasets import load_dataset
import soundfile as sf
from pathlib import Path

dataset = load_dataset(
    "reazon-research/reazonspeech",
    "small-v1",
    trust_remote_code=True
)

print(f"Dataset type: {type(dataset)}")
print(f"Dataset keys: {dataset.keys()}\n")

# dataset['train'] でアクセス
train_data = dataset['train']
print(f"Train dataset: {train_data}\n")

# 10サンプルをダウンロード
output_dir = Path("/Users/ryuhei/speech_ai/data/reazonspeech_wav")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading 10 samples to {output_dir}\n")

for idx, sample in enumerate(train_data):
    if idx >= 10:
        break

    try:
        # オーディオを保存
        audio = sample['audio']['array']
        sr = sample['audio']['sampling_rate']

        output_file = output_dir / f"{idx:06d}.wav"
        sf.write(str(output_file), audio, sr)

        duration = len(audio) / sr
        print(f"[{idx}] Saved: {output_file.name} ({duration:.2f}s)")

    except Exception as e:
        print(f"[{idx}] Error: {e}")

print(f"\n✓ Done! Files saved to {output_dir}")