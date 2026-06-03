import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
import re
import soundfile as sf
import numpy as np
from datetime import datetime
import subprocess
import sys

def extract_speech_units(text):
    """テキストから音声ユニット（<0>-<999>）を抽出"""
    units = re.findall(r'<(\d+)>', text)
    return [int(u) for u in units if 0 <= int(u) <= 999]

def units_to_wav_with_subprocess(units, output_path, vocoder_script="speechgpt/src/infer/fairseq_generate_wav.py"):
    """subprocess で fairseq 環境を使って音声を生成"""
    if not units:
        print(f"  No speech units found")
        return False

    if not os.path.exists(vocoder_script):
        print(f"  ERROR: Vocoder script not found: {vocoder_script}")
        return False

    try:
        # ユニットを JSON 形式で準備
        units_json = json.dumps(units)

        # fairseq 環境で fairseq_generate_wav.py を実行
        print(f"  Generating WAV with fairseq (subprocess)...")
        cmd = [
            "bash", "-l", "-c",
            f"conda activate fairseq_env && python {vocoder_script} --units '{units_json}' --output '{output_path}'"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ERROR: fairseq subprocess failed")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False

        print(f"  Speech units: {len(units)} units")
        print(f"  WAV file saved: {output_path}")
        return True

    except Exception as e:
        print(f"  ERROR: Failed to generate WAV: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="こんにちは")
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="inference_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデル・トークナイザー読み込み
    print(f"Loading model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0

    model.eval()

    # 生成
    print(f"Generating from prompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            top_p=0.9,
            temperature=0.7,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")

    # 音声ユニットを抽出
    speech_units = extract_speech_units(generated_text)

    # やり取りをファイルに保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"inference_log_{timestamp}.json")

    result = {
        "timestamp": timestamp,
        "model": args.model_name_or_path,
        "input_prompt": args.prompt,
        "full_output": generated_text,
        "speech_units": speech_units,
        "num_units": len(speech_units)
    }

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLog saved: {log_file}")

    # 音声ユニットがあれば .wav を出力（subprocess で fairseq を使用）
    if speech_units:
        print("\nGenerating WAV file (using fairseq subprocess)...")
        wav_file = os.path.join(args.output_dir, f"output_{timestamp}.wav")
        success = units_to_wav_with_subprocess(speech_units, wav_file)
        if not success:
            print("  Note: WAV generation failed. Try running fairseq_env separately.")
            print(f"  conda activate fairseq_env")
            print(f"  python speechgpt/src/infer/fairseq_generate_wav.py --units '{speech_units}' --output {wav_file}")
    else:
        print("\nNo speech units generated")

if __name__ == "__main__":
    main()
