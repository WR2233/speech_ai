import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
import re
import soundfile as sf
import numpy as np
from datetime import datetime

def extract_speech_units(text):
    """テキストから音声ユニット（<0>-<999>）を抽出"""
    units = re.findall(r'<(\d+)>', text)
    return [int(u) for u in units if 0 <= int(u) <= 999]

def units_to_wav(units, output_path, sr=16000):
    """音声ユニットから簡単な音声波形を生成して保存

    注: 本格的な音声生成には fairseq の HiFiGAN が必要です
    ここでは、ユニット値を利用した簡易的な波形を生成します
    """
    if not units:
        print(f"  No speech units found")
        return False

    # ユニット値を正規化して振幅に変換
    units_array = np.array(units, dtype=np.float32)
    # 0-999 を -1～1 に正規化
    normalized = (units_array / 500.0) - 1.0
    # クリッピング
    normalized = np.clip(normalized, -1.0, 1.0)

    # 波形を保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, normalized, sr)
    print(f"  Speech units: {len(units)} units")
    print(f"  WAV file saved: {output_path}")
    return True

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

    # 音声ユニットがあれば .wav を出力
    if speech_units:
        print("\nGenerating WAV file...")
        wav_file = os.path.join(args.output_dir, f"output_{timestamp}.wav")
        units_to_wav(speech_units, wav_file)
    else:
        print("\nNo speech units generated")

if __name__ == "__main__":
    main()
