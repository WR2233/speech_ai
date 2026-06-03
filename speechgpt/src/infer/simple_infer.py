import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="こんにちは")
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args()

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

if __name__ == "__main__":
    main()
