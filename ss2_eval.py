"""
SST-2 Sentiment Classification — Zero-shot & Few-shot
======================================================
Evaluates LLaMA-2 7B (4-bit quantized) on SST-2 using two prompting
strategies described in Sections 3.2.1 and 3.2.2 of the paper.

Instead of free-form generation, we sum the softmax probabilities of
multiple token variations (e.g. "Positive", "positive", "Pos") for each
class and compare — this handles tokenizer variations robustly.

Usage
-----
    python sst2_eval.py                                    # defaults
    python sst2_eval.py --model meta-llama/Llama-2-7b-hf  # explicit
    python sst2_eval.py --no_quantize                      # fp16 (needs ≥16 GB)
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm


# ── Prompt templates (Section 3.2) ──────────────────────────────────

ZERO_SHOT_TEMPLATE = "Review: {text}\nSentiment:"

FEW_SHOT_TEMPLATE = (
    "Review: The movie was fantastic.\n"
    "Sentiment: Positive\n\n"
    "Review: The plot was boring.\n"
    "Sentiment: Negative\n\n"
    "Review: {text}\n"
    "Sentiment:"
)

LABEL_MAP = {0: "Negative", 1: "Positive"}

POSITIVE_VARIANTS = [" Positive", " positive", " Pos", " pos"]
NEGATIVE_VARIANTS = [" Negative", " negative", " Neg", " neg"]


# ── Core evaluation logic ───────────────────────────────────────────

def get_label_token_ids(tokenizer):
    """Resolve token IDs for all positive/negative surface forms.

    Returns two lists of token IDs — one per class.  We take the last
    token of each encoding because the tokenizer may split a word into
    multiple sub-tokens (the last one is the disambiguating piece).
    """
    pos_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in POSITIVE_VARIANTS]
    neg_ids = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in NEGATIVE_VARIANTS]
    pos_ids = list(set(pos_ids))
    neg_ids = list(set(neg_ids))
    return pos_ids, neg_ids


def predict(model, tokenizer, prompt, pos_ids, neg_ids):
    """Return 'Positive' or 'Negative' by summing softmax probs."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
    pos_prob = probs[0, pos_ids].sum().item()
    neg_prob = probs[0, neg_ids].sum().item()

    return "Positive" if pos_prob >= neg_prob else "Negative"


def evaluate(model, tokenizer, dataset, template, pos_ids, neg_ids, desc):
    """Run evaluation over the full dataset with a given prompt template."""
    correct = 0

    for item in tqdm(dataset, desc=desc):
        prompt = template.format(text=item["sentence"])
        prediction = predict(model, tokenizer, prompt, pos_ids, neg_ids)
        gold = LABEL_MAP[item["label"]]

        if prediction == gold:
            correct += 1

    accuracy = correct / len(dataset) * 100
    return correct, len(dataset), accuracy


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SST-2 zero-shot & few-shot evaluation"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no_quantize", action="store_true",
        help="Disable 4-bit quantization (needs ≥16 GB VRAM)",
    )
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading model: {args.model}")

    load_kwargs = {
        "device_map": "auto",
        "token": args.token,
    }

    if not args.no_quantize:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("      Using 4-bit quantization (bitsandbytes)")
    else:
        load_kwargs["torch_dtype"] = torch.float16
        print("      Using fp16 (no quantization)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"      Model ready\n")

    # ── Load SST-2 ──────────────────────────────────────────────────
    print("[2/4] Loading SST-2 validation set ...")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"      {len(dataset)} examples\n")

    # ── Resolve label tokens ────────────────────────────────────────
    pos_ids, neg_ids = get_label_token_ids(tokenizer)
    print(f"[3/4] Label token IDs:")
    for tid in pos_ids:
        print(f"        Positive: {tid} → {repr(tokenizer.decode([tid]))}")
    for tid in neg_ids:
        print(f"        Negative: {tid} → {repr(tokenizer.decode([tid]))}")
    print()

    # ── Zero-shot ───────────────────────────────────────────────────
    print("[4/4] Evaluating ...\n")
    zs_correct, zs_total, zs_acc = evaluate(
        model, tokenizer, dataset,
        ZERO_SHOT_TEMPLATE, pos_ids, neg_ids,
        desc="Zero-shot",
    )

    # ── Few-shot ────────────────────────────────────────────────────
    fs_correct, fs_total, fs_acc = evaluate(
        model, tokenizer, dataset,
        FEW_SHOT_TEMPLATE, pos_ids, neg_ids,
        desc="Few-shot ",
    )

    # ── Report ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  SST-2 Sentiment Classification Results")
    print(f"{'=' * 55}")
    print(f"  Model : {args.model}")
    print(f"  Quant : {'4-bit' if not args.no_quantize else 'fp16'}")
    print(f"  Split : validation ({zs_total} examples)")
    print(f"{'=' * 55}")
    print(f"  {'Method':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*7} {'-'*10}")
    print(f"  {'Zero-shot':<12} {zs_correct:>8} {zs_total:>7} {zs_acc:>9.2f}%")
    print(f"  {'Few-shot':<12} {fs_correct:>8} {fs_total:>7} {fs_acc:>9.2f}%")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
