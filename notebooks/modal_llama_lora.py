"""
LoRA Fine-Tuning on Llama-2-7B via Modal
=========================================
Runs on Modal A10G GPU (24GB VRAM).
- Base model: meta-llama/Llama-2-7b-hf (fp16 / bfloat16)
- LoRA: r=8, alpha=16, target q_proj + v_proj
- Task: SST-2 sentiment (1000 train / 200 eval)
- Bias eval: CrowS-Pairs SPS + Geometric SPS + Bolukbasi DirectBias

Results are saved to a Modal Volume and also printed to stdout
so you can copy them locally after the run.

Run with:
    modal run notebooks/modal_llama_lora.py
"""

import modal
import os

# ── Modal app setup ────────────────────────────────────────────────────────────
app = modal.App("llama-lora-bias")

# Persistent volume to store adapter weights + results
volume = modal.Volume.from_name("nlp-results", create_if_missing=True)
VOLUME_PATH = "/results"

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy==1.26.4",
        "torch==2.1.2",
        "transformers==4.40.0",
        "peft==0.10.0",
        "datasets==2.19.0",
        "accelerate==0.29.3",
        "bitsandbytes==0.43.1",
        "scipy",
        "scikit-learn",
        "tqdm",
        "pandas",
        "evaluate",
    ])
)

# ── Training + Eval function ───────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={VOLUME_PATH: volume},
    timeout=7200,
    memory=32768,
)
def run_lora():
    import torch
    import json
    import re
    import csv
    import io
    import urllib.request
    import numpy as np
    from tqdm import tqdm
    from sklearn.decomposition import PCA
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    HF_TOKEN  = os.environ["HF_TOKEN"]
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DEVICE     = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Helpers ──────────────────────────────────────────────────────────────
    GENDER_PAIRS = [
        ("he","she"),("him","her"),("his","hers"),("man","woman"),
        ("men","women"),("boy","girl"),("male","female"),
        ("father","mother"),("son","daughter"),("brother","sister"),
        ("husband","wife"),("king","queen"),
    ]
    CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

    def get_repr(model, tokenizer, word):
        inputs = tokenizer(" " + word, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[-1][0].mean(0).cpu().float().numpy()

    def compute_gender_dir(model, tokenizer):
        diffs = [get_repr(model, tokenizer, m) - get_repr(model, tokenizer, f)
                 for m, f in GENDER_PAIRS]
        pca = PCA(n_components=1)
        pca.fit(np.array(diffs))
        return pca.components_[0]

    def direct_bias(vecs, g):
        g_n = g / (np.linalg.norm(g) + 1e-8)
        return float(np.mean([abs(np.dot(v / (np.linalg.norm(v)+1e-8), g_n)) for v in vecs]))

    def score_sentence(model, tokenizer, sentence):
        inputs = tokenizer(sentence, return_tensors="pt",
                           truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        return -loss.item()

    def get_diff_words(s1, s2):
        def clean(s): return set(re.sub(r"[^\w\s]","",s.lower()).split())
        a, b = clean(s1), clean(s2)
        return list(a - b), list(b - a)

    def load_crows():
        with urllib.request.urlopen(CROWS_URL) as r:
            content = r.read().decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(content)))
        return [r for r in rows if r["bias_type"] == "gender"]

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Baseline eval ─────────────────────────────────────────────────────────
    print("\nLoading baseline Llama-2-7B (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()

    # SST-2 baseline accuracy
    print("Evaluating SST-2 baseline...")
    sst = load_dataset("glue", "sst2")
    eval_data = sst["validation"].select(range(200))
    pos_token = tokenizer(" positive", add_special_tokens=False).input_ids[-1]
    neg_token = tokenizer(" negative", add_special_tokens=False).input_ids[-1]

    def sst2_accuracy(mdl):
        correct = 0
        for item in tqdm(eval_data, desc="  SST-2 eval", leave=False):
            prompt = f"Sentiment of \"{item['sentence']}\":\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = mdl(**inputs).logits[0, -1, :]
            pred = 1 if logits[pos_token] > logits[neg_token] else 0
            correct += int(pred == item["label"])
        return correct / len(eval_data)

    baseline_sst2 = sst2_accuracy(model)
    print(f"Baseline SST-2 accuracy: {baseline_sst2:.3f}")

    # CrowS-Pairs baseline
    print("Evaluating CrowS-Pairs baseline...")
    crows = load_crows()
    def crows_sps(mdl):
        stereo_wins = 0
        for row in tqdm(crows, desc="  CrowS SPS", leave=False):
            s_more = score_sentence(mdl, tokenizer, row["sent_more"])
            s_less = score_sentence(mdl, tokenizer, row["sent_less"])
            if s_more > s_less:
                stereo_wins += 1
        return stereo_wins / len(crows) * 100

    baseline_sps = crows_sps(model)
    print(f"Baseline SPS: {baseline_sps:.1f}%")

    # Bolukbasi baseline
    print("Computing baseline Bolukbasi DirectBias...")
    g_baseline = compute_gender_dir(model, tokenizer)
    stereo_vecs, anti_vecs = [], []
    for row in tqdm(crows, desc="  DirectBias", leave=False):
        dm, dl = get_diff_words(row["sent_more"], row["sent_less"])
        for w in dm: stereo_vecs.append(get_repr(model, tokenizer, w))
        for w in dl: anti_vecs.append(get_repr(model, tokenizer, w))
    baseline_db_stereo = direct_bias(stereo_vecs, g_baseline)
    baseline_db_anti   = direct_bias(anti_vecs,   g_baseline)
    print(f"Baseline DirectBias stereo={baseline_db_stereo:.4f} anti={baseline_db_anti:.4f}")

    del model
    torch.cuda.empty_cache()

    # ── LoRA fine-tuning ──────────────────────────────────────────────────────
    print("\nLoading Llama-2-7B for LoRA training...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # SST-2 training data
    train_data = sst["train"].select(range(1000))
    def tokenize(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for sentence, label in zip(batch["sentence"], batch["label"]):
            prompt  = f"Sentiment of \"{sentence}\":\n"
            answer  = " positive" if label == 1 else " negative"
            full    = prompt + answer
            prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
            full_enc   = tokenizer(full, truncation=True, max_length=128,
                                   padding="max_length", add_special_tokens=True)
            input_ids  = full_enc["input_ids"]
            attn_mask  = full_enc["attention_mask"]
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            labels = [l if input_ids[i] != tokenizer.pad_token_id else -100
                      for i, l in enumerate(labels)]
            labels = labels[:128] + [-100] * (128 - len(labels[:128]))
            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attn_mask)
            out["labels"].append(labels)
        return out

    train_tok = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
    train_tok.set_format("torch")

    adapter_path = os.path.join(VOLUME_PATH, "lora_adapter_llama")
    args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    print("\nTraining LoRA...")
    trainer.train()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    volume.commit()
    print(f"Adapter saved to {adapter_path}")

    # ── Post-LoRA eval ────────────────────────────────────────────────────────
    print("\nMerging LoRA and evaluating...")
    merged = model.merge_and_unload()
    merged.eval()

    lora_sst2 = sst2_accuracy(merged)
    print(f"Post-LoRA SST-2 accuracy: {lora_sst2:.3f}")

    lora_sps = crows_sps(merged)
    print(f"Post-LoRA SPS: {lora_sps:.1f}%")

    g_lora = compute_gender_dir(merged, tokenizer)
    stereo_vecs, anti_vecs = [], []
    for row in tqdm(crows, desc="  DirectBias LoRA", leave=False):
        dm, dl = get_diff_words(row["sent_more"], row["sent_less"])
        for w in dm: stereo_vecs.append(get_repr(merged, tokenizer, w))
        for w in dl: anti_vecs.append(get_repr(merged, tokenizer, w))
    lora_db_stereo = direct_bias(stereo_vecs, g_lora)
    lora_db_anti   = direct_bias(anti_vecs,   g_lora)
    print(f"Post-LoRA DirectBias stereo={lora_db_stereo:.4f} anti={lora_db_anti:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model": MODEL_NAME,
        "method": "LoRA",
        "lora_config": {"r": 8, "alpha": 16, "target": "q_proj,v_proj"},
        "baseline": {
            "sst2_accuracy": round(baseline_sst2, 4),
            "crows_sps": round(baseline_sps, 2),
            "direct_bias_stereo": round(baseline_db_stereo, 4),
            "direct_bias_anti": round(baseline_db_anti, 4),
        },
        "post_lora": {
            "sst2_accuracy": round(lora_sst2, 4),
            "crows_sps": round(lora_sps, 2),
            "direct_bias_stereo": round(lora_db_stereo, 4),
            "direct_bias_anti": round(lora_db_anti, 4),
        },
    }

    out_path = os.path.join(VOLUME_PATH, "llama_lora_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))
    return results


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    result = run_lora.remote()
    print("\nDone. Results:")
    import json
    print(json.dumps(result, indent=2))
