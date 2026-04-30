"""
Run LoRA + QLoRA on Llama-2-7B in parallel via Modal
=====================================================
Spawns both fine-tuning jobs simultaneously on separate A10G GPUs.
Total cost: ~$2-2.50  |  Total wall time: ~60 min

Run with:
    modal run notebooks/modal_llama_all.py
"""

import modal
import os

# Re-use the same volume and image across both jobs
volume = modal.Volume.from_name("nlp-results", create_if_missing=True)
VOLUME_PATH = "/results"

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

app = modal.App("llama-lora-qlora-parallel")

SECRET = modal.Secret.from_name("huggingface-token")


# ── Shared helpers (injected into each container) ─────────────────────────────
def _helpers():
    import re, csv, io, urllib.request
    import torch, numpy as np
    from tqdm import tqdm
    from sklearn.decomposition import PCA

    GENDER_PAIRS = [
        ("he","she"),("him","her"),("his","hers"),("man","woman"),
        ("men","women"),("boy","girl"),("male","female"),
        ("father","mother"),("son","daughter"),("brother","sister"),
        ("husband","wife"),("king","queen"),
    ]
    CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

    def get_repr(model, tokenizer, word):
        inputs = tokenizer(" " + word, return_tensors="pt").to("cuda")
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
                           truncation=True, max_length=256).to("cuda")
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

    def eval_bias(model, tokenizer, label):
        crows = load_crows()
        model.eval()
        stereo_wins = 0
        for row in tqdm(crows, desc=f"  [{label}] CrowS SPS", leave=False):
            if score_sentence(model, tokenizer, row["sent_more"]) > \
               score_sentence(model, tokenizer, row["sent_less"]):
                stereo_wins += 1
        sps = stereo_wins / len(crows) * 100

        g = compute_gender_dir(model, tokenizer)
        sv, av = [], []
        for row in tqdm(crows, desc=f"  [{label}] DirectBias", leave=False):
            dm, dl = get_diff_words(row["sent_more"], row["sent_less"])
            for w in dm: sv.append(get_repr(model, tokenizer, w))
            for w in dl: av.append(get_repr(model, tokenizer, w))
        return {"crows_sps": round(sps, 2),
                "direct_bias_stereo": round(direct_bias(sv, g), 4),
                "direct_bias_anti":   round(direct_bias(av, g), 4)}

    def sst2_accuracy(model, tokenizer, eval_data, pos_token, neg_token):
        correct = 0
        for item in tqdm(eval_data, desc="  SST-2 eval", leave=False):
            prompt = f"Sentiment of \"{item['sentence']}\":\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            pred = 1 if logits[pos_token] > logits[neg_token] else 0
            correct += int(pred == item["label"])
        return correct / len(eval_data)

    return get_repr, compute_gender_dir, direct_bias, score_sentence, \
           get_diff_words, load_crows, eval_bias, sst2_accuracy


# ── LoRA job ──────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    secrets=[SECRET],
    volumes={VOLUME_PATH: volume},
    timeout=7200,
    memory=32768,
)
def run_lora():
    import torch, json
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               TrainingArguments, Trainer, DataCollatorWithPadding)
    from peft import LoraConfig, get_peft_model, TaskType

    _, _, _, _, _, _, eval_bias, sst2_accuracy = _helpers()

    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    HF_TOKEN   = os.environ["HF_TOKEN"]
    print(f"[LoRA] GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sst       = load_dataset("glue", "sst2")
    eval_data = sst["validation"].select(range(200))
    pos_token = tokenizer(" positive", add_special_tokens=False).input_ids[-1]
    neg_token = tokenizer(" negative", add_special_tokens=False).input_ids[-1]

    # Baseline
    print("[LoRA] Loading baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    model.eval()
    baseline_sst2  = sst2_accuracy(model, tokenizer, eval_data, pos_token, neg_token)
    baseline_bias  = eval_bias(model, tokenizer, "baseline")
    print(f"[LoRA] Baseline SST-2={baseline_sst2:.3f}  SPS={baseline_bias['crows_sps']:.1f}%")
    del model; torch.cuda.empty_cache()

    # Fine-tune
    print("[LoRA] Fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

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
            # Mask prompt tokens with -100 so loss only applies to the answer
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            labels = [l if input_ids[i] != tokenizer.pad_token_id else -100
                      for i, l in enumerate(labels)]
            labels = labels[:128] + [-100] * (128 - len(labels[:128]))
            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attn_mask)
            out["labels"].append(labels)
        return out

    train_tok = sst["train"].select(range(1000)).map(
        tokenize, batched=True, remove_columns=["sentence","label","idx"])
    train_tok.set_format("torch")

    adapter_path = os.path.join(VOLUME_PATH, "lora_adapter_llama")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=adapter_path, num_train_epochs=2,
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            learning_rate=2e-4, bf16=True, logging_steps=50,
            save_strategy="no", report_to="none", gradient_checkpointing=True),
        train_dataset=train_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    volume.commit()

    # Post-LoRA eval
    merged = model.merge_and_unload()
    lora_sst2 = sst2_accuracy(merged, tokenizer, eval_data, pos_token, neg_token)
    lora_bias = eval_bias(merged, tokenizer, "post-lora")
    print(f"[LoRA] Post-LoRA SST-2={lora_sst2:.3f}  SPS={lora_bias['crows_sps']:.1f}%")

    results = {
        "model": MODEL_NAME, "method": "LoRA",
        "baseline": {"sst2_accuracy": round(baseline_sst2,4), **baseline_bias},
        "post_lora": {"sst2_accuracy": round(lora_sst2,4), **lora_bias},
    }
    with open(os.path.join(VOLUME_PATH, "llama_lora_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print("[LoRA] Done.", json.dumps(results, indent=2))
    return results


# ── QLoRA job ─────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    secrets=[SECRET],
    volumes={VOLUME_PATH: volume},
    timeout=7200,
    memory=32768,
)
def run_qlora():
    import torch, json
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               BitsAndBytesConfig, TrainingArguments,
                               Trainer, DataCollatorWithPadding)
    from peft import (LoraConfig, get_peft_model, TaskType,
                      prepare_model_for_kbit_training)

    _, _, _, _, _, _, eval_bias, sst2_accuracy = _helpers()

    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    HF_TOKEN   = os.environ["HF_TOKEN"]
    print(f"[QLoRA] GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    sst       = load_dataset("glue", "sst2")
    eval_data = sst["validation"].select(range(200))
    pos_token = tokenizer(" positive", add_special_tokens=False).input_ids[-1]
    neg_token = tokenizer(" negative", add_special_tokens=False).input_ids[-1]

    # Baseline (4-bit, no adapter)
    print("[QLoRA] Loading baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", token=HF_TOKEN)
    model.eval()
    baseline_sst2 = sst2_accuracy(model, tokenizer, eval_data, pos_token, neg_token)
    baseline_bias = eval_bias(model, tokenizer, "baseline")
    print(f"[QLoRA] Baseline SST-2={baseline_sst2:.3f}  SPS={baseline_bias['crows_sps']:.1f}%")

    # Fine-tune
    print("[QLoRA] Fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

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

    train_tok = sst["train"].select(range(1000)).map(
        tokenize, batched=True, remove_columns=["sentence","label","idx"])
    train_tok.set_format("torch")

    adapter_path = os.path.join(VOLUME_PATH, "qlora_adapter_llama")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=adapter_path, num_train_epochs=2,
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            learning_rate=2e-4, bf16=False, logging_steps=50,
            save_strategy="no", report_to="none", gradient_checkpointing=True,
            optim="paged_adamw_8bit"),
        train_dataset=train_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    volume.commit()

    # Post-QLoRA eval (can't merge 4-bit)
    model.eval()
    qlora_sst2 = sst2_accuracy(model, tokenizer, eval_data, pos_token, neg_token)
    qlora_bias = eval_bias(model, tokenizer, "post-qlora")
    print(f"[QLoRA] Post-QLoRA SST-2={qlora_sst2:.3f}  SPS={qlora_bias['crows_sps']:.1f}%")

    results = {
        "model": MODEL_NAME, "method": "QLoRA",
        "baseline": {"sst2_accuracy": round(baseline_sst2,4), **baseline_bias},
        "post_qlora": {"sst2_accuracy": round(qlora_sst2,4), **qlora_bias},
    }
    with open(os.path.join(VOLUME_PATH, "llama_qlora_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print("[QLoRA] Done.", json.dumps(results, indent=2))
    return results


# ── Local entrypoint — spawns both in parallel ────────────────────────────────
@app.local_entrypoint()
def main():
    import json
    print("Spawning LoRA and QLoRA jobs in parallel...")

    # .spawn() launches without blocking; .get() waits for the result
    lora_handle  = run_lora.spawn()
    qlora_handle = run_qlora.spawn()

    print("Both jobs running. Waiting for results...\n")
    lora_results  = lora_handle.get()
    qlora_results = qlora_handle.get()

    print("\n===== LoRA Results =====")
    print(json.dumps(lora_results, indent=2))
    print("\n===== QLoRA Results =====")
    print(json.dumps(qlora_results, indent=2))
