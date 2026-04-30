"""
Bolukbasi DirectBias / IndirectBias on CrowS-Pairs (gender subset)
===================================================================
Applies the Bolukbasi et al. (2016) geometric bias metrics directly
to words extracted from the CrowS-Pairs dataset, rather than a fixed
profession word list.

For each (sent_more, sent_less) gender pair:
  - diff_more : words in sent_more but not sent_less  (stereotypical words)
  - diff_less : words in sent_less but not sent_more  (anti-stereotypical words)

Metrics:
  DirectBias(stereo)    = mean |cos(w, g)|   over all stereo words
  DirectBias(anti)      = mean |cos(w, g)|   over all anti-stereo words
  IndirectBias(w, v)    = [cos(w,v) - cos(w⊥, v⊥)] / cos(w,v)
                          computed for each (stereo_word, anti_stereo_word) pair

Models: Baseline (fp16) | Post-LoRA (merged) | Post-QLoRA (4-bit NF4 + adapter)
"""

import os, sys, re, json, csv, io
import urllib.request
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(ROOT_DIR, "results", "processed", "opt_1.3b")
FIGURES_DIR = os.path.join(ROOT_DIR, "results", "figures", "exploratory")
ADAPTERS_DIR = os.path.join(ROOT_DIR, "artifacts", "adapters", "opt_1.3b")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "facebook/opt-1.3b"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load CrowS-Pairs gender subset ───────────────────────────────────────────
CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
print("Downloading CrowS-Pairs...")
with urllib.request.urlopen(CROWS_URL) as r:
    content = r.read().decode("utf-8")
all_rows     = list(csv.DictReader(io.StringIO(content)))
crows_gender = [row for row in all_rows if row["bias_type"] == "gender"]
print(f"Gender subset: {len(crows_gender)} pairs")

# ── Gender direction pairs (same as geometric_bias.py) ───────────────────────
GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"), ("man", "woman"),
    ("men", "women"), ("boy", "girl"), ("male", "female"),
    ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
    ("husband", "wife"), ("king", "queen"),
]


# ── Core helpers ─────────────────────────────────────────────────────────────
def get_repr(model, tokenizer, word):
    """Last-layer hidden state averaged over token positions."""
    inputs = tokenizer(" " + word, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0].mean(0).cpu().float().numpy()


def compute_gender_direction(model, tokenizer):
    """First PC of (male_repr − female_repr) difference vectors."""
    diffs = []
    for m_word, f_word in tqdm(GENDER_PAIRS, desc="  Gender direction", leave=False):
        diffs.append(get_repr(model, tokenizer, m_word) -
                     get_repr(model, tokenizer, f_word))
    pca = PCA(n_components=1)
    pca.fit(np.array(diffs))
    return pca.components_[0]


def direct_bias_score(vecs, g):
    """mean |cos(w, g)| over a list of vectors."""
    g_n = g / (np.linalg.norm(g) + 1e-8)
    scores = []
    for v in vecs:
        v_n = v / (np.linalg.norm(v) + 1e-8)
        scores.append(abs(float(np.dot(v_n, g_n))))
    return float(np.mean(scores)) if scores else 0.0


def indirect_bias(w_vec, v_vec, g):
    """[cos(w,v) - cos(w⊥, v⊥)] / cos(w,v)"""
    g_n = g / (np.linalg.norm(g) + 1e-8)
    w_n = w_vec / (np.linalg.norm(w_vec) + 1e-8)
    v_n = v_vec / (np.linalg.norm(v_vec) + 1e-8)
    cos_wv = float(np.dot(w_n, v_n))
    if abs(cos_wv) < 1e-8:
        return 0.0
    w_perp = w_n - np.dot(w_n, g_n) * g_n
    v_perp = v_n - np.dot(v_n, g_n) * g_n
    w_perp /= np.linalg.norm(w_perp) + 1e-8
    v_perp /= np.linalg.norm(v_perp) + 1e-8
    cos_perp = float(np.dot(w_perp, v_perp))
    return (cos_wv - cos_perp) / cos_wv


def get_diff_words(sent_more, sent_less):
    """Word-level set difference after punctuation stripping."""
    def clean(s):
        return set(re.sub(r"[^\w\s]", "", s.lower()).split())
    w_more, w_less = clean(sent_more), clean(sent_less)
    return list(w_more - w_less), list(w_less - w_more)


# ── Main analysis ─────────────────────────────────────────────────────────────
def analyse_model(model, tokenizer, label):
    model.eval()
    print(f"\n[{label}] Computing gender direction...")
    g = compute_gender_direction(model, tokenizer)

    stereo_vecs  = []   # all diff_more word vectors across all pairs
    anti_vecs    = []   # all diff_less word vectors across all pairs
    ib_scores    = []   # IndirectBias for each valid (stereo_word, anti_word) combo
    n_valid      = 0    # pairs where both sides have diff words

    print(f"[{label}] Processing {len(crows_gender)} CrowS-Pairs pairs...")
    for row in tqdm(crows_gender, desc="  Pairs", leave=False):
        diff_more, diff_less = get_diff_words(row["sent_more"], row["sent_less"])
        if not diff_more or not diff_less:
            continue
        n_valid += 1

        vecs_more = [get_repr(model, tokenizer, w) for w in diff_more]
        vecs_less = [get_repr(model, tokenizer, w) for w in diff_less]
        stereo_vecs.extend(vecs_more)
        anti_vecs.extend(vecs_less)

        # IndirectBias: all (stereo, anti) cross-pairs for this sentence pair
        for vm in vecs_more:
            for vl in vecs_less:
                ib_scores.append(indirect_bias(vm, vl, g))

    db_stereo = direct_bias_score(stereo_vecs, g)
    db_anti   = direct_bias_score(anti_vecs,   g)
    ib_mean   = float(np.mean(ib_scores))   if ib_scores  else 0.0
    ib_std    = float(np.std(ib_scores))    if ib_scores  else 0.0
    ib_pos    = float(np.mean([s > 0 for s in ib_scores])) if ib_scores else 0.0

    result = {
        "direct_bias_stereo":      round(db_stereo, 4),
        "direct_bias_anti":        round(db_anti,   4),
        "direct_bias_delta":       round(db_stereo - db_anti, 4),
        "indirect_bias_mean":      round(ib_mean,   4),
        "indirect_bias_std":       round(ib_std,    4),
        "indirect_bias_pct_pos":   round(ib_pos * 100, 1),
        "n_valid_pairs":           n_valid,
        "n_stereo_words":          len(stereo_vecs),
        "n_anti_words":            len(anti_vecs),
    }
    print(f"  DirectBias (stereo words): {db_stereo:.4f}")
    print(f"  DirectBias (anti words):   {db_anti:.4f}  (delta={db_stereo-db_anti:+.4f})")
    print(f"  IndirectBias mean={ib_mean:.4f}  std={ib_std:.4f}  pct_positive={ib_pos*100:.1f}%")
    return result


# ── Load helpers ──────────────────────────────────────────────────────────────
def load_baseline():
    print("\nLoading baseline OPT-1.3B (fp16)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return m, tok


def load_lora(adapter_path):
    print("\nLoading post-LoRA (merged)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    m = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    return m, tok


def load_qlora(adapter_path):
    print("\nLoading post-QLoRA (4-bit NF4 + adapter)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto")
    m = PeftModel.from_pretrained(base, adapter_path)
    return m, tok


def free(model):
    del model
    torch.cuda.empty_cache()


# ── Run ───────────────────────────────────────────────────────────────────────
results = {}

m, tok = load_baseline()
results["baseline"] = analyse_model(m, tok, "Baseline")
free(m)

lora_path = os.path.join(ADAPTERS_DIR, "lora_adapter")
m, tok    = load_lora(lora_path)
results["post_lora"] = analyse_model(m, tok, "Post-LoRA")
free(m)

qlora_path = os.path.join(ADAPTERS_DIR, "qlora_adapter")
m, tok     = load_qlora(qlora_path)
results["post_qlora"] = analyse_model(m, tok, "Post-QLoRA")
free(m)

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_path = os.path.join(PROCESSED_DIR, "bolukbasi_crows_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
labels = ["Baseline", "Post-LoRA", "Post-QLoRA"]
keys   = ["baseline", "post_lora", "post_qlora"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Bolukbasi Metrics on CrowS-Pairs (Gender) — OPT-1.3B", fontsize=13)

# Panel 1: DirectBias — stereo vs anti-stereo words
ax = axes[0]
x  = np.arange(len(labels))
w  = 0.3
stereo_vals = [results[k]["direct_bias_stereo"] for k in keys]
anti_vals   = [results[k]["direct_bias_anti"]   for k in keys]
ax.bar(x - w/2, stereo_vals, w, label="Stereo words (sent_more)", color="#E15759")
ax.bar(x + w/2, anti_vals,   w, label="Anti-stereo words (sent_less)", color="#76B7B2")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("|cos(w, g)|")
ax.set_title("DirectBias (Bolukbasi)\nStereo vs Anti-Stereo CrowS-Pairs Words")
ax.legend(fontsize=8)
for i, (sv, av) in enumerate(zip(stereo_vals, anti_vals)):
    ax.text(i - w/2, sv + 0.001, f"{sv:.4f}", ha="center", fontsize=8)
    ax.text(i + w/2, av + 0.001, f"{av:.4f}", ha="center", fontsize=8)

# Panel 2: DirectBias delta (stereo − anti)
ax = axes[1]
delta_vals = [results[k]["direct_bias_delta"] for k in keys]
bar_colors = ["#E15759" if d > 0 else "#76B7B2" for d in delta_vals]
bars = ax.bar(labels, delta_vals, color=bar_colors, width=0.45)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("DirectBias delta (stereo − anti)")
ax.set_title("DirectBias Delta\n(positive = stereo words more gender-aligned)")
for b, v in zip(bars, delta_vals):
    ax.text(b.get_x() + b.get_width()/2, v + (0.0002 if v >= 0 else -0.0006),
            f"{v:+.4f}", ha="center", fontsize=10)

# Panel 3: IndirectBias mean
ax = axes[2]
ib_vals  = [results[k]["indirect_bias_mean"] for k in keys]
ib_stds  = [results[k]["indirect_bias_std"]  for k in keys]
bars = ax.bar(labels, ib_vals, color=colors, width=0.45,
              yerr=ib_stds, capsize=5, error_kw={"elinewidth": 1.2})
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("IndirectBias (mean ± std)")
ax.set_title("IndirectBias\n(fraction of word-pair similarity\nexplained by gender direction)")
for b, v in zip(bars, ib_vals):
    ax.text(b.get_x() + b.get_width()/2, v + (0.002 if v >= 0 else -0.005),
            f"{v:.4f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "opt_bolukbasi_crows_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved.")

print("\n=== SUMMARY ===")
print(f"{'':15s}  {'DB-stereo':>10}  {'DB-anti':>8}  {'Delta':>8}  {'IB-mean':>8}  {'IB%pos':>7}")
for k, lbl in zip(keys, labels):
    r = results[k]
    print(f"{lbl:15s}  {r['direct_bias_stereo']:>10.4f}  {r['direct_bias_anti']:>8.4f}"
          f"  {r['direct_bias_delta']:>+8.4f}  {r['indirect_bias_mean']:>8.4f}"
          f"  {r['indirect_bias_pct_pos']:>6.1f}%")
