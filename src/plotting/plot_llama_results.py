"""
Llama-2-7B bias shift visualizations.
Run with: python src/plotting/plot_llama_results.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT_DIR, "results", "processed", "llama2_7b")
FIGURES = os.path.join(ROOT_DIR, "results", "figures", "paper")
os.makedirs(FIGURES, exist_ok=True)

with open(os.path.join(RESULTS, "llama_lora_results.json"))  as f: lora  = json.load(f)
with open(os.path.join(RESULTS, "llama_qlora_results.json")) as f: qlora = json.load(f)

# ── Data ──────────────────────────────────────────────────────────────────────
conditions   = ["Baseline", "Post-LoRA", "Post-QLoRA"]
sst2         = [lora["baseline"]["sst2_accuracy"],
                lora["post_lora"]["sst2_accuracy"],
                qlora["post_qlora"]["sst2_accuracy"]]
sps          = [lora["baseline"]["crows_sps"],
                lora["post_lora"]["crows_sps"],
                qlora["post_qlora"]["crows_sps"]]
db_stereo    = [lora["baseline"]["direct_bias_stereo"],
                lora["post_lora"]["direct_bias_stereo"],
                qlora["post_qlora"]["direct_bias_stereo"]]
db_anti      = [lora["baseline"]["direct_bias_anti"],
                lora["post_lora"]["direct_bias_anti"],
                qlora["post_qlora"]["direct_bias_anti"]]
db_delta     = [s - a for s, a in zip(db_stereo, db_anti)]

colors = ["#9ecae1", "#2171b5", "#08306b"]  # light → dark blue

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Accuracy vs Bias (side-by-side bars, shared x-axis)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Llama-2-7B: Task Performance vs Gender Bias Shift", fontsize=14, fontweight="bold")

x = np.arange(len(conditions))
bars = ax1.bar(x, sst2, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
ax1.set_xticks(x); ax1.set_xticklabels(conditions, fontsize=11)
ax1.set_ylabel("SST-2 Accuracy", fontsize=11); ax1.set_ylim(0, 1.08)
ax1.set_title("Task Performance", fontsize=12, fontweight="bold")
ax1.axhline(1.0, color="gray", lw=0.7, ls="--")
for b, v in zip(bars, sst2):
    ax1.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

bars2 = ax2.bar(x, sps, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
ax2.axhline(50, color="red", lw=1.5, ls="--", label="Unbiased (50%)")
ax2.set_xticks(x); ax2.set_xticklabels(conditions, fontsize=11)
ax2.set_ylabel("Stereotype Preference Score (%)", fontsize=11)
ax2.set_ylim(48, 65); ax2.set_title("Behavioral Bias (CrowS-Pairs SPS)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)
for b, v in zip(bars2, sps):
    ax2.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig_llama_accuracy_vs_bias.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_accuracy_vs_bias.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — DirectBias: stereo vs anti with delta annotation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Llama-2-7B: Representation-Level Gender Bias (Bolukbasi DirectBias)",
             fontsize=13, fontweight="bold")

x = np.arange(len(conditions))
w = 0.3
b1 = ax.bar(x - w/2, db_stereo, w, label="Stereo words (sent_more)", color="#E15759", edgecolor="white")
b2 = ax.bar(x + w/2, db_anti,   w, label="Anti-stereo words (sent_less)", color="#76B7B2", edgecolor="white")

for i, (sv, av, dv) in enumerate(zip(db_stereo, db_anti, db_delta)):
    ax.text(i-w/2, sv+0.003, f"{sv:.4f}", ha="center", fontsize=9)
    ax.text(i+w/2, av+0.003, f"{av:.4f}", ha="center", fontsize=9)
    top = max(sv, av) + 0.012
    ax.annotate(f"Δ = {dv:+.4f}", xy=(i, top), ha="center", fontsize=9,
                color="#E15759" if dv > 0 else "#2d6a4f",
                fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylabel("|cos(w, g)|  — gender alignment", fontsize=11)
ax.set_ylim(0, 0.28)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig_llama_directbias.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_directbias.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Bias drift summary: SPS change + DB delta change
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Llama-2-7B: Bias Drift After Fine-Tuning (Change from Baseline)",
             fontsize=13, fontweight="bold")

# SPS change
ax = axes[0]
labels_ft  = ["Post-LoRA", "Post-QLoRA"]
sps_change = [sps[1]-sps[0], sps[2]-sps[0]]
bar_cols   = ["#E15759" if v > 0 else "#4CAF50" for v in sps_change]
bars = ax.bar(labels_ft, sps_change, color=bar_cols, width=0.4, edgecolor="white", linewidth=1.2)
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("ΔSPS (pp)", fontsize=11)
ax.set_title("Behavioral Bias Change\n(+ = more stereotypical)", fontsize=11, fontweight="bold")
for b, v in zip(bars, sps_change):
    ax.text(b.get_x()+b.get_width()/2, v + (0.03 if v>=0 else -0.08),
            f"{v:+.2f} pp", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(min(sps_change)-0.5, max(sps_change)+0.5)

# DB delta change
ax = axes[1]
delta_change = [db_delta[1]-db_delta[0], db_delta[2]-db_delta[0]]
bar_cols2    = ["#E15759" if v > 0 else "#4CAF50" for v in delta_change]
bars2 = ax.bar(labels_ft, delta_change, color=bar_cols2, width=0.4, edgecolor="white", linewidth=1.2)
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Δ(DirectBias delta)", fontsize=11)
ax.set_title("Representation Bias Change\n(+ = stereo words more gender-aligned)", fontsize=11, fontweight="bold")
for b, v in zip(bars2, delta_change):
    ax.text(b.get_x()+b.get_width()/2, v + (0.0003 if v>=0 else -0.0008),
            f"{v:+.4f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(min(delta_change)-0.005, max(delta_change)+0.005)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig_llama_bias_drift.png"), dpi=150, bbox_inches="tight")
plt.close(); print("Saved llama_bias_drift.png")

print("\nAll Llama-2-7B plots saved.")
