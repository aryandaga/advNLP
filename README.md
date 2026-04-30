# Gender Bias Across Adaptation Regimes in LLMs

This repository contains experiments for studying gender bias across prompting, LoRA, QLoRA, and bias-aware LoRA adaptation regimes. The project evaluates downstream task performance on SST-2 and bias behavior on CrowS-Pairs, with additional representation-level geometric bias analysis inspired by Bolukbasi et al.

The current paper focuses on the relationship between behavioral bias, measured by Stereotype Preference Score (SPS), and representation-level bias, measured through projection onto a gender direction.

## Research Questions

1. How does gender bias vary across zero-shot prompting, few-shot prompting, LoRA, and QLoRA?
2. Does reducing representation-level gender bias during bias-aware LoRA reduce behavioral bias?
3. Can bias-aware fine-tuning preserve SST-2 performance while mitigating bias?
4. How closely do representation-level and behavioral bias metrics align?

## Repository Structure

```text
advNLP/
  paper/                         # Report artifacts and paper-facing materials
  src/
    evaluation/                  # SST-2, CrowS-Pairs, BBQ, and geometric-bias evaluators
    training/                    # OPT-1.3B LoRA and QLoRA training scripts
    plotting/                    # Figure generation scripts
    utils/                       # Shared helpers, reserved for future cleanup
  scripts/
    modal/                       # Modal jobs for LLaMA-2-7B experiments
  notebooks/
    exploratory/                 # Development notebooks and earlier experiments
    report/                      # Analysis notebooks used for report figures
  experiments/
    llama2_7b/bias_aware_lora/   # Lambda sweeps, Kaggle outputs, and bias-aware runs
    opt_1.3b/                    # Reserved for OPT experiment bundles
    legacy/                      # Older or supporting experiment material
  results/
    processed/                   # JSON/CSV result files grouped by model family
    figures/
      paper/                     # Figures used by the current report
      exploratory/               # Supporting and earlier figures
    logs/                        # Reserved for plain training logs
  artifacts/
    adapters/                    # Saved LoRA/QLoRA adapter folders
    checkpoints/                 # Reserved for large checkpoints
  docs/                          # Repo map and result manifests
```

## Main Scripts

```bash
python src/evaluation/sst2_prompt_eval.py
python src/training/train_lora.py
python src/training/train_qlora.py
python src/evaluation/geometric_bias.py
python src/evaluation/crows_pairs_eval.py
python src/evaluation/geometric_crows_eval.py
python src/plotting/plot_llama_results.py
```

Modal jobs for LLaMA-2-7B:

```bash
modal run scripts/modal/run_llama_lora.py
modal run scripts/modal/run_llama_qlora.py
modal run scripts/modal/run_llama_all.py
```

## Key Result Locations

- OPT-1.3B processed metrics: `results/processed/opt_1.3b/`
- LLaMA-2-7B processed metrics: `results/processed/llama2_7b/`
- Paper figures: `results/figures/paper/`
- Exploratory figures: `results/figures/exploratory/`
- Bias-aware LoRA lambda runs: `experiments/llama2_7b/bias_aware_lora/`
- Saved OPT adapters: `artifacts/adapters/opt_1.3b/`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: Python 3.11+, CUDA-capable NVIDIA GPU for local OPT experiments, and Modal/Kaggle/Colab for larger LLaMA-2-7B runs.

## Notes

Large model weights are intentionally not tracked directly. Adapter folders in `artifacts/` may contain configs/tokenizers, while full adapter weight files should be regenerated, downloaded from the experiment environment, or tracked with Git LFS if needed.
