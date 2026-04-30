# Repository Map

This repository is organized around the current report rather than around where each experiment originally ran.

## Source Code

- `src/evaluation/`: evaluation scripts for prompting, CrowS-Pairs SPS, BBQ, and geometric bias.
- `src/training/`: local OPT-1.3B LoRA and QLoRA training scripts.
- `src/plotting/`: figure generation scripts.
- `scripts/modal/`: Modal jobs for LLaMA-2-7B cloud experiments.

## Research Artifacts

- `results/processed/`: compact JSON/CSV outputs used for tables and plots.
- `results/figures/paper/`: figures intended for the report.
- `results/figures/exploratory/`: earlier or supporting figures.
- `experiments/llama2_7b/bias_aware_lora/`: lambda sweeps and training logs from bias-aware LoRA runs.
- `artifacts/adapters/`: saved adapter folders.

## Notebook Policy

- `notebooks/report/` contains notebooks that support the written report.
- `notebooks/exploratory/` contains development notebooks and earlier experiment drafts.

