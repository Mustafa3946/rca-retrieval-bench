# LAR-RAG Experiments - Quick Run Guide

## Prerequisites
- Python environment activated: `.venv\Scripts\Activate.ps1`
- BGL data processed (parsed logs, templates, embeddings)
- Indices built (FAISS, BM25, occurrence store)

## Run Experiments

### 1. Full Run (All Baselines + LAR-RAG)
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config config\bgl_full_run.yaml
```
**Output:** `results/bgl_full_run/`

### 2. Ablation Study (LAR-RAG Variants)
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config config\bgl_ablations.yaml
```
**Output:** `results/bgl_ablations/`

Evaluates:
- `lar_rag_full` (α=1.0, β=0.25, γ=0.25)
- `lar_rag_no_time` (α=1.0, β=0.0, γ=0.25)
- `lar_rag_no_topo` (α=1.0, β=0.25, γ=0.0)

### 3. Parameter Sweep (16 Configurations)
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config config\bgl_sweep.yaml
```
**Output:** `results/bgl_sweep/`

Grid: β ∈ {0.0, 0.25, 0.5, 1.0} × γ ∈ {0.0, 0.25, 0.5, 1.0}

## Results
Each run generates:
- `results.json` - Aggregate metrics
- `results.csv` - Summary table
- `per_query_*.jsonl` - Per-query results
- `run_meta.json` - Configuration metadata

## Force Rebuild
To regenerate ground truth or indices:
```powershell
.venv\Scripts\python.exe scripts\paper_run_local.py --config <config> --force-rebuild
```

## Typical Runtime
- Full run: ~20 minutes
- Ablations: ~25 minutes
- Sweep (baselines only): ~7 minutes
