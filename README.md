# LAR-RAG: Latency-Aware Retrieval for Root Cause Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1080%2F08839514.2026.2684312-blue)](https://doi.org/10.1080/08839514.2026.2684312)

**LAR-RAG** is a training-free retrieval system for log-based Root Cause Analysis (RCA) that combines semantic search with RCA-specific temporal and topology priors.

> **📄 Published Paper:** [Quality–Latency Benchmarking of RCA Template Retrieval for GenAI-Driven Network Operations](https://www.tandfonline.com/doi/full/10.1080/08839514.2026.2684312) — *Applied Artificial Intelligence*, Taylor & Francis, 2026.

## 📚 Overview

LAR-RAG addresses the challenge of retrieving relevant log entries for root cause analysis by incorporating:
- **Semantic retrieval** via embeddings (text-embedding-ada-002)
- **Temporal proximity** weighting for time-correlated failures
- **Topology awareness** for distributed system dependencies

See [docs/LAR-RAG-description.md](docs/LAR-RAG-description.md) for algorithm details.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11
- Azure subscription (OpenAI + AI Search)
- 8GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/Mustafa3946/rca-retrieval-bench.git
cd rca-retrieval-bench

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configure Azure Services

Create `.env` file:
```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_ADMIN_KEY=your_search_key
```

Or deploy via Terraform:
```bash
cd infra
terraform init
terraform apply
```

### Run Experiments

**Option 1: Automated (Recommended)**
```powershell
.\scripts\reproduce_all.ps1
```

**Option 2: Full Pipeline**
```bash
python scripts/run_bgl_full_pipeline.py
```

**Option 3: Manual Steps**
```bash
# 1. Parse BGL logs
python src/data/parse_bgl.py

# 2. Extract templates
python src/preprocess/template_bgl.py

# 3. Generate embeddings (PRODUCTION - v2)
python src/embeddings/embed_templates_safe_v2.py

# 4. Run ablation study
python src/evaluation/ablation_study.py
```

> **📌 Production Embedding (v2):** The pipeline now uses `embed_templates_safe_v2.py` for production-grade embedding with:
> - **Sharded Parquet output** (no memory wall on resume)
> - **Drift detection** (re-embeds changed templates)
> - **Robust retry** (proper OpenAI error handling)
> - **CI/automation support** (`--non-interactive` flag)
> 
> See [docs/SAFE_EMBEDDING_V2_FIXES.md](docs/SAFE_EMBEDDING_V2_FIXES.md) for details or [docs/V1_TO_V2_MIGRATION.md](docs/V1_TO_V2_MIGRATION.md) if migrating from v1.

Results are saved to `results/LAR-RAG_*_metrics.csv`.

## 📁 Repository Structure

```
LAR_RAG/
├── src/
│   ├── data/           # Log parsing (BGL, HDFS)
│   ├── embeddings/     # Template + log embedding
│   ├── evaluation/     # Ground truth, metrics, experiments
│   ├── indexing/       # Azure AI Search indexing
│   └── preprocess/     # Template extraction
├── config/             # YAML experiment configs
├── docs/               # Algorithm, paper tables, guides
│   ├── REPRODUCTION.md    # Step-by-step reproduction
│   ├── PRODUCTION.md      # Deployment guide
│   └── PUBLICATION_READINESS.md
├── scripts/            # Automation scripts
├── infra/              # Terraform (Azure)
└── notebooks/          # Jupyter experiments
```

## 📊 Experimental Results

LAR-RAG achieves:
- **HR@10:** 0.867 (vs. 0.733 baseline)
- **MRR:** 0.752 (vs. 0.616 baseline)
- **Latency:** <100ms per query

See [docs/experimental_results.md](docs/experimental_results.md) and [docs/paper_tables.md](docs/paper_tables.md) for full results.

## 📖 Documentation

### Core Guides
- **[SAFE_EMBEDDING_V2_FIXES.md](docs/SAFE_EMBEDDING_V2_FIXES.md)** - ⭐ **PRODUCTION** Critical fixes in v2 (sharded output, drift detection)
- **[V1_TO_V2_MIGRATION.md](docs/V1_TO_V2_MIGRATION.md)** - ⭐ **NEW** Migration guide from v1 to v2
- **[SAFE_EMBEDDING_QUICK_REF.md](docs/SAFE_EMBEDDING_QUICK_REF.md)** - Quick reference for safe embedding
- **[AZURE_SETUP_GUIDE.md](AZURE_SETUP_GUIDE.md)** - Azure configuration and connection setup
- **[REPRODUCTION.md](docs/REPRODUCTION.md)** - Complete step-by-step guide to reproduce all experiments
- **[PRODUCTION.md](docs/PRODUCTION.md)** - Production deployment and usage guide
- **[LAR-RAG-description.md](docs/LAR-RAG-description.md)** - Algorithm description and pseudocode

### Implementation Details
- **[SAFE_EMBEDDING_IMPLEMENTATION.md](docs/SAFE_EMBEDDING_IMPLEMENTATION.md)** - ⭐ **NEW** Publication-grade embedding implementation
- **[EMBEDDING_COMPARISON.md](docs/EMBEDDING_COMPARISON.md)** - ⭐ **NEW** Old vs new embedding comparison
- **[BLOB_EXPANSION_GUIDE.md](docs/BLOB_EXPANSION_GUIDE.md)** - Template-first architecture with Azure Blob expansion
- **[PUBLICATION_READINESS.md](docs/PUBLICATION_READINESS.md)** - Paper submission checklist
- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - High-level project summary

## 🧹 Repository Cleanup

To remove generated artifacts and restore to a clean state:

```powershell
# Preview what would be deleted
.\scripts\cleanup_repo.ps1 -DryRun

# Clean up (keeps core code/docs)
.\scripts\cleanup_repo.ps1

# Clean and archive legacy files
.\scripts\cleanup_repo.ps1 -ArchiveLegacy
```

This removes:
- Generated data (`data/processed/**`, `data/raw/**`)
- Experiment results (`results/**`)
- Logs and cache (`logs-raw/**`, `__pycache__/`)
- Terraform state

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{mustafa2026larrag,
  title={Quality--Latency Benchmarking of {RCA} Template Retrieval for {GenAI}-Driven Network Operations},
  author={Mustafa, Mohammad},
  journal={Applied Artificial Intelligence},
  year={2026},
  publisher={Taylor \& Francis},
  doi={10.1080/08839514.2026.2684312},
  url={https://www.tandfonline.com/doi/full/10.1080/08839514.2026.2684312}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

This is a research prototype. For issues or questions, please open a GitHub issue.

---

**Status:** Published — *Applied Artificial Intelligence*, Taylor & Francis, 2026  
**Last Updated:** June 2026
