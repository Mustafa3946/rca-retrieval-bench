# LAR-RAG: Quality-Latency Benchmarking for RCA Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1080%2F08839514.2026.2684312-blue)](https://doi.org/10.1080/08839514.2026.2684312)

LAR-RAG is a training-free retrieval method for log-based root cause analysis (RCA), optimized for retrieval quality and latency.

Paper:
- [Quality-Latency Benchmarking of Log-Template Retrieval for GenAI-Assisted Operational RCA](https://www.tandfonline.com/doi/full/10.1080/08839514.2026.2684312)

## Quick Start

```bash
git clone https://github.com/Mustafa3946/rca-retrieval-bench.git
cd rca-retrieval-bench
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Essential Links

- [Quick Start Guide](docs/quickstart.md)
- [Algorithm Overview](docs/algorithm.md)
- [Example Usage](examples/simple_demo.py)
- [Citation Metadata](CITATION.cff)
- [License](LICENSE)

## Citation (BibTeX)

```bibtex
@article{mustafa2026larrag,
  title={Quality--Latency Benchmarking of Log-Template Retrieval for GenAI-Assisted Operational RCA},
  author={Mustafa, Mohammad Abdur Rahim and Mamun, Quazi},
  journal={Applied Artificial Intelligence},
  year={2026},
  doi={10.1080/08839514.2026.2684312},
  url={https://doi.org/10.1080/08839514.2026.2684312}
}
```
