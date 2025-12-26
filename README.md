# Quality–Latency Benchmarking of RCA Template Retrieval for GenAI-Driven Network Operations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Benchmarking framework for evaluating retrieval quality and latency tradeoffs in RCA template retrieval for GenAI-driven network operations.

## Key Features

- **Training-free**: Works with pre-trained embeddings, no model training required
- **Interpretable**: Explicit scoring components (semantic, temporal, topology)
- **Production-ready**: <100ms latency for GenAI network operations
- **Flexible**: Pluggable Stage 1 retriever (BM25, dense, hybrid)

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/LAR_RAG.git
cd LAR_RAG
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Basic Usage

```python
from src.retrieval.lar_rag_reranker import LARRAGReranker
from src.utils.graph_utils import SystemTopologyGraph

# Setup topology
graph = SystemTopologyGraph()
graph.add_edge("web", "app")
graph.add_edge("app", "db")

# Create reranker
reranker = LARRAGReranker(
    topology_graph=graph,
    lambda_pre=0.01,
    lambda_post=0.5,
    lambda_g=0.3
)

# Rerank candidates
results = reranker.rerank(
    query="connection timeout",
    candidates=initial_results,
    incident_time=timestamp,
    incident_node="app",
    k=10
)
```

See [examples/simple_demo.py](examples/simple_demo.py) for a complete working example.

## Documentation

- **[Algorithm Details](docs/algorithm.md)** - How LAR-RAG works
- **[Quick Start Guide](docs/quickstart.md)** - Usage and examples
- **[Examples](examples/)** - Working code samples

## Repository Structure

```
LAR_RAG/
├── src/
│   ├── retrieval/          # Core LAR-RAG implementation
│   ├── utils/              # Temporal & topology scoring
│   ├── preprocess/         # Log parsing utilities
│   ├── embeddings/         # Embedding generation
│   └── storage/            # Data structures
├── examples/               # Usage examples
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Algorithm

LAR-RAG is a two-stage retrieval approach optimized for quality-latency tradeoffs:

**Stage 1**: Initial candidate retrieval (BM25/dense/hybrid)

**Stage 2**: Reranking with combined score:
```
score = α × semantic_sim + β × temporal_decay + γ × topology_proximity
```

Where:
- **Semantic**: Cosine similarity of embeddings
- **Temporal**: Exponential decay based on time difference (asymmetric)
- **Topology**: Graph distance decay on system topology

See [docs/algorithm.md](docs/algorithm.md) for details.

## Use Cases

- **GenAI Network Operations**: AI-driven network troubleshooting and root cause analysis
- **Retrieval Benchmarking**: Evaluate retrieval systems with quality-latency tradeoffs
- **Log Investigation**: Trace errors through distributed systems with latency guarantees

## Requirements

- Python 3.10+
- Core: `numpy`, `networkx`, `scipy`
- Retrieval: `rank-bm25`, `faiss-cpu`
- Optional: `openai` (for embeddings), `tqdm` (progress bars)

## License

MIT License - see [LICENSE](LICENSE) for details

## Contributing

Contributions welcome via issues or pull requests.

## Citation

If you use LAR-RAG in your research, please cite:

```bibtex
@inproceedings{larrag2026,
  title={Quality–Latency Benchmarking of RCA Template Retrieval for GenAI-Driven Network Operations},
  author={Mohammad Abdur Rahim Mustafa and Quazi Mamun},
  booktitle={IEEE INFOCOM},
  year={2026},
  organization={Charles Sturt University}
}
```
