# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/LAR_RAG.git
cd LAR_RAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

```python
from src.retrieval.lar_rag_reranker import LARRAGReranker
from src.utils.graph_utils import SystemTopologyGraph

# Initialize system topology
graph = SystemTopologyGraph()
graph.add_edge("node1", "node2")
graph.add_edge("node2", "node3")

# Create reranker
reranker = LARRAGReranker(
    topology_graph=graph,
    lambda_pre=0.01,      # Temporal decay (before incident)
    lambda_post=0.5,       # Temporal decay (after incident)
    lambda_g=0.3,          # Topology decay
    alpha=0.5,             # Semantic weight
    beta=0.3,              # Temporal weight
    gamma=0.2              # Topology weight
)

# Perform retrieval
query = "Database connection timeout"
incident_time = 1234567890.0  # Unix timestamp
incident_node = "node2"

# Stage 1: Get initial candidates (implement your own retriever)
candidates = your_retriever.search(query, k=50)

# Stage 2: LAR-RAG reranking
results = reranker.rerank(
    query=query,
    candidates=candidates,
    incident_time=incident_time,
    incident_node=incident_node,
    k=10
)

# Use results
for i, result in enumerate(results[:5]):
    print(f"{i+1}. {result.text} (score: {result.score:.3f})")
```

## Data Format

### Log Format
```python
{
    "text": "Database connection timeout",
    "timestamp": 1234567890.0,  # Unix timestamp
    "node": "node2",             # System component
    "template_id": "T123"        # Optional: template identifier
}
```

### Graph Format
```python
# Option 1: Programmatic
from src.utils.graph_utils import SystemTopologyGraph

graph = SystemTopologyGraph()
graph.add_edge("web_server", "app_server")
graph.add_edge("app_server", "database")

# Option 2: From JSON
graph = SystemTopologyGraph.from_json({
    "nodes": ["web_server", "app_server", "database"],
    "edges": [
        ["web_server", "app_server"],
        ["app_server", "database"]
    ]
})
```

## Examples

See [`examples/`](../examples/) for complete working examples:
- `simple_demo.py` - Basic usage with synthetic data
- `custom_retriever.py` - Implementing your own Stage 1 retriever

## Next Steps

- Read the [algorithm documentation](algorithm.md) for details
- Check out the [examples](../examples/) folder
- Implement your own first-stage retriever (BM25, dense, hybrid)
- Tune hyperparameters for your dataset
