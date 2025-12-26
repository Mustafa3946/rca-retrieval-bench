# LAR-RAG Algorithm

## Overview

This document describes the **LAR-RAG** (Latency-Aware Retrieval for RCA) algorithm, part of the **Quality–Latency Benchmarking of RCA Template Retrieval for GenAI-Driven Network Operations** framework. LAR-RAG is a two-stage retrieval system that combines semantic search with RCA-specific temporal and topological priors.

## Algorithm

### Input
- **Query** `q`: Natural language incident description
- **Incident time** `t_q`: Timestamp of the incident
- **Incident node** `v_q`: System component where incident occurred
- **Log corpus** `C`: Set of all system logs

### Stage 1: Initial Retrieval

Perform hybrid search to get candidate set:
```
candidates = HybridSearch(q, C, k=50)
# Combines BM25 (keyword) + Dense (semantic) with RRF fusion
```

### Stage 2: LAR-RAG Reranking

For each log `ℓ` in candidates:

1. **Semantic Score**
   ```
   s_sem = cosine_similarity(embed(q), embed(ℓ))
   ```

2. **Temporal Score** (asymmetric decay)
   ```
   Δt = t_q - ℓ.timestamp
   
   if Δt ≥ 0:  # Log before incident
       g_t = exp(-λ_pre × Δt)
   else:        # Log after incident (penalized)
       g_t = exp(-λ_post × |Δt|)
   ```

3. **Topology Score** (graph distance)
   ```
   d_g = shortest_path(v_q, ℓ.node, G)
   g_g = exp(-λ_g × d_g)
   ```

4. **Combined Score**
   ```
   score = α × s_sem + β × g_t + γ × g_g
   ```

### Output

Return top-k logs sorted by score descending

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `λ_pre` | Temporal decay rate (before incident) | 0.001-0.01 |
| `λ_post` | Temporal decay rate (after incident) | 0.1-1.0 (higher penalty) |
| `λ_g` | Topology decay rate | 0.1-0.5 |
| `α`, `β`, `γ` | Score weights | α=0.5, β=0.3, γ=0.2 |

## Implementation

See [`src/retrieval/`](../src/retrieval/) for the core implementation:
- `lar_rag_reranker.py` - Main reranking logic
- `bm25_retriever.py` - BM25 baseline
- Temporal scoring: [`src/utils/temporal_scoring.py`](../src/utils/temporal_scoring.py)
- Topology scoring: [`src/utils/graph_utils.py`](../src/utils/graph_utils.py)
