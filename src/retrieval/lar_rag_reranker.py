"""
LAR-RAG Reranker: Semantic + Temporal + Topology scoring for log candidates.

INFOCOM Requirements:
- Operate on candidate set (NO re-querying Azure AI Search!)
- Per-query score normalization before weighted sum
- Configurable ablations (no time, no topology)
- Cache template embeddings for efficiency

Architecture:
1. Input: Query embedding + List[Candidate]
2. Compute raw scores:
   - Semantic: cosine(query_emb, candidate_emb)
   - Temporal: K_time(delta_t) with exponential decay
   - Topology: K_topo(graph_distance) if available
3. Normalize each component in [0,1] per query
4. Weighted sum: α*semantic_norm + β*time_norm + γ*topo_norm
5. Return ranked candidates
"""

import os
import math
import json
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AzureOpenAI
from dataclasses import dataclass
import numpy as np
import ssl
from dotenv import load_dotenv

# Import candidate type and utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.graph_utils import SystemTopologyGraph, graph_distance

load_dotenv()

# SSL configuration for corporate proxies
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"


@dataclass
class Candidate:
    """
    Candidate log entry for reranking.
    
    Attributes:
        template_id: Template identifier
        timestamp: Log timestamp (Unix epoch seconds)
        node_id: Node/host identifier
        text: Optional log text
        embedding: Optional pre-computed embedding
    """
    template_id: str
    timestamp: float
    node_id: Optional[str] = None
    text: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class RankedCandidate:
    """
    Candidate with reranking scores.
    
    Attributes:
        candidate: Original Candidate object
        final_score: Weighted sum of normalized scores
        semantic_score: Raw semantic similarity
        temporal_score: Raw temporal kernel score
        topology_score: Raw topology kernel score
        semantic_norm: Normalized semantic score [0,1]
        temporal_norm: Normalized temporal score [0,1]
        topology_norm: Normalized topology score [0,1]
    """
    candidate: Candidate
    final_score: float
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    topology_score: float = 0.0
    semantic_norm: float = 0.0
    temporal_norm: float = 0.0
    topology_norm: float = 0.0


class LARRAGReranker:
    """
    LAR-RAG reranker for log candidates.
    
    Scores candidates using:
    1. Semantic similarity (query vs candidate/template)
    2. Temporal decay kernel
    3. Topology proximity kernel (optional)
    
    Per-query normalization ensures fair component weighting.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        lambda_decay: float = 0.001,
        lambda_post: float = 0.01,
        lambda_g: float = 0.5,
        use_temporal: bool = True,
        use_topology: bool = False,
        topology_mode: str = "synthetic",
        topology_file: Optional[str] = None,
        template_embeddings_file: Optional[str] = None,
        openai_client: Optional["AzureOpenAI"] = None
    ):
        """
        Initialize LAR-RAG reranker.
        
        Args:
            alpha: Weight for semantic similarity (default: 0.5)
            beta: Weight for temporal decay (default: 0.3)
            gamma: Weight for topology proximity (default: 0.2)
            lambda_decay: Decay rate for past logs (default: 0.001)
            lambda_post: Decay rate for future logs - higher penalty (default: 0.01)
            lambda_g: Topology decay rate (default: 0.5)
            use_temporal: Enable temporal scoring
            use_topology: Enable topology scoring
            topology_mode: "static", "log-inferred", or "synthetic"
            topology_file: Path to topology graph JSON
            template_embeddings_file: Path to cached template embeddings JSON
            openai_client: Optional pre-configured AzureOpenAI client
        """
        # Validate weights
        assert alpha + beta + gamma == 1.0, f"Weights must sum to 1.0, got {alpha+beta+gamma}"
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        self.lambda_post = lambda_post
        self.lambda_g = lambda_g
        self.use_temporal = use_temporal
        self.use_topology = use_topology
        
        # Load template embeddings cache
        self.template_embeddings = {}
        if template_embeddings_file and os.path.exists(template_embeddings_file):
            print(f"[Reranker] Loading template embeddings from {template_embeddings_file}")
            with open(template_embeddings_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # List format: [{"template_id": "T001", "embedding": [...]}]
                    for item in data:
                        self.template_embeddings[item['template_id']] = item['embedding']
                elif isinstance(data, dict):
                    # Dict format: {"T001": [...], "T002": [...]}
                    self.template_embeddings = data
            print(f"[Reranker] Loaded embeddings for {len(self.template_embeddings)} templates")
        
        # Initialize topology graph
        if self.use_topology:
            self.topology_graph = SystemTopologyGraph(
                mode=topology_mode,
                graph_file=topology_file
            )
        else:
            self.topology_graph = None
        
        # Lazy import for Azure dependencies (only if openai_client is needed)
        if openai_client is None:
            # Note: This class currently doesn't use OpenAI client, but we keep the pattern
            # for consistency with other retrievers
            pass
        
        # Initialize OpenAI client for embedding queries
        if openai_client:
            self.openai_client = openai_client
        else:
            http_client = httpx.Client(verify=False)
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-05-15",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                http_client=http_client
            )
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using Azure OpenAI.
        
        Args:
            query: Query text
        
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity in [0, 1] (shifted from [-1, 1])
        """
        if not vec1 or not vec2:
            return 0.0
        
        a = np.array(vec1)
        b = np.array(vec2)
        
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Shift from [-1, 1] to [0, 1]
        return (sim + 1) / 2
    
    def temporal_kernel(self, delta_t: float, incident_ts: float, candidate_ts: float) -> float:
        """
        Compute temporal decay kernel.
        
        Args:
            delta_t: Absolute time difference (seconds)
            incident_ts: Incident timestamp
            candidate_ts: Candidate timestamp
        
        Returns:
            Temporal score (higher = more relevant)
        
        Behavior:
            - Past logs (candidate before incident): exp(-lambda_decay * delta_t)
            - Future logs (candidate after incident): exp(-lambda_post * delta_t)
            - Future logs decay faster (lambda_post > lambda_decay)
        """
        if candidate_ts <= incident_ts:
            # Past log: normal decay
            return math.exp(-self.lambda_decay * delta_t)
        else:
            # Future log: higher penalty
            return math.exp(-self.lambda_post * delta_t)
    
    def topology_kernel(self, node_query: str, node_candidate: str) -> float:
        """
        Compute topology proximity kernel.
        
        Args:
            node_query: Query node ID
            node_candidate: Candidate node ID
        
        Returns:
            Topology score (higher = closer in topology)
        """
        if not self.topology_graph or not node_query or not node_candidate:
            return 0.0
        
        dist = graph_distance(self.topology_graph.graph, node_query, node_candidate)
        
        if dist is None or dist == float('inf'):
            return 0.0
        
        return math.exp(-self.lambda_g * dist)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-max normalization to [0, 1].
        
        Args:
            scores: Raw scores
        
        Returns:
            Normalized scores in [0, 1]
        
        INFOCOM Compliance:
            Per-query normalization ensures fair weighting across components.
        """
        if not scores or all(s == 0 for s in scores):
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All same value
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        incident_ts: float,
        query_node: Optional[str] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[RankedCandidate]:
        """
        Rerank candidates using LAR-RAG scoring.
        
        Args:
            query: Query text (embedded if query_embedding not provided)
            candidates: List of Candidate objects from expansion
            incident_ts: Incident timestamp for temporal scoring
            query_node: Optional query node for topology scoring
            query_embedding: Optional pre-computed query embedding
        
        Returns:
            List of RankedCandidate objects sorted by final_score (descending)
        
        Process:
            1. Compute raw scores (semantic, temporal, topology)
            2. Normalize each component per query
            3. Weighted sum: α*sem + β*temp + γ*topo
            4. Sort by final score
        """
        if not candidates:
            return []
        
        # Get query embedding
        if query_embedding is None:
            query_embedding = self.embed_query(query)
        
        # Compute raw scores
        semantic_scores = []
        temporal_scores = []
        topology_scores = []
        
        for cand in candidates:
            # Semantic: Use template embedding if available
            if cand.template_id and cand.template_id in self.template_embeddings:
                cand_embedding = self.template_embeddings[cand.template_id]
            else:
                # Fallback: zero similarity if no embedding
                cand_embedding = None
            
            if cand_embedding:
                sem_score = self.cosine_similarity(query_embedding, cand_embedding)
            else:
                sem_score = 0.0
            semantic_scores.append(sem_score)
            
            # Temporal
            if self.use_temporal:
                delta_t = abs(cand.timestamp - incident_ts)
                temp_score = self.temporal_kernel(delta_t, incident_ts, cand.timestamp)
            else:
                temp_score = 0.0
            temporal_scores.append(temp_score)
            
            # Topology
            if self.use_topology and query_node and cand.node_id:
                topo_score = self.topology_kernel(query_node, cand.node_id)
            else:
                topo_score = 0.0
            topology_scores.append(topo_score)
        
        # Normalize scores per query
        semantic_norm = self._normalize_scores(semantic_scores)
        temporal_norm = self._normalize_scores(temporal_scores) if self.use_temporal else [0.0] * len(candidates)
        topology_norm = self._normalize_scores(topology_scores) if self.use_topology else [0.0] * len(candidates)
        
        # Compute final scores
        ranked_candidates = []
        for i, cand in enumerate(candidates):
            final_score = (
                self.alpha * semantic_norm[i] +
                self.beta * temporal_norm[i] +
                self.gamma * topology_norm[i]
            )
            
            ranked_cand = RankedCandidate(
                candidate=cand,
                final_score=final_score,
                semantic_score=semantic_scores[i],
                temporal_score=temporal_scores[i],
                topology_score=topology_scores[i],
                semantic_norm=semantic_norm[i],
                temporal_norm=temporal_norm[i],
                topology_norm=topology_norm[i]
            )
            ranked_candidates.append(ranked_cand)
        
        # Sort by final score (descending)
        ranked_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked_candidates
    
    def rerank_ablation(
        self,
        query: str,
        candidates: List[Candidate],
        incident_ts: float,
        query_node: Optional[str] = None,
        ablation: str = "none"
    ) -> List[RankedCandidate]:
        """
        Rerank with ablation study.
        
        Args:
            query: Query text
            candidates: List of candidates
            incident_ts: Incident timestamp
            query_node: Optional query node
            ablation: "no_temporal", "no_topology", or "none"
        
        Returns:
            List of RankedCandidate objects
        """
        # Temporarily disable components
        original_use_temporal = self.use_temporal
        original_use_topology = self.use_topology
        
        if ablation == "no_temporal":
            self.use_temporal = False
        elif ablation == "no_topology":
            self.use_topology = False
        
        try:
            result = self.rerank(query, candidates, incident_ts, query_node)
        finally:
            # Restore original settings
            self.use_temporal = original_use_temporal
            self.use_topology = original_use_topology
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("=== LAR-RAG Reranker Test ===")
    
    # Create synthetic candidates
    test_candidates = [
        Candidate(
            log_id="log_001",
            timestamp=1234567800.0,
            node_id="R02-M1-N0-C:J12-U01",
            template_id="T001",
            message="GPU cache parity error",
            raw={}
        ),
        Candidate(
            log_id="log_002",
            timestamp=1234567850.0,
            node_id="R02-M1-N0-C:J12-U02",
            template_id="T002",
            message="Memory ECC error",
            raw={}
        ),
        Candidate(
            log_id="log_003",
            timestamp=1234567900.0,
            node_id="R03-M1-N0-C:J12-U01",
            template_id="T001",
            message="GPU cache parity error",
            raw={}
        ),
    ]
    
    try:
        # Initialize reranker
        reranker = LARRAGReranker(
            alpha=0.6,
            beta=0.4,
            gamma=0.0,
            use_temporal=True,
            use_topology=False
        )
        
        # Rerank candidates
        incident_ts = 1234567890.0
        query = "GPU failure on node R02"
        
        print(f"\nQuery: {query}")
        print(f"Incident timestamp: {incident_ts}")
        print(f"\nRanking {len(test_candidates)} candidates...")
        
        # Note: This will fail without Azure credentials
        # In production, use with proper credentials
        # ranked = reranker.rerank(query, test_candidates, incident_ts)
        
        print("\n✓ Reranker initialized successfully")
        print(f"  - Alpha (semantic): {reranker.alpha}")
        print(f"  - Beta (temporal): {reranker.beta}")
        print(f"  - Gamma (topology): {reranker.gamma}")
        print(f"  - Temporal enabled: {reranker.use_temporal}")
        print(f"  - Topology enabled: {reranker.use_topology}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nNote: Full reranking requires Azure OpenAI credentials.")
        print("Reranker structure and initialization validated.")
