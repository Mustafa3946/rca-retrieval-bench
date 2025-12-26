"""
Hybrid retriever combining BM25 and dense search.

Two-stage architecture:
1. BM25 retrieval for candidate generation
2. Dense reranking using embedding similarity

Template-first architecture for INFOCOM compliance.
"""

import logging
from typing import List, Optional
import numpy as np

from src.retrieval.base import Retriever, RetrievalResult
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """
    Hybrid retriever: BM25 + Dense reranking.
    
    Architecture:
    1. Stage 1: BM25 retrieves top-N candidates (fast, broad recall)
    2. Stage 2: Rerank candidates using dense similarity (precise)
    
    This is more efficient than pure dense search for large corpora.
    """
    
    def __init__(
        self,
        templates_file: str,
        index_dir: str,
        template_map_file: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        expand_to_logs: bool = False,
        stage1_k: int = 200,
        alpha: float = 0.5,
        beta: float = 0.5,
        local_only: bool = True,
        embeddings_manifest: Optional[str] = None,
        embeddings_shards_dir: Optional[str] = None,
        max_cached_shards: int = 16
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            templates_file: Path to templates JSONL file
            index_dir: Directory containing FAISS index
            template_map_file: Path to template-to-log mapping
            embedding_model: OpenAI embedding model name
            expand_to_logs: Expand templates to logs
            stage1_k: Number of BM25 candidates (default: 200)
            alpha: Weight for BM25 score (default: 0.5)
            beta: Weight for dense score (default: 0.5)
            local_only: Use offline mode (no Azure API calls)
            embeddings_manifest: Path to manifest.json (required if local_only=True)
            embeddings_shards_dir: Path to embeddings shards (required if local_only=True)
            max_cached_shards: Max shards in memory (default: 16)
        """
        self.stage1_k = stage1_k
        self.alpha = alpha
        self.beta = beta
        self.local_only = local_only
        
        # Validate offline mode requirements
        if local_only and (not embeddings_manifest or not embeddings_shards_dir):
            raise ValueError(
                "local_only=True requires embeddings_manifest and embeddings_shards_dir.\n"
                "For paper runs, provide paths to precomputed embeddings."
            )
        
        # Validate weights
        if not np.isclose(alpha + beta, 1.0):
            logger.warning(f"Weights don't sum to 1.0: alpha={alpha}, beta={beta}")
        
        # Initialize BM25 retriever
        logger.info("Initializing BM25 retriever (Stage 1)...")
        self.bm25_retriever = BM25Retriever(
            templates_file=templates_file,
            template_map_file=template_map_file,
            expand_to_logs=False  # Keep at template level for reranking
        )
        
        # Initialize Dense retriever
        logger.info("Initializing Dense retriever (Stage 2)...")
        self.dense_retriever = DenseRetriever(
            index_dir=index_dir,
            embedding_model=embedding_model,
            template_map_file=template_map_file,
            expand_to_logs=expand_to_logs,
            local_only=local_only,
            embeddings_manifest=embeddings_manifest,
            embeddings_shards_dir=embeddings_shards_dir,
            max_cached_shards=max_cached_shards
        )
        
        logger.info("✅ Hybrid retriever ready")
    
    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_template_id: Optional[str] = None,
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid BM25 + Dense.
        
        Args:
            query_text: Query text (for BM25, optional in local_only mode)
            query_template_id: Query template ID (required for local_only mode)
            incident_time: Ignored (hybrid doesn't use time)
            incident_node: Ignored (hybrid doesn't use topology)
            top_k: Number of final results
            
        Returns:
            List of RetrievalResult objects
        """
        # Validate inputs for local_only mode
        if self.local_only and not query_template_id:
            raise ValueError(
                "local_only=True requires query_template_id for template-first retrieval.\n"
                "Provide incident_template_id from ground truth incidents."
            )
        
        # Stage 1: BM25 candidate generation
        logger.debug(f"Stage 1: BM25 retrieval (k={self.stage1_k})")
        
        # BM25 uses query_text (or can fall back to template content if needed)
        if not query_text and query_template_id:
            logger.warning("BM25 stage requires query_text, using template_id as fallback")
            query_text = query_template_id  # Fallback for BM25
        
        bm25_results = self.bm25_retriever.retrieve(
            query_text=query_text,
            top_k=self.stage1_k
        )
        
        # Extract candidate template_ids
        candidate_template_ids = [r.template_id for r in bm25_results]
        
        if not candidate_template_ids:
            logger.warning("No BM25 candidates found")
            return []
        
        # Create lookup for BM25 scores
        bm25_scores = {r.template_id: r.score for r in bm25_results}
        
        # Stage 2: Dense reranking
        logger.debug(f"Stage 2: Dense reranking ({len(candidate_template_ids)} candidates)")
        
        # Get dense scores for candidates (pass query_template_id for offline mode)
        dense_results = self.dense_retriever.retrieve(
            query_text=query_text,
            query_template_id=query_template_id,
            top_k=len(candidate_template_ids)
        )
        
        # Create lookup for dense scores
        dense_scores = {r.template_id: r.score for r in dense_results}
        
        # Combine scores
        combined_results = []
        
        for template_id in candidate_template_ids:
            bm25_score = bm25_scores.get(template_id, 0.0)
            dense_score = dense_scores.get(template_id, 0.0)
            
            # Weighted combination
            combined_score = self.alpha * bm25_score + self.beta * dense_score
            
            combined_results.append(RetrievalResult(
                template_id=template_id,
                log_id=None,
                score=combined_score,
                rank=0,  # Will be set after sorting
                method=self.get_name(),
                debug={
                    'bm25_score': bm25_score,
                    'dense_score': dense_score,
                    'alpha': self.alpha,
                    'beta': self.beta
                }
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda r: r.score, reverse=True)
        
        # Update ranks and truncate to top_k
        for rank, result in enumerate(combined_results[:top_k], start=1):
            result.rank = rank
        
        return combined_results[:top_k]
    
    def get_name(self) -> str:
        """Get retriever name."""
        return "Hybrid"
    
    def get_config(self) -> dict:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "stage1_k": self.stage1_k,
            "alpha": self.alpha,
            "beta": self.beta,
            "bm25_config": self.bm25_retriever.get_config(),
            "dense_config": self.dense_retriever.get_config()
        })
        return config
