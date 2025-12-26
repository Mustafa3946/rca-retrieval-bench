"""
Offline BM25 retriever using prebuilt disk-backed index (rev6 - local paper run).

NO Azure AI Search calls - uses pickled rank-bm25 index.

RATIONALE:
- INFOCOM paper run must be fully offline
- BM25 index prebuilt via scripts/build_bm25_index.py
- Deterministic, reproducible lexical baseline

Usage:
    retriever = OfflineBM25Retriever(index_dir="data/processed/bgl/indices/bm25")
    results = retriever.retrieve(query_text="kernel panic error", top_k=10)
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np

from src.retrieval.base import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


class OfflineBM25Retriever(Retriever):
    """
    Offline BM25 retriever using prebuilt index.
    
    NO online API calls - fully local, deterministic baseline.
    """
    
    def __init__(self, index_dir: str):
        """
        Initialize offline BM25 retriever.
        
        Args:
            index_dir: Directory containing:
                - bm25_index.pkl (pickled BM25Okapi object)
                - template_ids.json (template ID mapping)
        """
        self.index_dir = Path(index_dir)
        
        # Load BM25 index
        index_path = self.index_dir / "bm25_index.pkl"
        ids_path = self.index_dir / "template_ids.json"
        
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {index_path}\n"
                f"Build it first: python scripts/build_bm25_index.py"
            )
        
        if not ids_path.exists():
            raise FileNotFoundError(
                f"Template IDs file not found: {ids_path}\n"
                f"Build it first: python scripts/build_bm25_index.py"
            )
        
        logger.info(f"Loading BM25 index from {index_path}...")
        with open(index_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        logger.info(f"Loading template IDs from {ids_path}...")
        with open(ids_path, 'r') as f:
            self.template_ids = json.load(f)
        
        if len(self.template_ids) != self.bm25.corpus_size:
            logger.warning(f"Template IDs mismatch: {len(self.template_ids)} IDs vs {self.bm25.corpus_size} corpus size")
        
        logger.info(f"✅ BM25 index loaded: {len(self.template_ids)} templates")
    
    def retrieve(
        self,
        query_text: str,
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve templates using BM25 lexical search.
        
        Args:
            query_text: Query text
            incident_time: Ignored (BM25 doesn't use time)
            incident_node: Ignored (BM25 doesn't use topology)
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # Tokenize query
        query_tokens = query_text.split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            template_id = self.template_ids[idx]
            score = float(scores[idx])
            
            results.append(RetrievalResult(
                template_id=template_id,
                log_id=None,
                score=score,
                rank=rank,
                method=self.get_name(),
                debug={
                    'bm25_score': score,
                    'query_tokens': len(query_tokens)
                }
            ))
        
        return results
    
    def get_name(self) -> str:
        """Get retriever name."""
        return "BM25"
    
    def get_config(self) -> dict:
        """Get retriever configuration."""
        return {
            'type': 'bm25',
            'offline': True,
            'index_dir': str(self.index_dir),
            'num_templates': len(self.template_ids)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test offline BM25 retriever")
    parser.add_argument('--index-dir', required=True, help='BM25 index directory')
    parser.add_argument('--query', required=True, help='Query text')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    retriever = OfflineBM25Retriever(args.index_dir)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    
    print(f"\n=== Top-{args.top_k} Results ===")
    for r in results:
        print(f"Rank {r.rank}: {r.template_id} (score={r.score:.4f})")
