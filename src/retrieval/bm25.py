"""
BM25 retriever for LAR-RAG baselines.

Pure lexical search using BM25 ranking over template text.
Template-first architecture for INFOCOM compliance.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from rank_bm25 import BM25Okapi

from src.retrieval.base import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """
    BM25-only retriever for template-level search.
    
    Architecture:
    1. Index templates with BM25
    2. Query returns ranked template_ids
    3. Optionally expand to log-level results
    """
    
    def __init__(
        self,
        templates_file: str = "data/processed/bgl/templates.jsonl",
        template_map_file: Optional[str] = None,
        k1: float = 1.2,
        b: float = 0.75,
        expand_to_logs: bool = False
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            templates_file: Path to templates JSONL file
            template_map_file: Path to template-to-log mapping (for log expansion)
            k1: BM25 term saturation parameter (default: 1.2)
            b: BM25 length normalization parameter (default: 0.75)
            expand_to_logs: Expand templates to logs (default: False)
        """
        self.templates_file = Path(templates_file)
        self.template_map_file = Path(template_map_file) if template_map_file else None
        self.k1 = k1
        self.b = b
        self.expand_to_logs = expand_to_logs
        
        # Load templates
        logger.info(f"Loading templates from {templates_file}...")
        self.templates = []
        self.template_ids = []
        
        with open(self.templates_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    template = json.loads(line)
                    self.templates.append(template['template'])
                    self.template_ids.append(template['template_id'])
        
        logger.info(f"Loaded {len(self.templates)} templates")
        
        # Tokenize templates for BM25
        logger.info("Tokenizing templates for BM25...")
        tokenized_corpus = [template.lower().split() for template in self.templates]
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        logger.info("✅ BM25 index ready")
        
        # Load template map if expanding to logs
        self.template_to_logs = None
        if self.expand_to_logs:
            if not self.template_map_file or not self.template_map_file.exists():
                raise ValueError("template_map_file required for log expansion")
            
            logger.info(f"Loading template map from {self.template_map_file}...")
            self.template_to_logs = defaultdict(list)
            
            with open(self.template_map_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        mapping = json.loads(line)
                        self.template_to_logs[mapping['template_id']].append({
                            'log_id': mapping['log_id'],
                            'timestamp': mapping['timestamp'],
                            'node': mapping['node']
                        })
            
            logger.info(f"Loaded mappings for {len(self.template_to_logs)} templates")
    
    def retrieve(
        self,
        query_text: str,
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve templates using BM25.
        
        Args:
            query_text: Query text
            incident_time: Ignored (BM25 doesn't use time)
            incident_node: Ignored (BM25 doesn't use topology)
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # Tokenize query
        tokenized_query = query_text.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        
        for rank, idx in enumerate(top_indices, start=1):
            template_id = self.template_ids[idx]
            score = float(scores[idx])
            
            if self.expand_to_logs and self.template_to_logs:
                # Expand to logs
                logs = self.template_to_logs.get(template_id, [])
                
                for log in logs:
                    results.append(RetrievalResult(
                        template_id=template_id,
                        log_id=log['log_id'],
                        score=score,
                        rank=rank,
                        method=self.get_name(),
                        debug={
                            'bm25_score': score,
                            'template_rank': rank
                        }
                    ))
            else:
                # Template-level results
                results.append(RetrievalResult(
                    template_id=template_id,
                    log_id=None,
                    score=score,
                    rank=rank,
                    method=self.get_name(),
                    debug={'bm25_score': score}
                ))
        
        return results
    
    def get_name(self) -> str:
        """Get retriever name."""
        return "BM25"
    
    def get_config(self) -> dict:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "k1": self.k1,
            "b": self.b,
            "expand_to_logs": self.expand_to_logs,
            "num_templates": len(self.templates)
        })
        return config
