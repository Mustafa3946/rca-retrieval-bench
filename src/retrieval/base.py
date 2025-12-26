"""
Base retriever interface for LAR-RAG.

Defines a common interface for all retrieval methods to enable
fair comparison and ablation studies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    Single retrieval result.
    
    Attributes:
        template_id: Template identifier
        log_id: Log line identifier (if expanded to logs)
        score: Retrieval score (higher = more relevant)
        rank: Rank in result list (1-indexed)
        method: Retrieval method name
        debug: Optional debug information
    """
    template_id: str
    log_id: Optional[str] = None
    score: float = 0.0
    rank: int = 0
    method: str = ""
    debug: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "log_id": self.log_id,
            "score": self.score,
            "rank": self.rank,
            "method": self.method,
            "debug": self.debug
        }


class Retriever(ABC):
    """
    Abstract base class for retrieval methods.
    
    All retrievers must implement:
    - retrieve(): Main retrieval method
    - get_name(): Return method name for reporting
    """
    
    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant templates/logs for a query.
        
        Args:
            query_text: Natural language query describing the incident
            incident_time: Timestamp of the incident (Unix epoch seconds)
            incident_node: Node/host identifier where incident occurred
            top_k: Number of results to return
            **kwargs: Additional method-specific parameters
            
        Returns:
            List of RetrievalResult objects, sorted by score (descending)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get retriever name for reporting.
        
        Returns:
            Method name string (e.g., "BM25", "Dense", "LAR-RAG")
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get retriever configuration for reproducibility.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            "method": self.get_name(),
            "class": self.__class__.__name__
        }
