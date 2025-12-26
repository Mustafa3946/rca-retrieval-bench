"""
Dense retriever using FAISS vector search.

Pure semantic search using embedding similarity.
Template-first architecture for INFOCOM compliance (rev6 - offline mode).

OFFLINE MODE (local_only=True):
- NO Azure OpenAI calls during retrieval
- Queries must provide query_template_id
- Uses precomputed embeddings from Parquet shards
- Fully reproducible, deterministic

ONLINE MODE (local_only=False):
- Embeds free-form query_text via Azure OpenAI
- NOT suitable for paper runs (non-deterministic, requires API)
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AzureOpenAI
from collections import defaultdict
import numpy as np
import os
from dotenv import load_dotenv

from src.retrieval.base import Retriever, RetrievalResult
from src.indexing.faiss_index import FAISSIndexLoader
from src.embeddings.embedding_cache import EmbeddingCache

load_dotenv()

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """
    Dense retriever using FAISS for vector search.
    
    Architecture:
    1. Embed query using precomputed cache (local_only=True) OR Azure OpenAI (local_only=False)
    2. Search FAISS index for nearest neighbors
    3. Return ranked template_ids
    4. Optionally expand to log-level results
    
    For INFOCOM paper runs: MUST use local_only=True with query_template_id
    """
    
    def __init__(
        self,
        index_dir: str,
        embedding_model: str = "text-embedding-3-small",
        template_map_file: Optional[str] = None,
        expand_to_logs: bool = False,
        local_only: bool = True,  # NEW: enforce offline mode
        embeddings_manifest: Optional[str] = None,  # NEW: for offline cache
        embeddings_shards_dir: Optional[str] = None,  # NEW: for offline cache
        max_cached_shards: int = 16,  # NEW: bound memory usage (rev9)
        openai_client: Optional["AzureOpenAI"] = None,
        insecure_ssl: bool = False
    ):
        """
        Initialize dense retriever.
        
        Args:
            index_dir: Directory containing FAISS index
            embedding_model: OpenAI embedding model name
            template_map_file: Path to template-to-log mapping (for log expansion)
            expand_to_logs: Expand templates to logs (default: False)
            local_only: If True, reject online API calls (default: True for reproducibility)
            embeddings_manifest: Path to manifest.json (required if local_only=True)
            embeddings_shards_dir: Directory with shard_*.parquet (required if local_only=True)
            max_cached_shards: Max shards in memory (default: 16, bounds memory for INFOCOM runs)
            openai_client: Optional pre-configured OpenAI client (only if local_only=False)
            insecure_ssl: Disable SSL verification (only for corporate proxies)
        """
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.template_map_file = Path(template_map_file) if template_map_file else None
        self.expand_to_logs = expand_to_logs
        self.local_only = local_only
        self.insecure_ssl = insecure_ssl
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_dir}...")
        self.index_loader = FAISSIndexLoader(str(index_dir))
        logger.info("✅ FAISS index loaded")
        
        # Initialize embedding source
        if self.local_only:
            # Offline mode: Use precomputed embeddings
            if not embeddings_manifest or not embeddings_shards_dir:
                raise ValueError(
                    "local_only=True requires embeddings_manifest and embeddings_shards_dir.\n"
                    "For paper runs, provide paths to precomputed embeddings."
                )
            
            logger.info("🔒 OFFLINE MODE: Using precomputed embeddings (no Azure API calls)")
            self.embedding_cache = EmbeddingCache(
                embeddings_manifest, 
                embeddings_shards_dir,
                max_cached_shards=max_cached_shards
            )
            self.openai_client = None
        else:
            # Online mode: Use Azure OpenAI API (lazy-loaded)
            logger.warning("⚠️  ONLINE MODE: Will call Azure OpenAI API at runtime (non-deterministic)")
            self.embedding_cache = None
            
            if openai_client:
                self.openai_client = openai_client
            else:
                # Lazy import: Only load openai when actually needed for Azure mode
                from openai import AzureOpenAI
                
                if insecure_ssl:
                    import httpx
                    import ssl
                    logger.warning("⚠️  SSL verification disabled (insecure_ssl=True)")
                    ssl._create_default_https_context = ssl._create_unverified_context
                    os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"
                    http_client = httpx.Client(verify=False)
                else:
                    http_client = None
                
                self.openai_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    http_client=http_client
                )
        
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
                            'node': mapping.get('node') or mapping.get('node_id', 'unknown')
                        })
            
            logger.info(f"Loaded mappings for {len(self.template_to_logs)} templates")
    
    def _get_query_embedding(
        self,
        query_text: Optional[str] = None,
        query_template_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Get query embedding (offline cache OR online API).
        
        Args:
            query_text: Free-form query text (only if local_only=False)
            query_template_id: Template ID for precomputed embedding (required if local_only=True)
            
        Returns:
            Query embedding vector
            
        Raises:
            RuntimeError: If local_only=True and query_text provided without template_id
            ValueError: If neither query provided
        """
        if self.local_only:
            # Offline mode: MUST use template_id
            if not query_template_id:
                raise ValueError(
                    "local_only=True requires query_template_id for template-first retrieval.\n"
                    "For paper runs, use precomputed embeddings."
                )
            
            # Ignore query_text if template_id is provided (BM25/Hybrid need text, Dense doesn't)
            embedding = self.embedding_cache.get_embedding_by_template_id(query_template_id)
            if embedding is None:
                raise ValueError(f"Template ID not found in cache: {query_template_id}")
            
            return embedding
        else:
            # Online mode: Use Azure OpenAI API
            if not query_text:
                raise ValueError("local_only=False requires query_text")
            
            return self._embed_query_online(query_text)
    
    def _embed_query_online(self, query_text: str) -> np.ndarray:
        """
        Embed query text using Azure OpenAI API (ONLINE MODE ONLY).
        
        Args:
            query_text: Query text
            
        Returns:
            Embedding vector
        """
        if self.local_only:
            raise RuntimeError("Cannot call _embed_query_online when local_only=True")
        
        try:
            response = self.openai_client.embeddings.create(
                input=query_text,
                model=self.embedding_model
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_template_id: Optional[str] = None,  # NEW: for offline mode
        incident_time: Optional[float] = None,
        incident_node: Optional[str] = None,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve templates using dense vector search.
        
        Args:
            query_text: Free-form query text (only if local_only=False)
            query_template_id: Template ID for offline mode (required if local_only=True)
            incident_time: Ignored (dense doesn't use time)
            incident_node: Ignored (dense doesn't use topology)
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # Get query embedding
        query_vec = self._get_query_embedding(query_text, query_template_id)
        
        # Search FAISS
        logger.debug(f"Searching FAISS for top-{top_k} templates...")
        search_results = self.index_loader.search(query_vec, top_k)
        
        # Build results
        results = []
        
        for rank, (template_id, distance) in enumerate(search_results, start=1):
            # Convert L2 distance to similarity score
            # For normalized vectors: cos_sim = 1 - (L2_dist^2 / 2)
            # Approximate: similarity ≈ 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)
            
            if self.expand_to_logs and self.template_to_logs:
                # Expand to logs
                logs = self.template_to_logs.get(template_id, [])
                
                for log in logs:
                    results.append(RetrievalResult(
                        template_id=template_id,
                        log_id=log['log_id'],
                        score=similarity,
                        rank=rank,
                        method=self.get_name(),
                        debug={
                            'faiss_distance': float(distance),
                            'similarity': similarity,
                            'template_rank': rank,
                            'offline_mode': self.local_only
                        }
                    ))
            else:
                # Template-level results
                results.append(RetrievalResult(
                    template_id=template_id,
                    log_id=None,
                    score=similarity,
                    rank=rank,
                    method=self.get_name(),
                    debug={
                        'faiss_distance': float(distance),
                        'similarity': similarity,
                        'offline_mode': self.local_only
                    }
                ))
        
        return results
    
    def get_name(self) -> str:
        """Get retriever name."""
        return "Dense"
    
    def get_config(self) -> dict:
        """Get configuration."""
        config = super().get_config()
        config.update({
            "embedding_model": self.embedding_model,
            "index_dir": str(self.index_dir),
            "expand_to_logs": self.expand_to_logs,
            "index_metadata": self.index_loader.metadata
        })
        return config
