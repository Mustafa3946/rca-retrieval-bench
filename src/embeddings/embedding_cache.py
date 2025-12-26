"""
Embedding cache for offline retrieval (INFOCOM rev6 - local paper run).

Provides template-first query embedding WITHOUT online API calls.
Reads from precomputed Parquet shards.

RATIONALE:
- INFOCOM paper run must be fully offline (no Azure calls during eval)
- Queries are templates, not raw text → use precomputed embeddings
- Deterministic, reproducible results

Usage:
    cache = EmbeddingCache(manifest_path, shards_dir)
    query_vec = cache.get_embedding_by_template_id("bgl_tpl_00123")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
from collections import OrderedDict
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShardData:
    """
    Cached shard data for O(1) embedding lookups.
    
    Attributes:
        table: Arrow table with template_id and embedding columns
        id_to_row: Dict mapping template_id to row index (O(1) lookup)
    """
    table: pa.Table
    id_to_row: Dict[str, int]


class EmbeddingCache:
    """
    Offline embedding cache using precomputed Parquet shards.
    
    NO online API calls - template-first architecture only.
    
    Performance:
    - O(1) template_id lookups via dict index (critical for INFOCOM latency evaluation)
    - Real bounded LRU shard caching (rev10 fix)
    - Shard-level loading (not full dataset load)
    """
    
    def __init__(self, manifest_path: str, shards_dir: str, max_cached_shards: int = 16):
        """
        Initialize embedding cache.
        
        Args:
            manifest_path: Path to manifest.json
            shards_dir: Directory containing shard_*.parquet files
            max_cached_shards: Maximum number of shards to keep in memory (default: 16)
        """
        self.manifest_path = Path(manifest_path)
        self.shards_dir = Path(shards_dir)
        
        # Validate max_cached_shards
        if max_cached_shards <= 0:
            raise ValueError(f"max_cached_shards must be > 0, got {max_cached_shards}")
        self.max_cached_shards = max_cached_shards
        
        # Real LRU cache using OrderedDict (rev10 fix)
        self._shard_cache: OrderedDict[int, ShardData] = OrderedDict()
        
        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Embeddings manifest not found: {manifest_path}\n"
                f"Generate embeddings first using scripts/embed_templates_to_blob.py"
            )
        
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Discover shard files (actual format uses part-*.parquet, not embedded shard list)
        self.shard_files = sorted(self.shards_dir.glob("part-*.parquet"))
        
        if not self.shard_files:
            raise FileNotFoundError(
                f"No parquet shards found in {shards_dir}\n"
                f"Expected files matching pattern: part-*.parquet"
            )
        
        # Build template_id -> shard_index mapping by reading first row of each shard
        # This is O(num_shards) but only happens once at init
        self.template_to_shard = {}
        for shard_idx, shard_path in enumerate(self.shard_files):
            # Read shard metadata to get template IDs
            table = pq.read_table(shard_path, columns=['template_id'])
            template_ids = table.column('template_id').to_pylist()
            
            for template_id in template_ids:
                self.template_to_shard[template_id] = shard_idx
        
        logger.info(f"Loaded manifest: {len(self.template_to_shard)} template IDs across {len(self.shard_files)} shards")
        logger.info(f"Max cached shards: {max_cached_shards} (real bounded LRU)")
    
    def _get_shard(self, shard_idx: int) -> ShardData:
        """
        Get shard data with real bounded LRU eviction (rev10 fix).
        
        Returns:
            ShardData with table and O(1) id_to_row index
            
        Performance:
        - Builds dict index ONCE per shard (O(n) amortized to O(1) per lookup)
        - Real LRU eviction when cache exceeds max_cached_shards
        - Critical for INFOCOM latency evaluation at scale
        """
        # Check if shard is in cache
        if shard_idx in self._shard_cache:
            # Move to end (most recently used)
            self._shard_cache.move_to_end(shard_idx)
            logger.debug(f"Cache hit: shard {shard_idx}")
            return self._shard_cache[shard_idx]
        
        # Cache miss - load shard
        shard_path = self.shard_files[shard_idx]
        
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        
        table = pq.read_table(shard_path)
        
        # Build O(1) index: template_id -> row_idx
        # to_pylist() is acceptable here: called once per shard load,
        # and shard-level LRU eviction bounds total cost to O(max_cached_shards).
        template_ids = table.column('template_id').to_pylist()
        id_to_row = {tid: idx for idx, tid in enumerate(template_ids)}
        
        shard_data = ShardData(table=table, id_to_row=id_to_row)
        
        logger.debug(f"Loaded shard {shard_idx}: {len(id_to_row)} templates")
        
        # Insert into cache
        self._shard_cache[shard_idx] = shard_data
        self._shard_cache.move_to_end(shard_idx)
        
        # Evict LRU shard if exceeding max_cached_shards
        if len(self._shard_cache) > self.max_cached_shards:
            evicted_idx, _ = self._shard_cache.popitem(last=False)
            logger.debug(f"Evicted LRU shard {evicted_idx} (cache size: {len(self._shard_cache)}/{self.max_cached_shards})")
        
        return shard_data
    
    def get_embedding_by_template_id(self, template_id: str) -> Optional[np.ndarray]:
        """
        Get precomputed embedding for a template_id.
        
        Args:
            template_id: Template ID (e.g., "bgl_tpl_00123")
            
        Returns:
            Embedding vector (numpy array) or None if not found
            
        Performance:
            O(1) lookup via dict index (critical for INFOCOM latency evaluation)
        """
        # Find shard (O(1))
        shard_idx = self.template_to_shard.get(template_id)
        if shard_idx is None:
            logger.warning(f"Template ID not found in cache: {template_id}")
            return None
        
        # Get shard with real LRU (cached)
        shard_data = self._get_shard(shard_idx)
        
        # Find row (O(1) dict lookup, NOT O(n) list.index)
        row_idx = shard_data.id_to_row.get(template_id)
        if row_idx is None:
            logger.warning(f"Template ID not in shard index: {template_id}")
            return None
        
        # Extract embedding
        embedding = shard_data.table.column('embedding')[row_idx].as_py()
        return np.array(embedding, dtype=np.float32)
    
    def get_embedding_by_text(self, query_text: str) -> Optional[np.ndarray]:
        """
        OFFLINE MODE ONLY: Cannot embed free-form text.
        
        Raises:
            RuntimeError: Always raises (no online API calls allowed)
        """
        raise RuntimeError(
            "Offline embedding cache cannot embed free-form text.\n"
            "Use template-first queries with get_embedding_by_template_id().\n"
            "For paper runs, queries must be template_ids, not raw text."
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test embedding cache")
    parser.add_argument('--manifest', required=True, help='Path to manifest.json')
    parser.add_argument('--shards-dir', required=True, help='Directory containing shards')
    parser.add_argument('--template-id', required=True, help='Template ID to lookup')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    cache = EmbeddingCache(args.manifest, args.shards_dir)
    embedding = cache.get_embedding_by_template_id(args.template_id)
    
    if embedding is not None:
        print(f"✅ Found embedding: shape={embedding.shape}, dtype={embedding.dtype}")
        print(f"First 5 values: {embedding[:5]}")
    else:
        print(f"❌ Template ID not found: {args.template_id}")
