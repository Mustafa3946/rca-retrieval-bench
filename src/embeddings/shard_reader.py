"""
EmbeddingShardReader: Load embeddings from sharded Parquet files.

Reads template embeddings stored in sharded format with manifest.
Supports streaming, counting, and indexed lookup.
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple
import pyarrow.parquet as pq
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingShardReader:
    """
    Read template embeddings from sharded Parquet files.
    
    Directory structure:
        embeddings_dir/
            manifest.json
            part-00000.parquet
            part-00001.parquet
            ...
    
    Manifest format:
        {
            "model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "total_templates": 1169037,
            "shard_count": 117,
            ...
        }
    
    Parquet schema:
        - template_id: string
        - embedding: list<float> or binary
        - template (optional): string
    """
    
    def __init__(self, embeddings_dir: str):
        """
        Initialize shard reader.
        
        Args:
            embeddings_dir: Directory containing manifest.json and shard files
        """
        self.embeddings_dir = Path(embeddings_dir)
        
        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
        # Load manifest
        manifest_path = self.embeddings_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Validate manifest
        required_fields = ["shard_count", "embedding_dimension"]
        for field in required_fields:
            if field not in self.manifest:
                raise ValueError(f"Manifest missing required field: {field}")
        
        # Handle different total count field names for backward compatibility
        self.total_templates = (
            self.manifest.get("successfully_embedded") or
            self.manifest.get("total_templates") or
            self.manifest.get("num_embeddings") or
            self.manifest.get("templates_total") or
            0
        )
        
        if self.total_templates == 0:
            logger.warning("Could not find total template count in manifest")
            logger.warning(f"Manifest keys: {list(self.manifest.keys())}")
        
        # List shard files
        self.shard_files = sorted(self.embeddings_dir.glob("part-*.parquet"))
        
        manifest_shard_count = self.manifest.get("shard_count", 0)
        if len(self.shard_files) != manifest_shard_count:
            logger.warning(
                f"Shard count mismatch: manifest says {manifest_shard_count}, "
                f"found {len(self.shard_files)} files"
            )
        
        logger.info(
            f"Initialized EmbeddingShardReader: {len(self.shard_files)} shards, "
            f"{self.total_templates:,} templates, "
            f"dim={self.manifest['embedding_dimension']}"
        )
    
    def iter_embeddings(self) -> Iterator[Tuple[str, np.ndarray, Optional[str]]]:
        """
        Stream all embeddings from shards.
        
        Yields:
            (template_id, embedding_vector, template_text)
        """
        for shard_file in self.shard_files:
            try:
                table = pq.read_table(shard_file)
                
                # Convert to pandas for easier access
                df = table.to_pandas()
                
                for _, row in df.iterrows():
                    template_id = row['template_id']
                    embedding = np.array(row['embedding'], dtype=np.float32)
                    template_text = row.get('template', None)
                    
                    yield template_id, embedding, template_text
                    
            except Exception as e:
                logger.error(f"Error reading shard {shard_file}: {e}")
                raise
    
    def count_embeddings(self) -> int:
        """
        Count total embeddings across all shards.
        
        Fast method that reads row counts from Parquet metadata.
        
        Returns:
            Total number of embeddings
        """
        total = 0
        
        for shard_file in self.shard_files:
            try:
                # Read metadata only (fast)
                parquet_file = pq.ParquetFile(shard_file)
                total += parquet_file.metadata.num_rows
            except Exception as e:
                logger.error(f"Error counting rows in {shard_file}: {e}")
                raise
        
        return total
    
    def get_embedding(self, template_id: str) -> Optional[np.ndarray]:
        """
        Lookup embedding for a specific template_id.
        
        Note: This is a linear scan and may be slow for large datasets.
        For batch lookups, use iter_embeddings() instead.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Embedding vector or None if not found
        """
        for tid, emb, _ in self.iter_embeddings():
            if tid == template_id:
                return emb
        
        return None
    
    def get_embeddings_batch(self, template_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Lookup embeddings for multiple template_ids.
        
        More efficient than calling get_embedding() repeatedly.
        
        Args:
            template_ids: List of template identifiers
            
        Returns:
            Dictionary mapping template_id -> embedding
        """
        target_ids = set(template_ids)
        results = {}
        
        for tid, emb, _ in self.iter_embeddings():
            if tid in target_ids:
                results[tid] = emb
                
                # Early exit if we found all
                if len(results) == len(target_ids):
                    break
        
        return results
    
    def validate(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Validate embeddings integrity.
        
        Checks:
        - Total count matches manifest
        - All embeddings have correct dimensionality
        - No duplicate template_ids
        - All vectors are valid (no NaN/Inf)
        
        Args:
            verbose: Print detailed validation info
            
        Returns:
            Validation report dictionary
        """
        expected_count = self.total_templates
        expected_dim = self.manifest["embedding_dimension"]
        
        actual_count = 0
        seen_ids = set()
        invalid_vectors = []
        wrong_dims = []
        
        logger.info("Validating embeddings...")
        
        for template_id, embedding, _ in self.iter_embeddings():
            actual_count += 1
            
            # Check for duplicates
            if template_id in seen_ids:
                logger.error(f"Duplicate template_id: {template_id}")
            seen_ids.add(template_id)
            
            # Check dimensionality
            if len(embedding) != expected_dim:
                wrong_dims.append((template_id, len(embedding)))
            
            # Check for invalid values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                invalid_vectors.append(template_id)
            
            # Progress
            if verbose and actual_count % 100000 == 0:
                logger.info(f"  Validated {actual_count:,} embeddings...")
        
        # Build report
        count_match = (expected_count == 0) or (actual_count == expected_count)
        
        report = {
            "expected_count": expected_count,
            "actual_count": actual_count,
            "count_match": count_match,
            "expected_dim": expected_dim,
            "wrong_dims_count": len(wrong_dims),
            "invalid_vectors_count": len(invalid_vectors),
            "duplicate_ids_count": actual_count - len(seen_ids),
            "valid": (
                count_match and
                len(wrong_dims) == 0 and
                len(invalid_vectors) == 0 and
                actual_count == len(seen_ids)
            )
        }
        
        if verbose or not report["valid"]:
            logger.info(f"Validation Report:")
            logger.info(f"  Expected count: {expected_count:,}")
            logger.info(f"  Actual count: {actual_count:,}")
            logger.info(f"  Count match: {count_match}")
            logger.info(f"  Wrong dimensions: {len(wrong_dims)}")
            logger.info(f"  Invalid vectors: {len(invalid_vectors)}")
            logger.info(f"  Duplicate IDs: {actual_count - len(seen_ids)}")
            logger.info(f"  Valid: {report['valid']}")
            
            if wrong_dims:
                logger.warning(f"Templates with wrong dimensions: {wrong_dims[:10]}")
            
            if invalid_vectors:
                logger.warning(f"Templates with invalid vectors: {invalid_vectors[:10]}")
        
        return report
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get shard reader information.
        
        Returns:
            Dictionary with reader metadata
        """
        return {
            "embeddings_dir": str(self.embeddings_dir),
            "manifest": self.manifest,
            "shard_files_count": len(self.shard_files),
            "shard_files": [f.name for f in self.shard_files[:5]] + (
                ["..."] if len(self.shard_files) > 5 else []
            )
        }
