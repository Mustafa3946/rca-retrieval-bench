"""
Retriever Factory with INFOCOM-Compliant Constraints

ENFORCED RULES:
1. Only v2 Parquet embeddings allowed (no JSONL)
2. No mixing of Azure Search and local retrievers
3. Fail fast on invalid configurations
4. Clear error messages for reviewers
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.retrieval.base import Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lar_rag_v2 import LARRAGRankerV2

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """
    Factory for creating INFOCOM-compliant retrievers.
    
    Enforces:
    - Local-first retrievers only
    - v2 Parquet embeddings
    - No Azure Search mixing
    """
    
    ALLOWED_TYPES = {
        'bm25': BM25Retriever,
        'dense': DenseRetriever,
        'hybrid': HybridRetriever,
        'lar_rag': LARRAGRankerV2
    }
    
    FORBIDDEN_TYPES = {
        'azure_search',
        'ai_search',
        'cognitive_search'
    }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate retriever configuration for INFOCOM compliance.
        
        Raises:
            ValueError: If configuration violates constraints
        """
        retriever_type = config.get('type', '').lower()
        
        # Check forbidden types
        if retriever_type in RetrieverFactory.FORBIDDEN_TYPES:
            raise ValueError(
                f"❌ INFOCOM VIOLATION: Azure Search retrievers not allowed in local experiments.\n"
                f"   Found type: '{retriever_type}'\n"
                f"   Allowed types: {list(RetrieverFactory.ALLOWED_TYPES.keys())}\n"
                f"   Rationale: Experiments must be fully reproducible offline."
            )
        
        # Check allowed types
        if retriever_type not in RetrieverFactory.ALLOWED_TYPES:
            raise ValueError(
                f"❌ Unknown retriever type: '{retriever_type}'\n"
                f"   Allowed types: {list(RetrieverFactory.ALLOWED_TYPES.keys())}"
            )
        
        # Validate embeddings path (must be v2 Parquet or None)
        embeddings_path = config.get('embeddings_path', config.get('index_dir'))
        
        if embeddings_path:
            embeddings_path = Path(embeddings_path)
            
            # Check for forbidden JSONL embeddings
            if str(embeddings_path).endswith('.jsonl'):
                raise ValueError(
                    f"❌ INFOCOM VIOLATION: JSONL embeddings not allowed.\n"
                    f"   Found: {embeddings_path}\n"
                    f"   Required: v2 Parquet format (shards + manifest.json)\n"
                    f"   Rationale: Single canonical embedding format for reproducibility."
                )
            
            # For dense/hybrid/lar_rag, validate v2 Parquet structure
            if retriever_type in ['dense', 'hybrid', 'lar_rag']:
                if not embeddings_path.exists():
                    raise ValueError(
                        f"❌ Embeddings path does not exist: {embeddings_path}\n"
                        f"   Required for retriever type: '{retriever_type}'"
                    )
                
                # Check for v2 Parquet markers
                index_dir = embeddings_path if embeddings_path.is_dir() else embeddings_path.parent
                manifest = index_dir / 'manifest.json'
                
                if not manifest.exists():
                    # Check if it's a FAISS index directory
                    faiss_index = index_dir / 'index.faiss'
                    if not faiss_index.exists():
                        raise ValueError(
                            f"❌ Invalid index directory: {index_dir}\n"
                            f"   Missing manifest.json (for Parquet shards)\n"
                            f"   Missing index.faiss (for FAISS index)\n"
                            f"   Required: v2 Parquet embeddings or built FAISS index"
                        )
        
        logger.info(f"✅ Configuration validated: {retriever_type}")
    
    @staticmethod
    def create(config: Dict[str, Any]) -> Retriever:
        """
        Create retriever from configuration.
        
        Args:
            config: Retriever configuration dict
            
        Returns:
            Configured Retriever instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate first
        RetrieverFactory.validate_config(config)
        
        retriever_type = config['type'].lower()
        retriever_class = RetrieverFactory.ALLOWED_TYPES[retriever_type]
        
        # Extract common parameters
        templates_file = config.get('templates_file', 'data/processed/bgl/templates.jsonl')
        
        # Type-specific instantiation
        if retriever_type == 'bm25':
            return retriever_class(
                templates_file=templates_file,
                template_map_file=config.get('template_map_file')
            )
        
        elif retriever_type == 'dense':
            return retriever_class(
                templates_file=templates_file,
                index_dir=config['index_dir'],
                embedding_model=config.get('embedding_model', 'text-embedding-3-small'),
                insecure_ssl=config.get('insecure_ssl', False)
            )
        
        elif retriever_type == 'hybrid':
            return retriever_class(
                templates_file=templates_file,
                index_dir=config['index_dir'],
                template_map_file=config.get('template_map_file'),
                embedding_model=config.get('embedding_model', 'text-embedding-3-small'),
                expand_to_logs=config.get('expand_to_logs', False),
                stage1_k=config.get('stage1_k', 200),
                bm25_weight=config.get('bm25_weight', 0.3),
                dense_weight=config.get('dense_weight', 0.7)
            )
        
        elif retriever_type == 'lar_rag':
            # Validate occurrence store
            occurrence_store_path = config.get('occurrence_store_path')
            if not occurrence_store_path:
                raise ValueError(
                    f"❌ LAR-RAG requires 'occurrence_store_path'\n"
                    f"   Build it with: python -m src.storage.occurrence_store --template-map <path> --output <path>"
                )
            
            if not Path(occurrence_store_path).exists():
                raise ValueError(
                    f"❌ Occurrence store not found: {occurrence_store_path}\n"
                    f"   Build it with: python -m src.storage.occurrence_store --template-map <path> --output <path>"
                )
            
            return retriever_class(
                templates_file=templates_file,
                index_dir=config['index_dir'],
                occurrence_store_path=occurrence_store_path,
                embedding_model=config.get('embedding_model', 'text-embedding-3-small'),
                alpha=config.get('alpha', 0.4),
                beta=config.get('beta', 0.3),
                gamma=config.get('gamma', 0.3),
                lambda_pre=config.get('lambda_pre', 0.0001),
                lambda_post=config.get('lambda_post', 0.001),
                lambda_g=config.get('lambda_g', 0.5),
                use_temporal=config.get('use_temporal', True),
                use_topology=config.get('use_topology', True),
                topology_mode=config.get('topology_mode', 'synthetic'),
                topology_file=config.get('topology_file'),
                stage1_k=config.get('stage1_k', 200),
                normalize_scores=config.get('normalize_scores', True)
            )
        
        else:
            raise ValueError(f"Unhandled retriever type: {retriever_type}")


def validate_experiment_config(config_path: str) -> None:
    """
    Validate entire experiment configuration file.
    
    Ensures:
    - All methods use local retrievers
    - No Azure Search mixing
    - v2 Parquet embeddings
    
    Args:
        config_path: Path to experiment YAML config
        
    Raises:
        ValueError: If configuration violates INFOCOM constraints
    """
    import yaml
    
    logger.info("="*80)
    logger.info("VALIDATING EXPERIMENT CONFIGURATION (INFOCOM COMPLIANCE)")
    logger.info("="*80)
    logger.info(f"Config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check methods
    methods = config.get('methods', {})
    
    if not methods:
        raise ValueError("❌ No methods defined in configuration")
    
    logger.info(f"Found {len(methods)} methods to validate")
    
    for method_name, method_config in methods.items():
        logger.info(f"\n  Validating method: {method_name}")
        
        try:
            RetrieverFactory.validate_config(method_config)
            logger.info(f"    ✅ {method_name}: Valid")
        except ValueError as e:
            logger.error(f"    ❌ {method_name}: {str(e)}")
            raise
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL METHODS VALIDATED - INFOCOM COMPLIANT")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate experiment configuration for INFOCOM compliance"
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to experiment YAML configuration'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    try:
        validate_experiment_config(args.config)
    except ValueError as e:
        logger.error(f"\n❌ VALIDATION FAILED:\n{str(e)}")
        exit(1)
