"""
Safe & Reproducible Template Embedding for LAR-RAG (v2)
========================================================

Publication-grade embedding with fixes for scale:
- Sharded Parquet output (no memory wall on resume)
- Resume keying on (template_id, content_hash)
- Robust retry with proper error detection
- Non-interactive mode for automation
- Fixed CLI semantics

See manifest.json for full provenance and reproducibility details.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import hashlib
import time
import random
import subprocess
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

from config import create_openai_client, get_azure_config
from utils.io_utils import read_json_or_jsonl

load_dotenv()

# ============================================================================
# CONFIGURATION & GUARDRAILS
# ============================================================================

# Hard limits to prevent embedding wrong data
MAX_TEMPLATES_SANITY_CHECK = 500_000  # Abort if input exceeds this
EXPECTED_BGL_TEMPLATE_RANGE = (1_000, 100_000)  # Typical range for BGL
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

# Storage efficiency
FLOAT_PRECISION = np.float32  # Use float32 instead of float64
PARQUET_COMPRESSION = 'zstd'  # Better compression than snappy
PARQUET_SHARD_SIZE = 10_000  # Embeddings per shard file

# Batch processing
DEFAULT_BATCH_SIZE = 256
MAX_RETRIES = 6
RATE_LIMIT_DELAY = 0.1  # seconds between batches


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TemplateRecord:
    """Template before embedding."""
    template_id: str
    template: str
    normalized_template: str
    content_hash: str
    count: int
    

@dataclass
class EmbeddingRecord:
    """Embedded template with metadata."""
    template_id: str
    content_hash: str
    embedding: np.ndarray
    model: str
    dimension: int
    timestamp: str


@dataclass
class EmbeddingManifest:
    """Provenance and configuration for embedding run."""
    timestamp: str
    git_commit: str
    model: str
    embedding_dimension: int
    input_file: str
    output_dir: str
    total_templates: int
    successfully_embedded: int
    failed: int
    batch_size: int
    shard_count: int
    config_snapshot: Dict
    runtime_seconds: float


# ============================================================================
# NORMALIZATION & HASHING
# ============================================================================

def normalize_template(template: str) -> str:
    """
    Normalize template for consistent hashing.
    
    - Trim whitespace
    - Collapse multiple spaces to single space
    
    Args:
        template: Raw template text
        
    Returns:
        Normalized template
    """
    normalized = template.strip()
    normalized = ' '.join(normalized.split())
    return normalized


def compute_content_hash(template: str) -> str:
    """
    Compute SHA256 hash of normalized template.
    
    This allows drift detection if template parsing changes.
    
    Args:
        template: Template text (should be normalized)
        
    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(template.encode('utf-8')).hexdigest()


# ============================================================================
# VALIDATION & GUARDRAILS
# ============================================================================

def validate_input_templates(
    templates: List[Dict],
    non_interactive: bool = False,
    allow_out_of_range: bool = False
) -> List[TemplateRecord]:
    """
    Validate and prepare templates with strict checks.
    
    Guardrails:
    - Check template count is sane
    - Verify required fields exist
    - Normalize and hash each template
    - Detect duplicates
    
    Args:
        templates: Raw template dicts from JSONL
        non_interactive: If True, fail fast on warnings instead of prompting
        allow_out_of_range: If True, skip expected range check (used with --force)
        
    Returns:
        Validated TemplateRecord objects
        
    Raises:
        ValueError: If validation fails
    """
    print("\n" + "="*80)
    print("TEMPLATE VALIDATION")
    print("="*80)
    
    # Guardrail 1: Check count
    n_templates = len(templates)
    print(f"Total templates loaded: {n_templates:,}")
    
    if n_templates == 0:
        raise ValueError("No templates found in input file!")
    
    if n_templates > MAX_TEMPLATES_SANITY_CHECK and not allow_out_of_range:
        raise ValueError(
            f"Template count {n_templates:,} exceeds sanity check limit "
            f"{MAX_TEMPLATES_SANITY_CHECK:,}. This suggests embedding raw logs instead of templates. "
            f"ABORTING to prevent expensive mistake."
        )
    
    # Guardrail 2: Check if in expected range for BGL (unless allow_out_of_range)
    min_expected, max_expected = EXPECTED_BGL_TEMPLATE_RANGE
    if not allow_out_of_range and not (min_expected <= n_templates <= max_expected):
        warning_msg = (
            f"⚠️  WARNING: Template count {n_templates:,} is outside expected range "
            f"[{min_expected:,}, {max_expected:,}] for BGL."
        )
        print(warning_msg)
        
        if non_interactive:
            # Fail fast in non-interactive mode
            raise ValueError(
                f"{warning_msg}\n"
                f"In non-interactive mode, aborting for safety. "
                f"Use --force to override (not recommended)."
            )
        else:
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response != 'yes':
                raise ValueError("User aborted due to unexpected template count.")
    elif allow_out_of_range and not (min_expected <= n_templates <= max_expected):
        print(f"⚠️  WARNING: Template count {n_templates:,} outside expected range - OVERRIDDEN by --force")
    
    # Guardrail 3: Validate schema
    print("\nValidating template schema...")
    required_fields = {'template_id', 'template'}
    
    for i, tmpl in enumerate(templates[:10]):  # Check first 10
        missing = required_fields - set(tmpl.keys())
        if missing:
            raise ValueError(
                f"Template {i} missing required fields: {missing}\n"
                f"Template: {tmpl}"
            )
    print("✓ Schema validation passed")
    
    # Guardrail 4: Process and deduplicate
    print("\nNormalizing and hashing templates...")
    records = []
    seen_hashes = {}
    
    for tmpl in templates:
        # Normalize
        normalized = normalize_template(tmpl['template'])
        
        # Hash
        content_hash = compute_content_hash(normalized)
        
        # Check for duplicates (should not happen if templates.jsonl is correct)
        if content_hash in seen_hashes:
            print(f"⚠️  Duplicate template detected:")
            print(f"   ID 1: {seen_hashes[content_hash]}")
            print(f"   ID 2: {tmpl['template_id']}")
            print(f"   Hash: {content_hash}")
            continue
        
        seen_hashes[content_hash] = tmpl['template_id']
        
        records.append(TemplateRecord(
            template_id=tmpl['template_id'],
            template=tmpl['template'],  # Keep for now, will drop after validation
            normalized_template=normalized,
            content_hash=content_hash,
            count=tmpl.get('count', 1)
        ))
    
    print(f"✓ Processed {len(records):,} unique templates")
    
    if len(records) < len(templates):
        print(f"⚠️  Removed {len(templates) - len(records)} duplicates")
    
    return records


def estimate_output_size(
    n_templates: int,
    embedding_dim: int
) -> Tuple[int, str]:
    """
    Estimate output file size for Parquet.
    
    Args:
        n_templates: Number of templates
        embedding_dim: Embedding dimension
        
    Returns:
        (size_bytes, human_readable_string)
    """
    # Parquet: float32 vectors + metadata + compression
    bytes_per_embedding = embedding_dim * 4  # float32
    bytes_per_metadata = 200  # template_id, hash, model, etc.
    bytes_per_row = bytes_per_embedding + bytes_per_metadata
    
    # Assume ~0.7 compression ratio with zstd
    estimated_bytes = int(n_templates * bytes_per_row * 0.7)
    
    # Human readable
    if estimated_bytes < 1024:
        size_str = f"{estimated_bytes} B"
    elif estimated_bytes < 1024**2:
        size_str = f"{estimated_bytes / 1024:.1f} KB"
    elif estimated_bytes < 1024**3:
        size_str = f"{estimated_bytes / (1024**2):.1f} MB"
    else:
        size_str = f"{estimated_bytes / (1024**3):.2f} GB"
    
    return estimated_bytes, size_str


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


# ============================================================================
# EMBEDDING WITH RETRY (IMPROVED)
# ============================================================================

def is_retryable_error(e: Exception) -> bool:
    """
    Check if error is retryable.
    
    Properly handles OpenAI SDK exceptions and common error patterns.
    
    Args:
        e: Exception from API call
        
    Returns:
        True if should retry
    """
    # OpenAI SDK specific exceptions
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    
    if isinstance(e, APIError):
        # Retry on 5xx server errors
        if hasattr(e, 'status_code') and e.status_code and e.status_code >= 500:
            return True
    
    # Fallback: check error message for common patterns
    error_str = str(e).lower()
    retryable_patterns = [
        'rate limit',
        'too many requests',
        '429',
        'service unavailable',
        '503',
        'timeout',
        'timed out',
        'connection',
        'temporary failure',
        '500',
        '502',
        '504'
    ]
    
    return any(pattern in error_str for pattern in retryable_patterns)


def embed_batch_with_retry(
    client: AzureOpenAI,
    texts: List[str],
    model: str,
    max_retries: int = MAX_RETRIES
) -> List[List[float]]:
    """
    Embed batch with exponential backoff retry.
    
    Improved error detection and retry logic.
    
    Args:
        client: Azure OpenAI client
        texts: List of template texts
        model: Model name
        max_retries: Maximum retry attempts
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model
            )
            return [item.embedding for item in response.data]
        
        except Exception as e:
            # Check if retryable
            if is_retryable_error(e) and attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"    Retryable error: {type(e).__name__}")
                print(f"    Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            
            # Non-retryable error or max retries reached
            raise


# ============================================================================
# RESUME LOGIC (FIXED - USES CONTENT HASH)
# ============================================================================

def load_existing_embeddings(output_dir: Path) -> Set[Tuple[str, str]]:
    """
    Load already embedded (template_id, content_hash) pairs from shards.
    
    FIXED: Now keys on both template_id AND content_hash to detect drift.
    
    Args:
        output_dir: Directory containing Parquet shards
        
    Returns:
        Set of (template_id, content_hash) tuples already embedded
    """
    if not output_dir.exists():
        return set()
    
    print(f"\nChecking for existing embeddings in: {output_dir}")
    
    existing_pairs = set()
    shard_files = sorted(output_dir.glob('part-*.parquet'))
    
    if not shard_files:
        return set()
    
    try:
        for shard_file in shard_files:
            # Read only template_id and content_hash columns (memory efficient)
            table = pq.read_table(shard_file, columns=['template_id', 'content_hash'])
            
            template_ids = table['template_id'].to_pylist()
            content_hashes = table['content_hash'].to_pylist()
            
            for tid, chash in zip(template_ids, content_hashes):
                existing_pairs.add((tid, chash))
        
        print(f"✓ Found {len(existing_pairs):,} already embedded (template_id, content_hash) pairs")
        print(f"  from {len(shard_files)} shard file(s)")
        return existing_pairs
        
    except Exception as e:
        print(f"⚠️  Could not load existing embeddings: {e}")
        return set()


def detect_drift(
    templates: List[TemplateRecord],
    existing_pairs: Set[Tuple[str, str]]
) -> Tuple[List[TemplateRecord], List[Tuple[str, str, str]]]:
    """
    Detect templates with changed content (drift).
    
    Args:
        templates: Current templates to embed
        existing_pairs: Set of (template_id, content_hash) already embedded
        
    Returns:
        (templates_to_embed, drift_cases)
        drift_cases: List of (template_id, old_hash, new_hash)
    """
    # Build lookup of existing hashes by template_id
    # FIXED: Use set to handle multiple hashes per template_id (from partial reruns)
    existing_by_id: Dict[str, Set[str]] = {}
    for tid, chash in existing_pairs:
        if tid not in existing_by_id:
            existing_by_id[tid] = set()
        existing_by_id[tid].add(chash)
    
    templates_to_embed = []
    drift_cases = []
    
    for tmpl in templates:
        pair = (tmpl.template_id, tmpl.content_hash)
        
        # Case 1: Never embedded before
        if tmpl.template_id not in existing_by_id:
            templates_to_embed.append(tmpl)
        
        # Case 2: Template ID exists with same hash - skip
        elif pair in existing_pairs:
            continue
        
        # Case 3: Template ID exists with DIFFERENT hash - DRIFT!
        else:
            # Report drift with first old hash found
            old_hashes = existing_by_id[tmpl.template_id]
            old_hash = next(iter(old_hashes))  # Get any existing hash
            drift_cases.append((tmpl.template_id, old_hash, tmpl.content_hash))
            templates_to_embed.append(tmpl)
    
    return templates_to_embed, drift_cases


# ============================================================================
# SHARDED PARQUET WRITER (FIXED - NO MEMORY WALL)
# ============================================================================

class ShardedParquetWriter:
    """
    Writes embeddings to sharded Parquet files.
    
    FIXED: No longer rewrites existing data on resume.
    Each shard is independent, avoiding memory wall.
    """
    
    def __init__(
        self,
        output_dir: Path,
        embedding_dim: int,
        model: str,
        shard_size: int = PARQUET_SHARD_SIZE
    ):
        self.output_dir = output_dir
        self.embedding_dim = embedding_dim
        self.model = model
        self.shard_size = shard_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next shard number
        existing_shards = list(self.output_dir.glob('part-*.parquet'))
        if existing_shards:
            # Extract shard numbers
            shard_nums = []
            for f in existing_shards:
                try:
                    num = int(f.stem.split('-')[1])
                    shard_nums.append(num)
                except:
                    pass
            self.next_shard_num = max(shard_nums) + 1 if shard_nums else 0
        else:
            self.next_shard_num = 0
        
        # Define schema with TRULY fixed-size list for embeddings
        # FIXED: Use pa.list_type() with size parameter for proper fixed-size list
        self.schema = pa.schema([
            ('template_id', pa.string()),
            ('content_hash', pa.string()),
            ('model', pa.string()),
            ('dimension', pa.int32()),
            ('timestamp', pa.string()),
            ('embedding', pa.list_(pa.float32(), list_size=embedding_dim))  # Fixed size
        ])
        
        self.buffer = []
        self.total_written = 0
        
    def write(self, record: EmbeddingRecord):
        """Add record to buffer."""
        self.buffer.append({
            'template_id': record.template_id,
            'content_hash': record.content_hash,
            'model': record.model,
            'dimension': record.dimension,
            'timestamp': record.timestamp,
            'embedding': record.embedding.tolist()
        })
        
        # Flush if buffer reaches shard size
        if len(self.buffer) >= self.shard_size:
            self._flush()
    
    def _flush(self):
        """Write buffer to new shard file."""
        if not self.buffer:
            return
        
        # Create shard file name
        shard_file = self.output_dir / f'part-{self.next_shard_num:05d}.parquet'
        
        # Convert buffer to table
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        
        # Write to new shard file (no rewrite of existing data!)
        pq.write_table(
            table,
            shard_file,
            compression=PARQUET_COMPRESSION
        )
        
        print(f"    ✓ Wrote shard {shard_file.name} ({len(self.buffer)} embeddings)")
        
        self.total_written += len(self.buffer)
        self.next_shard_num += 1
        self.buffer.clear()
    
    def close(self):
        """Flush remaining buffer."""
        self._flush()
        return self.total_written


# ============================================================================
# MAIN EMBEDDING FUNCTION
# ============================================================================

def embed_templates_safe(
    templates_file: str,
    output_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = True,
    overwrite: bool = False,
    non_interactive: bool = False,
    force: bool = False
):
    """
    Embed templates with publication-grade safety and provenance.
    
    IMPROVEMENTS:
    - Sharded Parquet output (no memory wall on resume)
    - Resume keys on (template_id, content_hash)
    - Robust retry with proper error detection
    - Non-interactive mode for automation
    
    Args:
        templates_file: Path to templates.jsonl
        output_dir: Directory for sharded Parquet output
        batch_size: Batch size for API calls
        resume: Skip already embedded templates
        overwrite: Delete existing output and re-embed
        non_interactive: Fail fast on warnings (for CI/automation)
        force: Override safety checks (use with caution)
    """
    start_time = time.time()
    
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*15 + "SAFE TEMPLATE EMBEDDING v2 (LAR-RAG)" + " "*22 + "║")
    print("╚" + "═"*78 + "╝\n")
    
    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================
    
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Input:           {templates_file}")
    print(f"Output dir:      {output_dir}")
    print(f"Batch size:      {batch_size}")
    print(f"Resume:          {resume}")
    print(f"Overwrite:       {overwrite}")
    print(f"Non-interactive: {non_interactive}")
    print(f"Force:           {force}")
    print(f"Git commit:      {get_git_commit()}")
    print(f"Timestamp:       {datetime.now().isoformat()}")
    
    # ========================================================================
    # 2. LOAD AND VALIDATE TEMPLATES
    # ========================================================================
    
    print(f"\nLoading templates from: {templates_file}")
    templates_path = Path(templates_file)
    
    if not templates_path.exists():
        raise FileNotFoundError(f"Template file not found: {templates_file}")
    
    raw_templates = list(read_json_or_jsonl(templates_file, stream=True))
    
    # Validate (with force override if needed)
    try:
        validated_templates = validate_input_templates(
            raw_templates, 
            non_interactive=non_interactive,
            allow_out_of_range=force  # FIXED: --force now actually overrides range check
        )
    except ValueError as e:
        if force:
            # This should not happen now, but keep as fallback
            print(f"\n⚠️  Unexpected validation error even with --force: {e}")
            raise
        else:
            raise
    
    # ========================================================================
    # 3. HANDLE OVERWRITE/RESUME
    # ========================================================================
    
    output_path = Path(output_dir)
    
    if overwrite and output_path.exists():
        print(f"\n⚠️  Overwrite mode: deleting {output_path}")
        import shutil
        shutil.rmtree(output_path)
        already_embedded = set()
        drift_cases = []
    elif resume:
        already_embedded = load_existing_embeddings(output_path)
        
        # Detect drift
        templates_to_embed, drift_cases = detect_drift(validated_templates, already_embedded)
        
        if drift_cases:
            print(f"\n⚠️  DRIFT DETECTED: {len(drift_cases)} templates have changed content:")
            for tid, old_hash, new_hash in drift_cases[:5]:
                print(f"   {tid}: {old_hash[:8]}... → {new_hash[:8]}...")
            if len(drift_cases) > 5:
                print(f"   ... and {len(drift_cases) - 5} more")
            
            print("\nThese templates will be re-embedded with new content.")
    else:
        already_embedded = set()
        templates_to_embed = validated_templates
        drift_cases = []
    
    if resume and not drift_cases:
        # Filter using already_embedded set
        templates_to_embed = [
            t for t in validated_templates
            if (t.template_id, t.content_hash) not in already_embedded
        ]
    
    print(f"\nTemplates to embed: {len(templates_to_embed):,} "
          f"(skipping {len(already_embedded):,})")
    
    if drift_cases:
        print(f"  Including {len(drift_cases)} drift re-embeds")
    
    if not templates_to_embed:
        print("\n✅ All templates already embedded!")
        return
    
    # ========================================================================
    # 4. PRE-FLIGHT CHECKS
    # ========================================================================
    
    print("\n" + "="*80)
    print("PRE-FLIGHT VALIDATION")
    print("="*80)
    
    # Get Azure config
    config = get_azure_config()
    model = config.openai_embedding_model
    
    print(f"Model:              {model}")
    print(f"Expected dimension: {EMBEDDING_DIMENSION}")
    print(f"Float precision:    {FLOAT_PRECISION}")
    
    # Estimate size
    est_bytes, est_size = estimate_output_size(len(templates_to_embed), EMBEDDING_DIMENSION)
    print(f"\nEstimated output size: {est_size}")
    
    # Token estimate (rough - don't use for cost)
    total_tokens = sum(len(t.normalized_template.split()) for t in templates_to_embed) * 1.3
    print(f"Estimated tokens:      ~{int(total_tokens):,}")
    print(f"(Cost estimate removed - depends on your Azure pricing)")
    
    # Final confirmation
    if not non_interactive:
        print("\n" + "-"*80)
        print("READY TO EMBED")
        print("-"*80)
        response = input("Proceed with embedding? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("\n❌ Aborted by user")
            return
    else:
        print("\n✓ Non-interactive mode: proceeding automatically")
    
    # ========================================================================
    # 5. CREATE CLIENT
    # ========================================================================
    
    print("\nConnecting to Azure OpenAI...")
    client = create_openai_client(config)
    print(f"✓ Connected to {config.openai_endpoint}")
    
    # ========================================================================
    # 6. EMBED IN BATCHES (SHARDED OUTPUT)
    # ========================================================================
    
    failed_batches_path = output_path / "failed_embeddings.jsonl"
    failed_batches_path.parent.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failure_count = 0
    failed_template_ids = []
    
    print("\n" + "="*80)
    print("EMBEDDING TEMPLATES (SHARDED OUTPUT)")
    print("="*80)
    
    writer = ShardedParquetWriter(output_path, EMBEDDING_DIMENSION, model)
    
    try:
        # Process in batches
        for i in range(0, len(templates_to_embed), batch_size):
            batch = templates_to_embed[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(templates_to_embed) - 1) // batch_size + 1
            
            print(f"  Batch {batch_num}/{total_batches} "
                  f"({len(batch)} templates)...", end=" ")
            
            try:
                # Extract normalized templates
                texts = [t.normalized_template for t in batch]
                
                # Embed with improved retry
                embeddings = embed_batch_with_retry(client, texts, model)
                
                # Verify dimension
                if embeddings and len(embeddings[0]) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embeddings[0])}, "
                        f"expected {EMBEDDING_DIMENSION}"
                    )
                
                # Write results to sharded output
                timestamp = datetime.now().isoformat()
                for template, embedding in zip(batch, embeddings):
                    record = EmbeddingRecord(
                        template_id=template.template_id,
                        content_hash=template.content_hash,
                        embedding=np.array(embedding, dtype=FLOAT_PRECISION),
                        model=model,
                        dimension=EMBEDDING_DIMENSION,
                        timestamp=timestamp
                    )
                    writer.write(record)
                
                success_count += len(batch)
                print(f"✓ ({success_count}/{len(templates_to_embed)})")
                
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)
            
            except Exception as e:
                failure_count += len(batch)
                print(f"✗ FAILED: {e}")
                
                # Log failure
                failed_template_ids.extend([t.template_id for t in batch])
                with open(failed_batches_path, 'a', encoding='utf-8') as fail_f:
                    fail_record = {
                        "batch_index": batch_num,
                        "template_ids": [t.template_id for t in batch],
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    fail_f.write(json.dumps(fail_record) + '\n')
    
    finally:
        total_written = writer.close()
    
    runtime = time.time() - start_time
    
    # ========================================================================
    # 7. CREATE MANIFEST
    # ========================================================================
    
    shard_files = list(output_path.glob('part-*.parquet'))
    
    manifest = EmbeddingManifest(
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
        model=model,
        embedding_dimension=EMBEDDING_DIMENSION,
        input_file=str(templates_file),
        output_dir=str(output_dir),
        total_templates=len(templates_to_embed),
        successfully_embedded=success_count,
        failed=failure_count,
        batch_size=batch_size,
        shard_count=len(shard_files),
        config_snapshot={
            'float_precision': str(FLOAT_PRECISION),
            'compression': PARQUET_COMPRESSION,
            'shard_size': PARQUET_SHARD_SIZE,
            'resume': resume,
            'overwrite': overwrite,
            'drift_detected': len(drift_cases)
        },
        runtime_seconds=runtime
    )
    
    manifest_path = output_path / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(manifest), f, indent=2)
    
    # ========================================================================
    # 8. FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("EMBEDDING COMPLETE")
    print("="*80)
    print(f"✅ Success:        {success_count:,} templates")
    if failure_count > 0:
        print(f"❌ Failed:         {failure_count:,} templates")
        print(f"   Failures logged: {failed_batches_path}")
    
    print(f"\nOutput dir:        {output_path}")
    print(f"Shard files:       {len(shard_files)}")
    print(f"Manifest:          {manifest_path}")
    print(f"Runtime:           {runtime:.1f} seconds")
    
    # Calculate actual size
    total_size = sum(f.stat().st_size for f in shard_files)
    total_size_mb = total_size / (1024**2)
    print(f"Total size:        {total_size_mb:.1f} MB")
    
    print("\n" + "="*80)
    print("PROVENANCE TRACKING")
    print("="*80)
    print(f"Git commit:        {manifest.git_commit}")
    print(f"Model:             {manifest.model}")
    print(f"Dimension:         {manifest.embedding_dimension}")
    print(f"Sharded output:    ✓ {len(shard_files)} shard(s)")
    print(f"Content hashes:    ✓ Stored for drift detection")
    if drift_cases:
        print(f"Drift re-embeds:   {len(drift_cases)} templates")
    print("="*80)
    
    if failure_count == 0:
        print("\n🎉 All embeddings completed successfully!")
    else:
        print(f"\n⚠️  {failure_count} templates failed - review {failed_batches_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Safe template embedding v2 with sharded output (INFOCOM 2025)"
    )
    parser.add_argument(
        "--templates",
        default="data/processed/bgl/templates.jsonl",
        help="Input templates JSONL file"
    )
    parser.add_argument(
        "--output",
        default="data/processed/bgl/template_embeddings",
        help="Output directory for sharded Parquet files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for API calls (default: {DEFAULT_BATCH_SIZE})"
    )
    
    # Fixed CLI semantics for resume
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already embedded templates (default)"
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-embed all templates (ignore existing)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output and re-embed all"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail fast on warnings (for CI/automation)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override safety checks (use with caution)"
    )
    
    args = parser.parse_args()
    
    # Handle resume flag
    resume = not args.no_resume
    
    embed_templates_safe(
        args.templates,
        args.output,
        args.batch_size,
        resume,
        args.overwrite,
        args.non_interactive,
        args.force
    )
