"""
Embed BGL templates using Azure OpenAI text-embedding-3-small

Embeds unique templates instead of all logs to reduce cost.
Supports resume, retry with backoff, and failure logging.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import time
import random
import ssl
from openai import AzureOpenAI
from dotenv import load_dotenv
from utils.io_utils import read_json_or_jsonl, load_embedded_ids

load_dotenv()


def create_embedding_client(insecure_ssl: bool = False) -> AzureOpenAI:
    """
    Create Azure OpenAI client.
    
    Args:
        insecure_ssl: If True, disable SSL verification (use only if necessary)
        
    Returns:
        AzureOpenAI client
    """
    if insecure_ssl:
        print("⚠️  WARNING: SSL verification disabled")
        import httpx
        http_client = httpx.Client(verify=False)
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["CURL_CA_BUNDLE"] = ""
    else:
        http_client = None
    
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        http_client=http_client
    )


def embed_batch_with_retry(
    client: AzureOpenAI,
    texts: list,
    model: str,
    max_retries: int = 6
) -> list:
    """
    Embed a batch of texts with exponential backoff retry.
    
    Args:
        client: Azure OpenAI client
        texts: List of text strings to embed
        model: Model name (text-embedding-3-small)
        max_retries: Maximum retry attempts
        
    Returns:
        List of embeddings
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model
            )
            return [item.embedding for item in response.data]
        
        except Exception as e:
            error_str = str(e)
            
            # Retry on rate limit or server errors
            if "429" in error_str or "5" in error_str[:3]:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
            
            # Non-retryable error or max retries reached
            raise


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


def embed_templates(
    templates_file: str,
    output_file: str,
    batch_size: int = 256,
    resume: bool = True,
    overwrite: bool = False,
    insecure_ssl: bool = False,
    price_per_1m_tokens: float = 0.02
):
    """
    Embed BGL templates with resume support.
    
    Args:
        templates_file: Path to templates.jsonl
        output_file: Path to save template_embeddings.jsonl
        batch_size: Number of templates per batch
        resume: Skip already embedded templates
        overwrite: Delete existing output and re-embed all
        insecure_ssl: Disable SSL verification (use only if necessary)
        price_per_1m_tokens: Price per 1M tokens for cost estimation
    """
    print("="*80)
    print("BGL Template Embedding")
    print("="*80)
    print(f"Input: {templates_file}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size}")
    print(f"Resume: {resume}")
    print(f"Overwrite: {overwrite}")
    print()
    
    # Handle overwrite
    output_path = Path(output_file)
    if overwrite and output_path.exists():
        print("Overwrite mode: deleting existing output...")
        output_path.unlink()
    
    # Load templates
    print("Loading templates...")
    templates = list(read_json_or_jsonl(templates_file, stream=True))
    print(f"Loaded {len(templates)} templates")
    
    # Load already embedded IDs
    already_embedded = set()
    if resume and not overwrite:
        already_embedded = load_embedded_ids(output_file)
        if already_embedded:
            print(f"Found {len(already_embedded)} already embedded templates")
    
    # Filter templates to embed
    templates_to_embed = [t for t in templates if t['template_id'] not in already_embedded]
    print(f"\nTemplates to embed: {len(templates_to_embed)} (skipping {len(already_embedded)})")
    
    if not templates_to_embed:
        print("\n✅ All templates already embedded!")
        return
    
    # Estimate cost
    total_tokens = sum(estimate_tokens(t['template']) for t in templates_to_embed)
    estimated_cost = (total_tokens / 1_000_000) * price_per_1m_tokens
    print(f"\nCost estimate:")
    print(f"  Total tokens: ~{total_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print()
    
    # Create client
    print("Connecting to Azure OpenAI...")
    client = create_embedding_client(insecure_ssl=insecure_ssl)
    model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    print(f"Using model: {model}")
    print()
    
    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_batches_path = Path("results/failed_batches.jsonl")
    failed_batches_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file in append mode
    mode = 'a' if resume and not overwrite else 'w'
    
    success_count = 0
    failure_count = 0
    
    print("Embedding templates...")
    with open(output_file, mode, encoding='utf-8') as out_f:
        # Process in batches
        for i in range(0, len(templates_to_embed), batch_size):
            batch = templates_to_embed[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(templates_to_embed) - 1) // batch_size + 1
            
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} templates)...", end=" ")
            
            try:
                # Extract texts
                texts = [t['template'] for t in batch]
                
                # Embed with retry
                embeddings = embed_batch_with_retry(client, texts, model)
                
                # Write results
                for template, embedding in zip(batch, embeddings):
                    result = {
                        "template_id": template['template_id'],
                        "template": template['template'],
                        "embedding": embedding,
                        "count": template.get('count', 1)
                    }
                    out_f.write(json.dumps(result) + '\n')
                
                success_count += len(batch)
                print(f"✓ ({success_count}/{len(templates_to_embed)})")
                
                # Rate limiting
                time.sleep(0.1)
            
            except Exception as e:
                failure_count += len(batch)
                print(f"✗ FAILED: {e}")
                
                # Log failure
                with open(failed_batches_path, 'a', encoding='utf-8') as fail_f:
                    fail_record = {
                        "batch_index": batch_num,
                        "template_ids": [t['template_id'] for t in batch],
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    fail_f.write(json.dumps(fail_record) + '\n')
    
    # Final summary
    print("\n" + "="*80)
    print("Embedding Complete!")
    print("="*80)
    print(f"✅ Success: {success_count} templates")
    if failure_count > 0:
        print(f"❌ Failed: {failure_count} templates")
        print(f"   See failures in: {failed_batches_path}")
    print(f"\nOutput: {output_file}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed BGL templates")
    parser.add_argument("--templates", default="data/processed/bgl/templates.jsonl",
                       help="Input templates JSONL file")
    parser.add_argument("--output", default="data/processed/bgl/template_embeddings.jsonl",
                       help="Output embeddings JSONL file")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for embedding")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Skip already embedded templates (default: True)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Re-embed all templates (overrides --resume)")
    parser.add_argument("--insecure-ssl", action="store_true",
                       help="Disable SSL verification (use only if required)")
    parser.add_argument("--price-per-1m-tokens", type=float, default=0.02,
                       help="Price per 1M tokens for cost estimation")
    
    args = parser.parse_args()
    
    embed_templates(
        args.templates,
        args.output,
        args.batch_size,
        args.resume,
        args.overwrite,
        args.insecure_ssl,
        args.price_per_1m_tokens
    )
