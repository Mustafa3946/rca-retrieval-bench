"""
BM25-only retriever for LAR-RAG baselines.

Uses Azure AI Search's built-in BM25 ranking algorithm for lexical search.
Now operates at TEMPLATE-LEVEL first (INFOCOM requirement), then expands to logs.
"""
import os
import ssl
import json
from collections import defaultdict
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv

load_dotenv()

# Disable SSL verification for corporate proxy
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"


class BM25Retriever:
    """
    BM25-only retriever using Azure AI Search.
    
    TEMPLATE-FIRST ARCHITECTURE (INFOCOM requirement):
    1. Query "bgl-templates" index with BM25 on template text
    2. Retrieve top-K templates
    3. Expand templates to logs using template_map.jsonl
    4. Return log-level results with BM25 scores inherited from templates
    """
    
    def __init__(self, template_map_file="data/processed/bgl/template_map.jsonl", k1=1.2, b=0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            template_map_file: Path to template-to-log mapping file (JSONL)
            k1: BM25 term saturation parameter (default: 1.2)
            b: BM25 length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Load template-to-log mapping
        print(f"Loading template map from {template_map_file}...")
        self.template_to_logs = defaultdict(list)
        with open(template_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    mapping = json.loads(line)
                    self.template_to_logs[mapping['template_id']].append({
                        'log_id': mapping['log_id'],
                        'timestamp': mapping['timestamp'],
                        'node': mapping['node']
                    })
        print(f"Loaded mappings for {len(self.template_to_logs)} templates")
        
        # Connect to template index (NOT log-embeddings)
        transport = RequestsTransport(connection_verify=False)
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="bgl-templates",  # ✅ TEMPLATE-FIRST
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
            transport=transport
        )
    
    def retrieve(self, query_text, top_k=10, top_k_templates=50, max_logs_per_template=100, query_timestamp=None, query_node=None):
        """
        Retrieve top-k logs using BM25 lexical search on TEMPLATES.
        
        TEMPLATE-FIRST ARCHITECTURE:
        1. BM25 search on template text field → top_k_templates
        2. Expand templates to logs using template_map.jsonl
        3. Limit logs per template to avoid explosion
        4. Return log-level results with inherited BM25 scores
        
        Args:
            query_text: Query string
            top_k: Number of final log results to return
            top_k_templates: Number of templates to retrieve in Stage 1 (default: 50)
            max_logs_per_template: Max logs to expand per template (default: 100)
            query_timestamp: (Ignored) Query timestamp
            query_node: (Ignored) Query node identifier
        
        Returns:
            List of scored log results with BM25 scores from templates
        """
        # Stage 1: BM25 search on TEMPLATES
        print(f"[BM25] Stage 1: Retrieving top {top_k_templates} templates...")
        template_results = self.search_client.search(
            search_text=query_text,
            top=top_k_templates,
            select=["template_id", "template", "count"]
        )
        
        # Stage 2: Expand templates to logs
        print(f"[BM25] Stage 2: Expanding templates to logs...")
        candidate_logs = []
        
        for template_result in template_results:
            template_id = template_result["template_id"]
            template_score = template_result["@search.score"]
            
            # Get logs for this template
            if template_id in self.template_to_logs:
                logs = self.template_to_logs[template_id]
                
                # Limit logs per template
                for log in logs[:max_logs_per_template]:
                    candidate_logs.append({
                        "log_id": log["log_id"],
                        "timestamp": log["timestamp"],
                        "node": log["node"],
                        "template_id": template_id,
                        "score": template_score,  # Inherit BM25 score from template
                        "semantic_score": 0.0,  # Not applicable
                        "temporal_score": 0.0,  # Not used in baseline
                        "topology_score": 0.0,  # Not used in baseline
                        "combined_score": template_score
                    })
        
        print(f"[BM25] Expanded to {len(candidate_logs)} candidate logs")
        
        # Sort by template BM25 score and return top-k
        candidate_logs.sort(key=lambda x: x["score"], reverse=True)
        
        return candidate_logs[:top_k]


if __name__ == "__main__":
    # Test BM25 retriever (template-first)
    query = "block corruption detected in datanode"
    
    print("=== BM25-Only Retriever (Template-First) ===")
    retriever = BM25Retriever(template_map_file="data/processed/bgl/template_map.jsonl")
    print(f"Query: {query}\n")
    
    results = retriever.retrieve(query, top_k=10, top_k_templates=20)
    
    print(f"\nFound {len(results)} log results (from {results[0]['template_id'] if results else 'N/A'}):\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [BM25 Score: {r['score']:.3f}] {r['log_id']}")
        print(f"   Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node {r['node']}\n")
        print(f"   {r['message'][:80]}...")
        print()
    
    # Test with different queries
    print("\n=== Testing Different Query Types ===")
    
    test_queries = [
        "exception error failure",
        "namenode datanode communication",
        "block replication hdfs",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        if results:
            print(f"Top result: {results[0]['message'][:60]}... (score: {results[0]['score']:.3f})")
        else:
            print("No results found")
