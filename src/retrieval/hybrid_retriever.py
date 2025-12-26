"""
Hybrid retriever combining BM25 (lexical) and dense (semantic) search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both approaches.
Now operates at TEMPLATE-LEVEL first (INFOCOM requirement), then expands to logs.
"""
import os
import ssl
import json
from collections import defaultdict
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from dotenv import load_dotenv

load_dotenv()

# Disable SSL verification for corporate proxy
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["AZURE_CORE_DISABLE_VERIFY_SSL"] = "1"


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense vector search.
    
    TEMPLATE-FIRST ARCHITECTURE (INFOCOM requirement):
    1. BM25 search on template text
    2. Dense vector search on template embeddings
    3. RRF fusion at TEMPLATE level
    4. Expand top-K templates to logs using template_map.jsonl
    5. Return log-level results
    
    Fusion strategies:
    - 'rrf': Reciprocal Rank Fusion (default)
    - 'weighted': Weighted linear combination of normalized scores
    """
    
    def __init__(self, template_map_file="data/processed/bgl/template_map.jsonl", fusion_strategy='rrf', rrf_k=60, bm25_weight=0.5, dense_weight=0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            template_map_file: Path to template-to-log mapping file (JSONL)
            fusion_strategy: 'rrf' or 'weighted' (default: 'rrf')
            rrf_k: RRF constant (default: 60, standard in literature)
            bm25_weight: Weight for BM25 in weighted fusion (default: 0.5)
            dense_weight: Weight for dense in weighted fusion (default: 0.5)
        """
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
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
        
        # Lazy imports for Azure-only dependencies
        from openai import AzureOpenAI
        import httpx
        
        # Initialize OpenAI client for embeddings
        http_client = httpx.Client(verify=False)
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            http_client=http_client
        )
        
        # Initialize search client (TEMPLATE index, not log-embeddings)
        transport = RequestsTransport(connection_verify=False)
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="bgl-templates",  # ✅ TEMPLATE-FIRST
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
            transport=transport
        )
    
    def _reciprocal_rank_fusion(self, bm25_templates, dense_templates, top_k):
        """
        Merge TEMPLATE results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1 / (k + rank(d))
        where rank(d) is the rank of template d in each result list.
        
        Args:
            bm25_templates: List of template results from BM25
            dense_templates: List of template results from dense search
            top_k: Number of templates to return
            
        Returns:
            Merged and sorted template results
        """
        # Create rank maps (at TEMPLATE level)
        bm25_ranks = {r["template_id"]: i + 1 for i, r in enumerate(bm25_templates)}
        dense_ranks = {r["template_id"]: i + 1 for i, r in enumerate(dense_templates)}
        
        # Collect all unique template IDs
        all_template_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for template_id in all_template_ids:
            score = 0.0
            if template_id in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[template_id])
            if template_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[template_id])
            rrf_scores[template_id] = score
        
        # Sort by RRF score
        sorted_template_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create template result map
        template_map = {}
        for r in bm25_templates + dense_templates:
            if r["template_id"] not in template_map:
                template_map[r["template_id"]] = r
        
        # Return top-k templates with RRF scores
        merged = []
        for template_id in sorted_template_ids[:top_k]:
            template = template_map[template_id].copy()
            template["combined_score"] = rrf_scores[template_id]
            template["fusion_method"] = "rrf"
            merged.append(template)
        
        return merged
    
    def _weighted_fusion(self, bm25_templates, dense_templates, top_k):
        """
        Merge TEMPLATE results using weighted score combination.
        
        Scores are min-max normalized before combining.
        
        Args:
            bm25_templates: List of template results from BM25
            dense_templates: List of template results from dense search
            top_k: Number of templates to return
            
        Returns:
            Merged and sorted template results
        """
        # Normalize BM25 scores
        if bm25_templates:
            bm25_scores = [r["score"] for r in bm25_templates]
            min_bm25 = min(bm25_scores)
            max_bm25 = max(bm25_scores)
            range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            for r in bm25_templates:
                r["normalized_score"] = (r["score"] - min_bm25) / range_bm25
        
        # Normalize dense scores
        if dense_templates:
            dense_scores = [r["semantic_score"] for r in dense_templates]
            min_dense = min(dense_scores)
            max_dense = max(dense_scores)
            range_dense = max_dense - min_dense if max_dense > min_dense else 1.0
            for r in dense_templates:
                r["normalized_semantic_score"] = (r["semantic_score"] - min_dense) / range_dense
        
        # Merge scores at TEMPLATE level
        template_map = {}
        for r in bm25_templates:
            template_id = r["template_id"]
            template_map[template_id] = r.copy()
            template_map[template_id]["combined_score"] = self.bm25_weight * r["normalized_score"]
        
        for r in dense_templates:
            template_id = r["template_id"]
            if template_id in template_map:
                template_map[template_id]["combined_score"] += self.dense_weight * r["normalized_semantic_score"]
                template_map[template_id]["semantic_score"] = r["semantic_score"]
            else:
                template_map[template_id] = r.copy()
                template_map[template_id]["combined_score"] = self.dense_weight * r["normalized_semantic_score"]
                template_map[template_id]["score"] = 0.0
        
        # Sort and return top-k templates
        merged = sorted(template_map.values(), key=lambda x: x["combined_score"], reverse=True)
        for r in merged:
            r["fusion_method"] = "weighted"
        
        return merged[:top_k]
    
    def retrieve(self, query_text, top_k=10, top_k_templates=50, max_logs_per_template=100, query_timestamp=None, query_node=None):
        """
        Retrieve top-k logs using hybrid search on TEMPLATES.
        
        TEMPLATE-FIRST ARCHITECTURE:
        1. BM25 search on template text
        2. Dense vector search on template embeddings  
        3. RRF/Weighted fusion at TEMPLATE level
        4. Expand top-K templates to logs
        5. Return log-level results
        
        Args:
            query_text: Query string
            top_k: Number of final log results to return
            top_k_templates: Number of templates to retrieve before fusion (default: 50)
            max_logs_per_template: Max logs to expand per template (default: 100)
            query_timestamp: (Ignored) Query timestamp
            query_node: (Ignored) Query node identifier
        
        Returns:
            List of scored log results with hybrid scores from templates
        """
        print(f"[Hybrid] Stage 1: Retrieving {top_k_templates} templates with BM25 + Dense...")
        
        # Retrieve more templates for fusion
        k_retrieve = max(top_k_templates, top_k * 2)
        
        # 1. BM25 search on TEMPLATES
        bm25_results = list(self.search_client.search(
            search_text=query_text,
            top=k_retrieve,
            select=["template_id", "template", "count"]
        ))
        
        bm25_formatted = []
        for r in bm25_results:
            bm25_formatted.append({
                "template_id": r["template_id"],
                "template": r["template"],
                "count": r.get("count", 0),
                "score": r["@search.score"],
                "semantic_score": 0.0,
            })
        
        # 2. Dense vector search on TEMPLATE embeddings
        response = self.openai_client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=k_retrieve,
            fields="embedding"
        )
        
        dense_results = list(self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["template_id", "template", "count"]
        ))
        
        dense_formatted = []
        for r in dense_results:
            dense_formatted.append({
                "template_id": r["template_id"],
                "template": r["template"],
                "count": r.get("count", 0),
                "score": 0.0,
                "semantic_score": r["@search.score"],
            })
        
        # 3. Fusion at TEMPLATE level
        print(f"[Hybrid] Fusing templates with {self.fusion_strategy}...")
        if self.fusion_strategy == 'rrf':
            merged_templates = self._reciprocal_rank_fusion(bm25_formatted, dense_formatted, top_k_templates)
        elif self.fusion_strategy == 'weighted':
            merged_templates = self._weighted_fusion(bm25_formatted, dense_formatted, top_k_templates)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # 4. Expand templates to logs
        print(f"[Hybrid] Stage 2: Expanding {len(merged_templates)} templates to logs...")
        candidate_logs = []
        
        for template_result in merged_templates:
            template_id = template_result["template_id"]
            template_score = template_result["combined_score"]
            
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
                        "score": template_result.get("score", 0.0),
                        "semantic_score": template_result.get("semantic_score", 0.0),
                        "temporal_score": 0.0,
                        "topology_score": 0.0,
                        "combined_score": template_score,
                        "fusion_method": template_result.get("fusion_method", "unknown")
                    })
        
        print(f"[Hybrid] Expanded to {len(candidate_logs)} candidate logs")
        
        # Sort by hybrid score and return top-k
        candidate_logs.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return candidate_logs[:top_k]


if __name__ == "__main__":
    # Test hybrid retriever (template-first)
    query = "block corruption detected in datanode"
    
    print("=== Hybrid Retriever (Template-First, RRF) ===")
    retriever_rrf = HybridRetriever(
        template_map_file="data/processed/bgl/template_map.jsonl",
        fusion_strategy='rrf'
    )
    print(f"Query: {query}\n")
    
    results = retriever_rrf.retrieve(query, top_k=5, top_k_templates=20)
    
    print(f"\nFound {len(results)} log results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Hybrid Score: {r['combined_score']:.4f}, BM25: {r['score']:.3f}, Dense: {r['semantic_score']:.3f}]")
        print(f"   Log ID: {r['log_id']} | Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node {r['node']}")
        print()
    
    print("\n=== Hybrid Retriever (Template-First, Weighted) ===")
    retriever_weighted = HybridRetriever(
        template_map_file="data/processed/bgl/template_map.jsonl",
        fusion_strategy='weighted',
        bm25_weight=0.3,
        dense_weight=0.7
    )
    print(f"Query: {query}\n")
    
    results = retriever_weighted.retrieve(query, top_k=5, top_k_templates=20)
    
    print(f"\nFound {len(results)} log results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Hybrid Score: {r['combined_score']:.4f}, BM25: {r['score']:.3f}, Dense: {r['semantic_score']:.3f}]")
        print(f"   Log ID: {r['log_id']} | Template: {r['template_id']}")
        print(f"   {r['timestamp']} | Node {r['node']}")
        print()
