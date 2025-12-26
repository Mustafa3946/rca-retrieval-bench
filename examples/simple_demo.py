"""
Simple LAR-RAG Demo

This example demonstrates basic usage of LAR-RAG with synthetic log data.
"""

import time
from datetime import datetime, timedelta
from src.retrieval.base import RetrievalResult
from src.utils.graph_utils import SystemTopologyGraph
from src.utils.temporal_scoring import g_time
from src.utils.graph_utils import graph_distance, g_topo


def create_synthetic_logs():
    """Create synthetic log entries for demo"""
    base_time = datetime(2024, 1, 1, 12, 0, 0).timestamp()
    
    logs = [
        {
            "id": "L1",
            "text": "Database connection pool exhausted",
            "timestamp": base_time - 300,  # 5 min before incident
            "node": "db_server"
        },
        {
            "id": "L2",
            "text": "High memory usage detected",
            "timestamp": base_time - 600,  # 10 min before
            "node": "app_server"
        },
        {
            "id": "L3",
            "text": "Request timeout from upstream",
            "timestamp": base_time - 60,   # 1 min before
            "node": "api_gateway"
        },
        {
            "id": "L4",
            "text": "Normal system operation",
            "timestamp": base_time + 300,  # 5 min after (should be penalized)
            "node": "web_server"
        },
        {
            "id": "L5",
            "text": "Connection refused error",
            "timestamp": base_time - 120,  # 2 min before
            "node": "app_server"
        },
    ]
    
    return logs, base_time


def create_topology():
    """Create system topology graph"""
    graph = SystemTopologyGraph()
    
    # Build a simple microservices topology
    graph.add_edge("web_server", "api_gateway")
    graph.add_edge("api_gateway", "app_server")
    graph.add_edge("app_server", "db_server")
    graph.add_edge("app_server", "cache_server")
    
    return graph


def simple_rerank(query, logs, incident_time, incident_node, topology_graph):
    """
    Simplified LAR-RAG reranking (without embeddings for demo)
    
    In production, you would:
    1. Use actual embeddings for semantic similarity
    2. Implement proper Stage 1 retrieval (BM25/dense)
    """
    
    # Parameters
    lambda_pre = 0.001   # Temporal decay before incident (seconds^-1)
    lambda_post = 0.01   # Temporal decay after incident (higher penalty)
    lambda_g = 0.5       # Topology decay
    
    alpha = 0.4  # Semantic weight
    beta = 0.3   # Temporal weight
    gamma = 0.3  # Topology weight
    
    results = []
    
    for log in logs:
        # 1. Semantic score (simplified: keyword matching for demo)
        query_words = set(query.lower().split())
        log_words = set(log["text"].lower().split())
        s_sem = len(query_words & log_words) / max(len(query_words), 1)
        
        # 2. Temporal score
        delta_t = incident_time - log["timestamp"]
        g_t = g_time(delta_t, lambda_pre, lambda_post)
        
        # 3. Topology score
        d_g = graph_distance(topology_graph.graph, incident_node, log["node"])
        g_g = g_topo(d_g, lambda_g)
        
        # 4. Combined score
        score = alpha * s_sem + beta * g_t + gamma * g_g
        
        results.append({
            "log": log,
            "score": score,
            "components": {
                "semantic": s_sem,
                "temporal": g_t,
                "topology": g_g
            }
        })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results


def main():
    print("=" * 60)
    print("LAR-RAG Simple Demo")
    print("=" * 60)
    
    # Setup
    logs, incident_time = create_synthetic_logs()
    topology = create_topology()
    
    # Incident details
    query = "connection timeout error"
    incident_node = "app_server"
    incident_dt = datetime.fromtimestamp(incident_time)
    
    print(f"\nIncident Details:")
    print(f"  Query: {query}")
    print(f"  Time: {incident_dt}")
    print(f"  Node: {incident_node}")
    
    print(f"\n{'='*60}")
    print("Available Logs:")
    print(f"{'='*60}")
    for log in logs:
        log_dt = datetime.fromtimestamp(log["timestamp"])
        delta = (log["timestamp"] - incident_time) / 60  # minutes
        print(f"\n{log['id']}: {log['text']}")
        print(f"  Time: {log_dt} ({delta:+.1f} min)")
        print(f"  Node: {log['node']}")
    
    # Perform LAR-RAG reranking
    results = simple_rerank(query, logs, incident_time, incident_node, topology)
    
    print(f"\n{'='*60}")
    print("LAR-RAG Ranked Results:")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        log = result["log"]
        comp = result["components"]
        
        print(f"\n{i}. {log['id']}: {log['text']}")
        print(f"   Overall Score: {result['score']:.3f}")
        print(f"   └─ Semantic:  {comp['semantic']:.3f}")
        print(f"   └─ Temporal:  {comp['temporal']:.3f}")
        print(f"   └─ Topology:  {comp['topology']:.3f}")
        print(f"   Node: {log['node']}")
    
    print("\n" + "="*60)
    print("Key Observations:")
    print("="*60)
    print("- Logs closer in time to incident score higher (temporal)")
    print("- Logs from topologically close nodes score higher")
    print("- Logs after incident are heavily penalized")
    print("- Semantic relevance is combined with RCA priors")


if __name__ == "__main__":
    main()
