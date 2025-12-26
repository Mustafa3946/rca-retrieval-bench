"""
System Topology Graph Utilities for LAR-RAG.

Provides real graph-based topology distance computation with:
- Proper node ID normalization
- Multiple graph construction strategies
- Efficient distance caching
- Coverage reporting
"""

import json
import math
import networkx as nx
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List
from functools import lru_cache
from .node_normalization import normalize_node_id


class SystemTopologyGraph:
    """
    Manages system topology for computing graph distances between nodes.
    
    Supports three construction modes:
    1. Static: Load from JSON file
    2. Log-inferred: Build from log communication patterns
    3. Synthetic: Generate consistent graph for datasets without topology
    """
    
    def __init__(self, mode: str = "static", graph_file: Optional[str] = None, node_ids: Optional[list] = None):
        """
        Initialize topology graph.
        
        Args:
            mode: Construction mode - "static", "log-inferred", or "synthetic"
            graph_file: Path to JSON graph file (required for static mode)
            node_ids: List of node IDs for synthetic mode (optional)
        """
        self.mode = mode
        self.graph = nx.Graph()
        self._node_cache: Dict[Tuple[str, str], float] = {}
        
        if mode == "static":
            if not graph_file:
                raise ValueError("graph_file required for static mode")
            self.load_from_file(graph_file)
        elif mode == "synthetic":
            self.build_synthetic_hdfs_graph(node_ids=node_ids)
        elif mode == "log-inferred":
            # Will be populated by add_edge() calls during log processing
            pass
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'static', 'log-inferred', or 'synthetic'")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load graph from JSON file.
        
        Expected format:
        {
            "nodes": ["node1", "node2", ...],
            "edges": [
                ["node1", "node2"],
                ["node2", "node3"],
                ...
            ]
        }
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Add nodes
        for node in data.get("nodes", []):
            self.graph.add_node(node)
        
        # Add edges
        for edge in data.get("edges", []):
            if len(edge) == 2:
                self.graph.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                # Support weighted edges: ["node1", "node2", weight]
                self.graph.add_edge(edge[0], edge[1], weight=edge[2])
        
        print(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        self._node_cache.clear()
    
    def build_synthetic_hdfs_graph(self, node_ids: Optional[list] = None) -> None:
        """
        Build synthetic but realistic HDFS topology graph with normalized node IDs.
        
        HDFS structure:
        - DataNodes (storage, organized in racks)
        
        This creates a rack-based cluster topology suitable for HDFS logs.
        
        Args:
            node_ids: List of datanode IDs to include (will be normalized)
        """
        # Use provided node IDs or detect from dataset
        if node_ids is None:
            # Default: assume node IDs 0-255 (typical for HDFS last-octet addressing)
            node_ids = [str(i) for i in range(256)]
        
        # Normalize all node IDs
        normalized_ids = [normalize_node_id(str(nid)) for nid in node_ids]
        normalized_ids = sorted(set(n for n in normalized_ids if n))
        
        if len(normalized_ids) == 0:
            print("Warning: No valid node IDs provided for synthetic graph")
            return
        
        # Organize datanodes into racks (create ~10-20 nodes per rack)
        nodes_per_rack = 20
        num_racks = max(1, (len(normalized_ids) + nodes_per_rack - 1) // nodes_per_rack)
        
        for rack_id in range(num_racks):
            # Assign nodes to this rack
            start_idx = rack_id * nodes_per_rack
            end_idx = min(start_idx + nodes_per_rack, len(normalized_ids))
            rack_nodes = normalized_ids[start_idx:end_idx]
            
            # Add nodes
            for node_id in rack_nodes:
                self.graph.add_node(node_id)
            
            # Fully connect within rack (low distance)
            for i in range(len(rack_nodes)):
                for j in range(i + 1, len(rack_nodes)):
                    self.graph.add_edge(rack_nodes[i], rack_nodes[j])
            
            # Connect to next rack (higher distance)
            if rack_id < num_racks - 1:
                next_rack_start = end_idx
                if next_rack_start < len(normalized_ids):
                    # Connect one representative from each rack
                    self.graph.add_edge(rack_nodes[0], normalized_ids[next_rack_start])
        
        print(f"Built synthetic HDFS graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges, {num_racks} racks")
        self._node_cache.clear()
    
    def infer_from_logs(self, logs: list) -> None:
        """
        Infer graph topology from log communication patterns.
        
        Args:
            logs: List of log entries with 'node' field
        """
        # Extract unique nodes
        nodes = set()
        for log in logs:
            if 'node' in log:
                nodes.add(log['node'])
        
        # Add all nodes
        for node in nodes:
            self.graph.add_node(node)
        
        # Simple heuristic: connect nodes that appear in temporal proximity
        # This is a placeholder - real implementation would parse communication logs
        sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))
        
        window_size = 10
        for i in range(len(sorted_logs) - window_size):
            window = sorted_logs[i:i+window_size]
            window_nodes = [log['node'] for log in window if 'node' in log]
            
            # Connect nodes that appear together
            unique_nodes = list(set(window_nodes))
            for j in range(len(unique_nodes)):
                for k in range(j+1, len(unique_nodes)):
                    if not self.graph.has_edge(unique_nodes[j], unique_nodes[k]):
                        self.graph.add_edge(unique_nodes[j], unique_nodes[k])
        
        print(f"Inferred graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        self._node_cache.clear()
    
    def add_edge(self, node_a: str, node_b: str, weight: float = 1.0) -> None:
        """Add edge to graph (useful for log-inferred mode)."""
        if not self.graph.has_node(node_a):
            self.graph.add_node(node_a)
        if not self.graph.has_node(node_b):
            self.graph.add_node(node_b)
        self.graph.add_edge(node_a, node_b, weight=weight)
        self._node_cache.clear()
    
    def compute_graph_distance(self, node_a: str, node_b: str) -> float:
        """
        Compute shortest-path hop count between two nodes.
        
        Args:
            node_a: Source node identifier
            node_b: Target node identifier
            
        Returns:
            Shortest path length (hop count). Returns 999.0 if no path exists.
        """
        # Check cache first
        cache_key = tuple(sorted([node_a, node_b]))
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]
        
        # Same node
        if node_a == node_b:
            self._node_cache[cache_key] = 0.0
            return 0.0
        
        # Check if nodes exist in graph
        if node_a not in self.graph or node_b not in self.graph:
            # Return large constant for unknown nodes
            self._node_cache[cache_key] = 999.0
            return 999.0
        
        try:
            # Compute shortest path length
            distance = nx.shortest_path_length(self.graph, source=node_a, target=node_b)
            self._node_cache[cache_key] = float(distance)
            return float(distance)
        except nx.NetworkXNoPath:
            # No path exists
            self._node_cache[cache_key] = 999.0
            return 999.0
    
    def topology_proximity_score(self, node_a: str, node_b: str, lambda_g: float = 0.5) -> float:
        """
        Compute topology proximity score using exponential decay.
        
        Formula: g_topo(d) = exp(-lambda_g * d)
        
        Args:
            node_a: Source node identifier
            node_b: Target node identifier
            lambda_g: Decay rate parameter (higher = faster decay)
            
        Returns:
            Proximity score in [0, 1], where 1 = same node, 0 = very far
        """
        distance = self.compute_graph_distance(node_a, node_b)
        
        # Exponential decay
        score = math.exp(-lambda_g * distance)
        return score
    
    def get_neighbors(self, node: str) -> Set[str]:
        """Get all neighbors of a node."""
        if node not in self.graph:
            return set()
        return set(self.graph.neighbors(node))
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save graph to JSON file.
        
        Args:
            filepath: Output file path
        """
        data = {
            "nodes": list(self.graph.nodes()),
            "edges": [[u, v] for u, v in self.graph.edges()]
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved graph to {filepath}")
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_connected": nx.is_connected(self.graph),
            "num_components": nx.number_connected_components(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }


# Convenience function for direct usage
def create_topology_graph(mode: str = "synthetic", graph_file: Optional[str] = None) -> SystemTopologyGraph:
    """
    Factory function to create topology graph.
    
    Args:
        mode: "static", "log-inferred", or "synthetic"
        graph_file: Path to JSON file (for static mode)
        
    Returns:
        SystemTopologyGraph instance
    """
    return SystemTopologyGraph(mode=mode, graph_file=graph_file)


# Module-level API functions (as required by publication-readiness prompt)

def load_graph_from_json(path: str) -> nx.Graph:
    """
    Load graph from JSON file.
    
    Args:
        path: Path to JSON graph file
        
    Returns:
        NetworkX Graph object
    """
    topo = SystemTopologyGraph(mode="static", graph_file=path)
    return topo.graph


def build_synthetic_graph(nodes: List[str], mode: str = "ring") -> nx.Graph:
    """
    Build deterministic synthetic graph from node list.
    
    Args:
        nodes: List of node IDs (will be normalized)
        mode: Graph topology type - "ring", "small-world", "clustered"
        
    Returns:
        NetworkX Graph with specified topology
    """
    # Normalize all node IDs
    normalized_nodes = [normalize_node_id(n) for n in nodes if normalize_node_id(n)]
    normalized_nodes = sorted(set(normalized_nodes))  # Unique, sorted for determinism
    
    G = nx.Graph()
    G.add_nodes_from(normalized_nodes)
    
    if len(normalized_nodes) < 2:
        return G
    
    if mode == "ring":
        # Connect each node to next in sorted order, plus wrap-around
        for i in range(len(normalized_nodes)):
            G.add_edge(normalized_nodes[i], normalized_nodes[(i + 1) % len(normalized_nodes)])
    
    elif mode == "small-world":
        # Watts-Strogatz small-world graph (k=4 neighbors, p=0.1 rewiring)
        k = min(4, len(normalized_nodes) - 1)
        G = nx.watts_strogatz_graph(len(normalized_nodes), k, 0.1, seed=42)
        # Relabel with actual node IDs
        mapping = {i: normalized_nodes[i] for i in range(len(normalized_nodes))}
        G = nx.relabel_nodes(G, mapping)
    
    elif mode == "clustered":
        # Divide nodes into clusters (racks) with intra-cluster density
        num_clusters = max(3, int(len(normalized_nodes) ** 0.5))
        cluster_size = len(normalized_nodes) // num_clusters
        
        for cluster_id in range(num_clusters):
            start = cluster_id * cluster_size
            end = start + cluster_size if cluster_id < num_clusters - 1 else len(normalized_nodes)
            cluster_nodes = normalized_nodes[start:end]
            
            # Fully connect within cluster
            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    G.add_edge(cluster_nodes[i], cluster_nodes[j])
            
            # Sparse inter-cluster connections
            if cluster_id < num_clusters - 1:
                next_cluster_start = end
                if next_cluster_start < len(normalized_nodes):
                    # Connect one node from this cluster to next
                    G.add_edge(cluster_nodes[0], normalized_nodes[next_cluster_start])
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'ring', 'small-world', or 'clustered'")
    
    return G


def infer_graph_from_logs(logs: List[dict], method: str = "cooccurrence", 
                         window_seconds: int = 60, min_cooccurrence: int = 2) -> nx.Graph:
    """
    Infer topology graph from log communication patterns.
    
    Args:
        logs: List of log dicts with 'node' and 'timestamp' fields
        method: Inference method - "cooccurrence" (sliding window)
        window_seconds: Time window for co-occurrence (seconds)
        min_cooccurrence: Minimum co-occurrences to create edge
        
    Returns:
        Inferred NetworkX Graph
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    
    G = nx.Graph()
    
    if method == "cooccurrence":
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', ''))
        
        # Track co-occurrences
        cooccurrence_counts = defaultdict(int)
        
        # Sliding window approach
        for i in range(len(sorted_logs)):
            log_i = sorted_logs[i]
            node_i = normalize_node_id(log_i.get('node', ''))
            if not node_i:
                continue
            
            G.add_node(node_i)
            time_i = datetime.fromisoformat(log_i.get('timestamp', ''))
            
            # Look ahead in window
            for j in range(i + 1, len(sorted_logs)):
                log_j = sorted_logs[j]
                time_j = datetime.fromisoformat(log_j.get('timestamp', ''))
                
                # Stop if outside window
                if (time_j - time_i).total_seconds() > window_seconds:
                    break
                
                node_j = normalize_node_id(log_j.get('node', ''))
                if not node_j or node_i == node_j:
                    continue
                
                # Count co-occurrence
                edge_key = tuple(sorted([node_i, node_j]))
                cooccurrence_counts[edge_key] += 1
        
        # Add edges with sufficient co-occurrence
        for (node_a, node_b), count in cooccurrence_counts.items():
            if count >= min_cooccurrence:
                G.add_edge(node_a, node_b, weight=count)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return G


@lru_cache(maxsize=10000)
def graph_distance(G: nx.Graph, a: str, b: str, disconnected_value: int = 999) -> int:
    """
    Compute shortest path distance between two nodes.
    
    Args:
        G: NetworkX Graph
        a: Source node ID (will be normalized)
        b: Target node ID (will be normalized)
        disconnected_value: Value to return if no path exists
        
    Returns:
        Shortest path length (hops) or disconnected_value
    """
    # Normalize node IDs
    node_a = normalize_node_id(a)
    node_b = normalize_node_id(b)
    
    # Same node
    if node_a == node_b:
        return 0
    
    # Check if nodes exist
    if node_a not in G or node_b not in G:
        return disconnected_value
    
    try:
        return nx.shortest_path_length(G, source=node_a, target=node_b)
    except nx.NetworkXNoPath:
        return disconnected_value


def g_topo(d: int, lambda_g: float) -> float:
    """
    Topology proximity kernel (exponential decay).
    
    Args:
        d: Graph distance (hops)
        lambda_g: Decay rate
        
    Returns:
        Proximity score in [0, 1]
    """
    return math.exp(-lambda_g * d)


if __name__ == "__main__":
    # Test synthetic HDFS graph
    print("=== Testing Synthetic HDFS Graph ===")
    graph = create_topology_graph(mode="synthetic")
    
    print("\nGraph Statistics:")
    stats = graph.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test distance computation
    print("\n=== Testing Distance Computation ===")
    test_pairs = [
        ("NameNode", "NameNode"),
        ("NameNode", "SecondaryNameNode"),
        ("NameNode", "blk-101"),
        ("blk-101", "blk-102"),
        ("blk-101", "blk-201"),
        ("blk-101", "blk-301"),
        ("blk-101", "unknown-node"),
    ]
    
    for node_a, node_b in test_pairs:
        distance = graph.compute_graph_distance(node_a, node_b)
        proximity = graph.topology_proximity_score(node_a, node_b, lambda_g=0.5)
        print(f"{node_a} <-> {node_b}: distance={distance:.1f}, proximity={proximity:.4f}")
    
    # Test save/load
    print("\n=== Testing Save/Load ===")
    graph.save_to_file("data/topology/hdfs_graph.json")
    
    graph2 = create_topology_graph(mode="static", graph_file="data/topology/hdfs_graph.json")
    print("Loaded graph successfully")
    
    # Verify loaded graph
    distance = graph2.compute_graph_distance("NameNode", "blk-101")
    print(f"Distance check after load: NameNode <-> blk-101 = {distance:.1f}")
