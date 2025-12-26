"""
Node ID normalization for LAR-RAG.

Ensures consistent node identifiers across:
- Raw log entries
- Parsed logs
- Ground truth incidents
- Topology graphs

This is critical for topology distance computation to work correctly.
"""

import re
from typing import Optional


def normalize_node_id(raw: str) -> str:
    """
    Normalize a node identifier to a canonical form.
    
    Handles various node ID formats found in distributed systems:
    - HDFS: "/10.251.70.211" -> "211" or "10.251.70.211"
    - BGL: "R45-M0-N3-C:J12-U11" -> "R45-M0-N3"
    - Thunderbird: "tbird-admin4.sandia.gov" -> "tbird-admin4"
    - Generic numeric: "19", " 19 ", "Node19" -> "19"
    
    Args:
        raw: Raw node identifier string from logs or config
        
    Returns:
        Normalized node identifier (lowercase, trimmed, canonical)
        
    Examples:
        >>> normalize_node_id("/10.251.70.211")
        '211'
        >>> normalize_node_id("  Node19  ")
        '19'
        >>> normalize_node_id("R45-M0-N3-C:J12-U11")
        'r45-m0-n3'
    """
    if not raw:
        return ""
    
    # Trim whitespace
    node = raw.strip()
    
    # Lowercase for consistency
    node = node.lower()
    
    # HDFS IP format: "/10.251.70.211" -> extract last octet for simplicity
    # This creates a compact node ID space (0-255) suitable for graph construction
    if node.startswith("/") and "." in node:
        octets = node.strip("/").split(".")
        if len(octets) == 4 and all(o.isdigit() for o in octets):
            # Use last octet as node ID (assumes unique within cluster)
            return octets[-1]
    
    # BGL format: "R45-M0-N3-C:J12-U11" -> keep rack-midplane-node only
    if re.match(r"r\d+-m\d+-n\d+", node):
        parts = node.split("-c:")
        if len(parts) > 0:
            return parts[0]
    
    # Thunderbird hostname: "tbird-admin4.sandia.gov" -> keep host only
    if "." in node and not node[0].isdigit():
        hostname = node.split(".")[0]
        return hostname
    
    # Generic "Node19" or "node19" -> extract number
    match = re.search(r"node[\s_-]*(\d+)", node)
    if match:
        return match.group(1)
    
    # Already numeric: "19", "42" -> keep as-is
    if node.isdigit():
        return node
    
    # Fallback: return lowercase trimmed version
    return node


def parse_node_from_log(log_line: str, dataset: str = "hdfs") -> Optional[str]:
    """
    Extract and normalize node ID from a raw log line.
    
    Args:
        log_line: Raw log line text
        dataset: Dataset type ("hdfs", "bgl", "thunderbird")
        
    Returns:
        Normalized node ID or None if not found
        
    Examples:
        >>> parse_node_from_log("081109 203518 19 INFO dfs.DataNode$PacketResponder...", "hdfs")
        '19'
    """
    if dataset == "hdfs":
        # HDFS format: "YYMMDD HHMMSS NODE_ID LEVEL message"
        parts = log_line.split()
        if len(parts) >= 3 and parts[2].strip():
            return normalize_node_id(parts[2])
    
    elif dataset == "bgl":
        # BGL format: starts with node identifier
        match = re.match(r"^(R\d+-M\d+-N\d+(-C:\S+)?)", log_line)
        if match:
            return normalize_node_id(match.group(1))
    
    elif dataset == "thunderbird":
        # Thunderbird format: "YYYY-MM-DD hostname message"
        parts = log_line.split()
        if len(parts) >= 2:
            return normalize_node_id(parts[1])
    
    return None


def validate_node_ids(node_ids: list[str], dataset: str = "hdfs") -> dict:
    """
    Validate a list of node IDs and report statistics.
    
    Args:
        node_ids: List of node IDs to validate
        dataset: Dataset type for context
        
    Returns:
        Dictionary with validation statistics:
        - unique_nodes: number of unique normalized node IDs
        - empty_nodes: number of empty/invalid IDs
        - format_distribution: counts of different ID formats
        - sample_ids: first 10 unique IDs
    """
    unique = set()
    empty = 0
    format_counts = {"numeric": 0, "ip": 0, "hostname": 0, "other": 0}
    
    for node_id in node_ids:
        normalized = normalize_node_id(node_id)
        
        if not normalized:
            empty += 1
            continue
        
        unique.add(normalized)
        
        # Classify format
        if normalized.isdigit():
            format_counts["numeric"] += 1
        elif "." in node_id and node_id[0].isdigit():
            format_counts["ip"] += 1
        elif "-" in normalized or "." in normalized:
            format_counts["hostname"] += 1
        else:
            format_counts["other"] += 1
    
    return {
        "unique_nodes": len(unique),
        "empty_nodes": empty,
        "format_distribution": format_counts,
        "sample_ids": sorted(list(unique))[:10]
    }


def build_node_mapping(raw_nodes: list[str]) -> dict[str, str]:
    """
    Build a mapping from raw node IDs to normalized IDs.
    
    Useful for bulk conversion and debugging.
    
    Args:
        raw_nodes: List of raw node identifier strings
        
    Returns:
        Dictionary mapping raw -> normalized node IDs
    """
    mapping = {}
    for raw in raw_nodes:
        normalized = normalize_node_id(raw)
        if normalized:
            mapping[raw] = normalized
    return mapping


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "/10.251.70.211",
        "  19  ",
        "Node19",
        "node_42",
        "R45-M0-N3-C:J12-U11",
        "tbird-admin4.sandia.gov",
        "192.168.1.100",
        ""
    ]
    
    print("Node Normalization Tests:")
    print("-" * 60)
    for raw in test_cases:
        normalized = normalize_node_id(raw)
        print(f"{raw!r:30} -> {normalized!r}")
    
    print("\n" + "=" * 60)
    stats = validate_node_ids(test_cases)
    print(f"Unique nodes: {stats['unique_nodes']}")
    print(f"Empty nodes: {stats['empty_nodes']}")
    print(f"Sample IDs: {stats['sample_ids']}")
