"""
Thunderbird Template Extraction

Extracts canonical log templates from parsed Thunderbird logs.
Builds on the same regex-normalisation approach as template_bgl.py and
template_hdfs.py, with Thunderbird-specific token patterns for cluster
node names, syslog PIDs, and InfiniBand identifiers.

Output schema is identical to the BGL/HDFS pipelines, so all downstream
scripts (build_occurrence_store, build_faiss_index, run_experiments_v2)
work unchanged.
"""

import sys
import json
import hashlib
import re
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_utils import read_json_or_jsonl


# ---------------------------------------------------------------------------
# Template normalisation — Thunderbird-specific patterns added on top of
# BGL/HDFS set.
# ---------------------------------------------------------------------------

def extract_template(message: str) -> str:
    """
    Normalise a Thunderbird log message into a canonical template.

    Replacement order matters: more-specific patterns first.

    Thunderbird additions (beyond BGL/HDFS patterns):
      \\[\\d+\\]            → [<PID>]   (syslog process-id in brackets)
      \\b(cn|bn|tn|mn|ln)\\d+\\b  → <NODE>  (cluster compute/blade/login nodes)
      lid=\\d+             → lid=<LID>  (InfiniBand local ID)
      InfiniHost\\d*       → <IB_DEV>   (InfiniBand device name)
    """
    t = message

    # 1. Syslog PIDs: component[12345]: → component[<PID>]:
    t = re.sub(r'\[\d+\]', '[<PID>]', t)

    # 2. Cluster node names (cn994, bn257, tn03, mn8, ln1, etc.)
    t = re.sub(r'\b(cn|bn|tn|mn|ln)\d+\b', '<NODE>', t, flags=re.IGNORECASE)

    # 3. InfiniBand device names (InfiniHost0, InfiniHost1, ...)
    t = re.sub(r'\bInfiniHost\d*\b', '<IB_DEV>', t)

    # 4. InfiniBand LID values  (lid=1234)
    t = re.sub(r'\blid=\d+\b', 'lid=<LID>', t, flags=re.IGNORECASE)

    # 5. Hex addresses / error codes  (0x1a2b3c4d)
    t = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', t)

    # 6. IP:port pairs  (before plain IP)
    t = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<ADDR>', t)

    # 7. Plain IP addresses
    t = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', t)

    # 8. UUIDs
    t = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}'
        r'-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        '<UUID>', t
    )

    # 9. Unix / Windows file paths (kernel module paths like /mnt_projects/...)
    t = re.sub(r'/[\w/\-\.]+', '<PATH>', t)
    t = re.sub(r'[A-Z]:\\[\w\\\-\.]+', '<PATH>', t)

    # 10. Long integers (≥4 digits) — preserves small error codes and event IDs
    t = re.sub(r'\b\d{4,}\b', '<NUM>', t)

    # 11. ISO timestamps
    t = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(\.\d+)?', '<TS>', t)
    t = re.sub(r'\d{2}:\d{2}:\d{2}(\.\d+)?', '<TS>', t)

    # 12. Normalise whitespace
    return ' '.join(t.split())


def create_template_id(template: str) -> str:
    """Deterministic SHA-1 id for a template string."""
    return hashlib.sha1(template.encode('utf-8')).hexdigest()


def extract_templates(
    input_file: str,
    templates_output: str,
    template_map_output: str,
):
    """
    Extract templates from parsed Thunderbird logs.

    Streaming implementation: reads line-by-line, writes template_map
    incrementally.  Only template metadata is held in memory.

    Args:
        input_file:           Path to parsed.jsonl  (may be on E: drive)
        templates_output:     Destination templates.jsonl
        template_map_output:  Destination template_map.jsonl (may be on E: drive)
    """
    print("=" * 70)
    print("Thunderbird Template Extraction (Streaming)")
    print("=" * 70)
    print(f"Input : {input_file}")
    print(f"Templates  : {templates_output}")
    print(f"Template map: {template_map_output}")
    print()

    Path(templates_output).parent.mkdir(parents=True, exist_ok=True)
    Path(template_map_output).parent.mkdir(parents=True, exist_ok=True)

    # Accumulate template metadata only (small)
    template_meta: dict = defaultdict(lambda: {"count": 0, "example_message": ""})
    total = 0

    with open(template_map_output, 'w', encoding='utf-8') as map_fh:
        for log in read_json_or_jsonl(input_file, stream=True):
            total += 1
            if 'message' not in log or 'log_id' not in log:
                continue

            template    = extract_template(log['message'])
            template_id = create_template_id(template)

            meta = template_meta[template_id]
            meta['count'] += 1
            if not meta['example_message']:
                meta['example_message'] = log['message']
                meta['template']        = template

            map_record = {
                "log_id":      log['log_id'],
                "template_id": template_id,
                "timestamp":   log.get('timestamp', ''),
                "node_id":     log.get('node_id', 'unknown'),
                # No block_id for Thunderbird
            }
            map_fh.write(json.dumps(map_record) + '\n')

            if total % 1_000_000 == 0:
                print(f"  Processed {total:,} logs, "
                      f"{len(template_meta):,} unique templates …", flush=True)

    print(f"\nTotal logs      : {total:,}")
    print(f"Unique templates: {len(template_meta):,}")
    print(f"Compression     : {total / max(len(template_meta), 1):.1f}x")

    # Write templates.jsonl (sorted by frequency)
    templates = [
        {
            "template_id":     tid,
            "template":        meta.get('template', extract_template(meta['example_message'])),
            "example_message": meta['example_message'],
            "count":           meta['count'],
        }
        for tid, meta in template_meta.items()
    ]
    templates.sort(key=lambda x: x['count'], reverse=True)

    with open(templates_output, 'w', encoding='utf-8') as out_fh:
        for t in templates:
            out_fh.write(json.dumps(t) + '\n')

    print(f"Templates written: {templates_output}")
    print(f"Template map written: {template_map_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract log templates from parsed Thunderbird JSONL"
    )
    parser.add_argument("--input",
                        default=r"E:\Paper\LAR-RAG\data\processed\thunderbird\parsed.jsonl",
                        help="Path to parsed.jsonl")
    parser.add_argument("--templates-output",
                        default="data/processed/thunderbird/templates.jsonl",
                        help="Destination templates.jsonl")
    parser.add_argument("--template-map-output",
                        default=r"E:\Paper\LAR-RAG\data\processed\thunderbird\template_map.jsonl",
                        help="Destination template_map.jsonl")
    args = parser.parse_args()

    extract_templates(
        args.input,
        args.templates_output,
        args.template_map_output,
    )
