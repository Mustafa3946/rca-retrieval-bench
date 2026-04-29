"""
HDFS Template Extraction

Extracts canonical log templates from parsed HDFS logs.
Re-uses the same regex-normalisation approach as template_bgl.py with
additional HDFS-specific token patterns (block IDs, DataNode ports).

Output schema is identical to the BGL pipeline, so all downstream scripts
(build_occurrence_store, build_faiss_index, run_experiments_v2) work unchanged.
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
# Template normalisation — HDFS specific patterns added on top of BGL set
# ---------------------------------------------------------------------------

def extract_template(message: str) -> str:
    """
    Normalise an HDFS log message into a canonical template.

    Replacement order matters: more-specific patterns first.

    HDFS additions (beyond BGL patterns):
      blk_-?\\d+            → <BLOCK>   (HDFS block identifiers)
      \\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}:\\d+  → <ADDR>  (ip:port pairs)
    """
    t = message

    # 1. HDFS block IDs  (before generic number replacement)
    t = re.sub(r'blk_-?\d+', '<BLOCK>', t)

    # 2. IP:port pairs  (before plain IP replacement)
    t = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<ADDR>', t)

    # 3. Plain IP addresses
    t = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', t)

    # 4. Hex addresses
    t = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', t)

    # 5. UUIDs
    t = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}'
        r'-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        '<UUID>', t
    )

    # 6. Unix / Windows file paths
    t = re.sub(r'/[\w/\-\.]+', '<PATH>', t)
    t = re.sub(r'[A-Z]:\\[\w\\\-\.]+', '<PATH>', t)

    # 7. Long integers (≥4 digits) — preserves small error codes
    t = re.sub(r'\b\d{4,}\b', '<NUM>', t)

    # 8. ISO timestamps
    t = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(\.\d+)?', '<TS>', t)
    t = re.sub(r'\d{2}:\d{2}:\d{2}(\.\d+)?', '<TS>', t)

    # 9. Normalise whitespace
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
    Extract templates from parsed HDFS logs.

    Streaming implementation: reads line-by-line, writes template_map
    incrementally.  Only template metadata is held in memory.

    Args:
        input_file:           Path to parsed.jsonl
        templates_output:     Destination templates.jsonl
        template_map_output:  Destination template_map.jsonl
    """
    print("=" * 70)
    print("HDFS Template Extraction (Streaming)")
    print("=" * 70)
    print(f"Input : {input_file}")
    print(f"Templates  : {templates_output}")
    print(f"Template map: {template_map_output}")
    print()

    Path(templates_output).parent.mkdir(parents=True, exist_ok=True)
    Path(template_map_output).parent.mkdir(parents=True, exist_ok=True)

    # Accumulate template metadata only
    template_meta: dict = defaultdict(lambda: {"count": 0, "example_message": ""})
    total = 0

    with open(template_map_output, 'w', encoding='utf-8') as map_fh:
        for log in read_json_or_jsonl(input_file):
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
                "block_id":    log.get('block_id'),    # HDFS-specific; None for non-block logs
            }
            map_fh.write(json.dumps(map_record) + '\n')

            if total % 200_000 == 0:
                print(f"  Processed {total:,} logs, "
                      f"{len(template_meta):,} unique templates …")

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
    templates.sort(key=lambda x: (-x['count'], x['template_id']))

    with open(templates_output, 'w', encoding='utf-8') as out_fh:
        for t in templates:
            out_fh.write(json.dumps(t) + '\n')

    print(f"Saved {len(templates):,} templates → {templates_output}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HDFS log templates")
    parser.add_argument("--input",
                        default="data/processed/hdfs/parsed.jsonl")
    parser.add_argument("--templates-output",
                        default="data/processed/hdfs/templates.jsonl")
    parser.add_argument("--template-map-output",
                        default="data/processed/hdfs/template_map.jsonl")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    extract_templates(args.input, args.templates_output, args.template_map_output)
