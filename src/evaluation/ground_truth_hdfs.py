"""
HDFS Ground Truth Builder (Silver-Standard Protocol)

Applies the same two-pass streaming silver-standard protocol as BGL
(§5.1.1) to HDFS, with one enhancement: HDFS provides explicit per-block
anomaly labels (anomaly_label.csv), so only blocks labelled "Anomaly" are
eligible as incident seeds.  This tightens the candidate pool without
changing the methodology.

Ground truth schema (matches BGL output):
  incidents.jsonl : one JSON record per incident
  qrels.jsonl     : one JSON record per incident, listing relevant template_ids

Incident trigger = first log for an anomalous block that matches a known
HDFS failure pattern.  Relevant templates = all unique templates associated
with logs from the same block_id (block-scoped relevance, analogous to
BGL's node-scoped time-window relevance).

Usage:
    python scripts/build_hdfs_ground_truth.py \
        --parsed-logs data/processed/hdfs/parsed.jsonl \
        --template-map data/processed/hdfs/template_map.jsonl \
        --anomaly-labels data/raw/HDFS/anomaly_label.csv \
        --output data/evaluation/hdfs_ground_truth.jsonl
"""

import csv
import json
import logging
import random
import re
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDFS Failure Marker Patterns
# These define what constitutes an "incident trigger" for HDFS.
# Pattern source: Loghub dataset documentation + HDFS DataNode error taxonomy.
# ---------------------------------------------------------------------------

HDFS_FAILURE_MARKERS = {
    'receive_block_error': {
        'patterns': [
            r'Exception in receiveBlock',
            r'PacketResponder.*Exception',
            r'IOException.*receiveBlock',
            r'receiveBlock.*exception',
        ],
        'severity': 'high',
    },
    'block_write_failure': {
        'patterns': [
            r'Failed to write block',
            r'IOException in appendBlock',
            r'error writing block',
            r'Failed to write',
        ],
        'severity': 'high',
    },
    'disk_io_error': {
        'patterns': [
            r'disk I/O error',
            r'IOException.*disk',
            r'BlockSender.*IOException',
            r'DiskOutOfSpace',
        ],
        'severity': 'high',
    },
    'block_corruption': {
        'patterns': [
            r'corrupt block',
            r'checksum.*mismatch',
            r'Got exception while serving',
            r'Checksum.*error',
        ],
        'severity': 'critical',
    },
    'replication_failure': {
        'patterns': [
            r'Failed to replicate',
            r'Unexpected.*block',
            r'Not valid to send block',
            r'replication.*failed',
        ],
        'severity': 'medium',
    },
    'connection_error': {
        'patterns': [
            r'Connection refused',
            r'ConnectException',
            r'Broken pipe',
            r'java\.net\.SocketException',
        ],
        'severity': 'medium',
    },
}

_BLOCK_RE = re.compile(r'blk_-?\d+')


def classify_failure(message: str) -> Optional[Tuple[str, str]]:
    """
    Classify log message against HDFS failure markers.

    Returns:
        (failure_type, severity) or None if no match
    """
    msg_lower = message.lower()
    for ftype, cfg in HDFS_FAILURE_MARKERS.items():
        for pattern in cfg['patterns']:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return (ftype, cfg['severity'])
    return None


# ---------------------------------------------------------------------------
# Ground Truth Builder
# ---------------------------------------------------------------------------

class HDFSGroundTruthBuilder:
    """
    Two-pass streaming ground truth builder for HDFS.

    Pass 1: Load anomaly labels + stream parsed logs to index
            (block_id → [log_id]) for anomalous blocks only.
    Pass 2: For each anomalous block identify the trigger log and
            collect relevant template_ids.

    Memory: O(anomalous_blocks × avg_block_logs)
    No full logs/template_map loading into Python RAM.
    """

    def __init__(
        self,
        parsed_logs_file: str,
        template_map_file: str,
        anomaly_labels_file: str,
        output_file: str,
        seed: int = 42,
        max_incidents: int = 100,
        min_relevant: int = 3,
        severity_filter: Optional[List[str]] = None,
    ):
        self.parsed_logs_file  = Path(parsed_logs_file)
        self.template_map_file = Path(template_map_file)
        self.anomaly_labels_file = Path(anomaly_labels_file)
        self.output_file       = Path(output_file)
        self.seed              = seed
        self.max_incidents     = max_incidents
        self.min_relevant      = min_relevant
        self.severity_filter   = set(severity_filter or ['critical', 'high', 'medium'])

        random.seed(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_anomaly_labels(self) -> Set[str]:
        """Load anomalous block IDs from anomaly_label.csv."""
        anomalous: Set[str] = set()

        if not self.anomaly_labels_file.exists():
            raise FileNotFoundError(
                f"Anomaly labels not found: {self.anomaly_labels_file}\n"
                "Expected at: data/raw/HDFS/HDFS_v1/preprocessed/anomaly_label.csv"
            )

        with open(self.anomaly_labels_file, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get('Label', '').strip() == 'Anomaly':
                    anomalous.add(row['BlockId'].strip())

        logger.info(f"Loaded {len(anomalous):,} anomalous block IDs")
        return anomalous

    def _load_template_map(self) -> Dict[str, dict]:
        """
        Load template_map.jsonl into a dict keyed by log_id.

        For HDFS, template_map.jsonl also carries block_id per record.
        Memory note: template_map for HDFS (~11 M lines) is too large to
        fully load; we only keep records for anomalous blocks.
        """
        raise NotImplementedError  # Not used — see pass1 below

    def _stream_parsed_logs(self):
        """Stream parsed.jsonl line by line."""
        with open(self.parsed_logs_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def _stream_template_map(self):
        """Stream template_map.jsonl line by line."""
        with open(self.template_map_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # Pass 1 — index anomalous blocks
    # ------------------------------------------------------------------

    def pass1_index_anomalous_blocks(
        self, anomalous_block_ids: Set[str]
    ) -> Dict[str, List[dict]]:
        """
        Stream parsed logs once; build block_id → [log_record] index
        for anomalous blocks only.

        Returns:
            block_index: {block_id: [log_record, ...]}  (only anomalous blocks)
        """
        logger.info("PASS 1: Indexing logs for anomalous blocks …")
        block_index: Dict[str, List[dict]] = defaultdict(list)
        total = matched = 0

        for log in self._stream_parsed_logs():
            total += 1
            block_id = log.get('block_id')
            if block_id and block_id in anomalous_block_ids:
                block_index[block_id].append(log)
                matched += 1

            if total % 1_000_000 == 0:
                logger.info(
                    f"  Scanned {total:,} logs, "
                    f"{matched:,} matched anomalous blocks …"
                )

        logger.info(
            f"Pass 1 done. Scanned={total:,}  "
            f"Anomalous blocks with logs={len(block_index):,}"
        )
        return dict(block_index)

    # ------------------------------------------------------------------
    # Pass 2 — build template_id index for anomalous blocks
    # ------------------------------------------------------------------

    def pass2_index_template_map(
        self, anomalous_block_ids: Set[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Stream template_map.jsonl once; build block_id → {log_id: template_id}
        for anomalous blocks only.

        Returns:
            tm_index: {block_id: {log_id: template_id}}
        """
        logger.info("PASS 2: Building template_id index for anomalous blocks …")
        tm_index: Dict[str, Dict[str, str]] = defaultdict(dict)
        total = matched = 0

        for rec in self._stream_template_map():
            total += 1
            block_id = rec.get('block_id')
            if block_id and block_id in anomalous_block_ids:
                tm_index[block_id][rec['log_id']] = rec['template_id']
                matched += 1

            if total % 1_000_000 == 0:
                logger.info(f"  Scanned {total:,} template-map rows …")

        logger.info(
            f"Pass 2 done. Scanned={total:,}  "
            f"Blocks with template_ids={len(tm_index):,}"
        )
        return dict(tm_index)

    # ------------------------------------------------------------------
    # Incident construction
    # ------------------------------------------------------------------

    def build_incidents(
        self,
        block_index: Dict[str, List[dict]],
        tm_index: Dict[str, Dict[str, str]],
    ) -> Tuple[List[dict], List[dict]]:
        """
        For each eligible anomalous block:
          - Find the first failure-pattern log as the incident trigger
          - Collect all unique template_ids for that block as relevant set
          - Grade: triggering template_id gets grade=2, rest grade=1

        Returns:
            (incidents, qrels)
        """
        logger.info("Building incidents from indexed blocks …")
        incidents = []
        qrels     = []

        for block_id, logs in block_index.items():
            if block_id not in tm_index:
                continue  # no template mappings for this block

            tm_for_block = tm_index[block_id]

            # Sort logs by timestamp for chronological processing
            sorted_logs = sorted(
                logs,
                key=lambda l: l.get('timestamp') or ''
            )

            # Find trigger = first log with a failure marker
            trigger_log = None
            trigger_type = None
            trigger_severity = None

            for log in sorted_logs:
                classification = classify_failure(log['message'])
                if classification:
                    ftype, sev = classification
                    if sev in self.severity_filter:
                        trigger_log      = log
                        trigger_type     = ftype
                        trigger_severity = sev
                        break

            if trigger_log is None:
                continue  # block has anomaly label but no matching failure pattern

            # Get trigger template_id
            trigger_tid = tm_for_block.get(trigger_log['log_id'])
            if not trigger_tid:
                continue

            # Collect all unique template_ids for this block
            all_tids: Set[str] = set()
            for log in sorted_logs:
                tid = tm_for_block.get(log['log_id'])
                if tid:
                    all_tids.add(tid)

            if len(all_tids) < self.min_relevant:
                continue

            # Build relevance grades:
            #   2 = the triggering template (most relevant)
            #   1 = other templates in the same block (contextually relevant)
            relevance_grade = {
                tid: (2 if tid == trigger_tid else 1)
                for tid in all_tids
            }

            # Parse trigger timestamp
            ts_str = trigger_log.get('timestamp', '')
            try:
                ts_unix = datetime.fromisoformat(ts_str).timestamp()
            except (ValueError, TypeError):
                ts_unix = 0.0

            incident = {
                'incident_id':          trigger_log['log_id'],
                'incident_template_id': trigger_tid,
                'incident_time':        ts_unix,
                'incident_node':        trigger_log.get('node_id', 'unknown'),
                'incident_text':        trigger_log['message'],
                'failure_type':         trigger_type,
                'severity':             trigger_severity,
                'block_id':             block_id,            # HDFS-specific
            }
            qrel = {
                'incident_id':          trigger_log['log_id'],
                'relevant_template_ids': list(all_tids),
                'relevance_grade':       relevance_grade,
            }

            incidents.append(incident)
            qrels.append(qrel)

        logger.info(f"Built {len(incidents)} raw incident candidates")
        return incidents, qrels

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self):
        """Run the full ground truth construction pipeline."""
        logger.info("=" * 70)
        logger.info("HDFS GROUND TRUTH BUILDER (Silver-Standard + Anomaly Labels)")
        logger.info("=" * 70)
        logger.info(f"Parsed logs  : {self.parsed_logs_file}")
        logger.info(f"Template map : {self.template_map_file}")
        logger.info(f"Anomaly labels: {self.anomaly_labels_file}")
        logger.info(f"Output       : {self.output_file}")
        logger.info(f"Max incidents: {self.max_incidents}")
        logger.info(f"Min relevant : {self.min_relevant}")
        logger.info(f"Seed         : {self.seed}")
        logger.info("=" * 70)

        # Load labels
        anomalous_block_ids = self._load_anomaly_labels()

        # Pass 1: index logs for anomalous blocks
        block_index = self.pass1_index_anomalous_blocks(anomalous_block_ids)

        # Pass 2: index template_ids for anomalous blocks
        tm_index = self.pass2_index_template_map(anomalous_block_ids)

        # Build incidents
        incidents, qrels = self.build_incidents(block_index, tm_index)

        # Sample down to max_incidents (deterministic)
        if len(incidents) > self.max_incidents:
            logger.info(f"Sampling {self.max_incidents} from {len(incidents)} candidates …")
            sample_idx = random.sample(range(len(incidents)), self.max_incidents)
            incidents = [incidents[i] for i in sorted(sample_idx)]
            qrels     = [qrels[i]     for i in sorted(sample_idx)]

        # Sort by incident_time for reproducibility
        paired = sorted(zip(incidents, qrels), key=lambda x: x[0]['incident_time'])
        incidents = [p[0] for p in paired]
        qrels     = [p[1] for p in paired]

        # Write output
        output_base    = str(self.output_file).replace('.jsonl', '')
        incidents_path = Path(output_base + '_incidents.jsonl')
        qrels_path     = Path(output_base + '_qrels.jsonl')

        incidents_path.parent.mkdir(parents=True, exist_ok=True)

        with open(incidents_path, 'w', encoding='utf-8') as fh:
            for inc in incidents:
                fh.write(json.dumps(inc) + '\n')

        with open(qrels_path, 'w', encoding='utf-8') as fh:
            for qr in qrels:
                fh.write(json.dumps(qr) + '\n')

        logger.info(f"Wrote {len(incidents)} incidents → {incidents_path}")
        logger.info(f"Wrote {len(qrels)} qrels      → {qrels_path}")
        logger.info("Ground truth build complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build HDFS ground truth (silver-standard + anomaly labels)"
    )
    parser.add_argument("--parsed-logs",
                        default="data/processed/hdfs/parsed.jsonl")
    parser.add_argument("--template-map",
                        default="data/processed/hdfs/template_map.jsonl")
    parser.add_argument("--anomaly-labels",
                        default="data/raw/HDFS/anomaly_label.csv")
    parser.add_argument("--output",
                        default="data/evaluation/hdfs_ground_truth.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-incidents", type=int, default=100)
    parser.add_argument("--min-relevant",  type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    builder = HDFSGroundTruthBuilder(
        parsed_logs_file  = args.parsed_logs,
        template_map_file = args.template_map,
        anomaly_labels_file = args.anomaly_labels,
        output_file       = args.output,
        seed              = args.seed,
        max_incidents     = args.max_incidents,
        min_relevant      = args.min_relevant,
    )
    builder.build()
