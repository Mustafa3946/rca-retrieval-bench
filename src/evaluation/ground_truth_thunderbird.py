"""
Thunderbird Ground Truth Builder (Silver-Standard Protocol)

Applies the silver-standard protocol (§5.1.1) to Thunderbird, adapted
for its per-line anomaly labels (no block grouping, unlike HDFS).

Anomaly labels are embedded directly in Thunderbird.log as the first
whitespace-separated field: '-' = normal, any other value = anomalous
(e.g., ECC, VAPI, CPU, SCSI, MPT).

Incident model
--------------
A burst of anomalous activity on a single node within a 1-hour window
constitutes one "incident."  More precisely:

  1. Collect all anomalous log records (label != '-') from parsed.jsonl.
  2. For each node_id, sort anomalous records by unix_ts.
  3. Split into clusters wherever consecutive gap > INCIDENT_GAP_S (3600 s).
  4. The first log of each cluster is the incident trigger.
  5. Relevant templates = all distinct template_ids from logs of the SAME
     node_id within [trigger_ts - BEFORE_S, trigger_ts + AFTER_S], queried
     from the DuckDB occurrence store (already built in step 4).

Using DuckDB for the template query avoids streaming the large (≥3 GB)
template_map.jsonl file — a node+time-range query on the indexed
occurrence store takes milliseconds.

Ground truth schema (matches BGL/HDFS output):
  incidents.jsonl : one JSON record per incident
  qrels.jsonl     : one JSON record per incident, listing relevant template_ids

Usage:
    python src/evaluation/ground_truth_thunderbird.py \\
        --parsed-logs  E:/Paper/LAR-RAG/data/processed/thunderbird/parsed.jsonl \\
        --occ-db       E:/Paper/LAR-RAG/data/processed/thunderbird/occurrences.duckdb \\
        --output       data/evaluation/thunderbird_ground_truth.jsonl
"""

import json
import logging
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

INCIDENT_GAP_S = 3600     # s — gap between anomalous lines that starts a new incident
BEFORE_S       = 1800     # s — relevant window before trigger
AFTER_S        = 3600     # s — relevant window after trigger


# ---------------------------------------------------------------------------
# Ground Truth Builder
# ---------------------------------------------------------------------------

class ThunderbirdGroundTruthBuilder:
    """
    Two-phase ground truth builder for Thunderbird.

    Phase 1: Stream parsed.jsonl → collect anomalous records per node_id.
             Memory: O(anomalous_lines) ≈ O(350K records for 30M-line sample)
    Phase 2: For each incident cluster, query DuckDB occurrence store for
             relevant templates in the time window around the trigger.
    """

    def __init__(
        self,
        parsed_logs_file: str,
        occ_db_path: str,
        output_file: str,
        seed: int = 42,
        max_incidents: int = 200,
        min_relevant: int = 3,
        incident_gap_s: int = INCIDENT_GAP_S,
        before_s: int = BEFORE_S,
        after_s: int = AFTER_S,
    ):
        self.parsed_logs_file = Path(parsed_logs_file)
        self.occ_db_path      = Path(occ_db_path)
        self.output_file      = Path(output_file)
        self.seed             = seed
        self.max_incidents    = max_incidents
        self.min_relevant     = min_relevant
        self.incident_gap_s   = incident_gap_s
        self.before_s         = before_s
        self.after_s          = after_s

        random.seed(seed)

    # ------------------------------------------------------------------
    # Phase 1 — collect anomalous records
    # ------------------------------------------------------------------

    def phase1_collect_anomalous(self) -> Dict[str, List[dict]]:
        """
        Stream parsed.jsonl once; collect all anomalous records grouped
        by node_id.

        Returns:
            node_anomalous: {node_id: [record, ...]}  sorted by unix_ts
        """
        logger.info("PHASE 1: Collecting anomalous records from parsed.jsonl …")
        node_anomalous: Dict[str, List[dict]] = defaultdict(list)
        total = anomalous = 0

        with open(self.parsed_logs_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get('label', '-') != '-':
                    node_anomalous[rec['node_id']].append(rec)
                    anomalous += 1

                if total % 2_000_000 == 0:
                    logger.info(
                        f"  Scanned {total:,} logs, "
                        f"{anomalous:,} anomalous on "
                        f"{len(node_anomalous):,} nodes …"
                    )

        # Sort each node's anomalous records by unix_ts
        for node_id in node_anomalous:
            node_anomalous[node_id].sort(key=lambda r: r.get('unix_ts', 0))

        logger.info(
            f"Phase 1 done. Total={total:,}  Anomalous={anomalous:,}  "
            f"Nodes with anomalies={len(node_anomalous):,}"
        )
        return dict(node_anomalous)

    # ------------------------------------------------------------------
    # Phase 2 — cluster into incidents
    # ------------------------------------------------------------------

    def phase2_cluster_incidents(
        self, node_anomalous: Dict[str, List[dict]]
    ) -> List[dict]:
        """
        For each node, split anomalous records into time-gap clusters.
        Each cluster yields one incident candidate (trigger = first record).

        Returns:
            List of incident candidate dicts.
        """
        logger.info(
            f"PHASE 2: Clustering incidents "
            f"(gap threshold={self.incident_gap_s}s) …"
        )
        candidates = []

        for node_id, records in node_anomalous.items():
            if not records:
                continue

            cluster_start = records[0]

            for i, rec in enumerate(records):
                prev_ts = records[i - 1].get('unix_ts', 0) if i > 0 else rec.get('unix_ts', 0)
                curr_ts = rec.get('unix_ts', 0)

                if i == 0 or (curr_ts - prev_ts) > self.incident_gap_s:
                    # Start a new incident cluster
                    cluster_start = rec

                    candidates.append({
                        'incident_id':          rec['log_id'],
                        'incident_node':        node_id,
                        'incident_time':        curr_ts,
                        'incident_text':        rec['message'],
                        'incident_label':       rec['label'],
                        'incident_template_id': None,  # filled from DuckDB in phase 3
                    })

        logger.info(f"Phase 2 done. Raw incident candidates: {len(candidates):,}")
        return candidates

    # ------------------------------------------------------------------
    # Phase 3 — query DuckDB for relevant templates
    # ------------------------------------------------------------------

    def phase3_build_qrels(
        self, candidates: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """
        For each incident candidate, query DuckDB occurrence store for
        all distinct template_ids within the trigger's time window on
        the same node.  Filter by min_relevant.

        Returns:
            (incidents, qrels)  — filtered and sampled lists.
        """
        logger.info(
            f"PHASE 3: Querying DuckDB for relevant templates "
            f"(window: -{self.before_s}s / +{self.after_s}s) …"
        )

        conn = duckdb.connect(str(self.occ_db_path), read_only=True)

        # Create a composite index if not already present (speeds up node+time queries)
        try:
            conn.close()
            conn = duckdb.connect(str(self.occ_db_path))  # re-open read-write for index
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_ts "
                "ON occurrences(node_id, timestamp)"
            )
            conn.execute("CHECKPOINT")
        except Exception as idx_err:
            logger.warning(f"Could not create composite index: {idx_err}")
        finally:
            conn.close()

        conn = duckdb.connect(str(self.occ_db_path), read_only=True)

        incidents = []
        qrels     = []
        skipped_no_templates = 0
        skipped_few_templates = 0

        for i, cand in enumerate(candidates):
            node_id    = cand['incident_node']
            t_trigger  = float(cand['incident_time'])
            t_start    = t_trigger - self.before_s
            t_end      = t_trigger + self.after_s

            rows = conn.execute(
                """
                SELECT DISTINCT template_id
                FROM occurrences
                WHERE node_id = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                """,
                [node_id, t_start, t_end]
            ).fetchall()

            template_ids: List[str] = [r[0] for r in rows]

            if not template_ids:
                skipped_no_templates += 1
                continue
            if len(template_ids) < self.min_relevant:
                skipped_few_templates += 1
                continue

            # Get trigger's template_id (the log closest to trigger time from same node)
            trigger_row = conn.execute(
                """
                SELECT template_id
                FROM occurrences
                WHERE node_id = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY ABS(timestamp - ?) ASC
                LIMIT 1
                """,
                [node_id, t_trigger, t_trigger + 5.0, t_trigger]
            ).fetchone()

            trigger_tid = trigger_row[0] if trigger_row else template_ids[0]
            cand['incident_template_id'] = trigger_tid

            # Grade: 2 = triggering template, 1 = others in window
            relevance_grade = {
                tid: (2 if tid == trigger_tid else 1)
                for tid in template_ids
            }

            incident = {
                'incident_id':          cand['incident_id'],
                'incident_template_id': trigger_tid,
                'incident_time':        t_trigger,
                'incident_node':        node_id,
                'incident_text':        cand['incident_text'],
                'failure_type':         cand['incident_label'],
                'severity':             'alert',
            }
            qrel = {
                'incident_id':           cand['incident_id'],
                'relevant_template_ids': template_ids,
                'relevance_grade':       relevance_grade,
            }

            incidents.append(incident)
            qrels.append(qrel)

            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i+1:,} candidates, "
                            f"{len(incidents):,} valid so far …")

        conn.close()

        logger.info(
            f"Phase 3 done. Valid={len(incidents):,}  "
            f"Skipped (no templates)={skipped_no_templates:,}  "
            f"Skipped (< {self.min_relevant} templates)={skipped_few_templates:,}"
        )
        return incidents, qrels

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_incidents(
        self,
        incidents: List[dict],
        qrels: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        """Sample max_incidents incident/qrel pairs."""
        if len(incidents) <= self.max_incidents:
            logger.info(f"Keeping all {len(incidents):,} incidents (≤ max_incidents)")
            return incidents, qrels

        idx = list(range(len(incidents)))
        random.shuffle(idx)
        sel = sorted(idx[:self.max_incidents])

        sampled_incidents = [incidents[i] for i in sel]
        sampled_qrels     = [qrels[i]     for i in sel]

        logger.info(f"Sampled {len(sampled_incidents):,} incidents from {len(incidents):,}")
        return sampled_incidents, sampled_qrels

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_output(
        self,
        incidents: List[dict],
        qrels: List[dict],
    ):
        """Write incidents.jsonl and qrels.jsonl."""
        stem = self.output_file.stem
        if stem.endswith('_ground_truth'):
            base = stem[:-len('_ground_truth')]
        else:
            base = stem

        parent   = self.output_file.parent
        inc_path = parent / f"{base}_ground_truth_incidents.jsonl"
        qrel_path = parent / f"{base}_ground_truth_qrels.jsonl"

        parent.mkdir(parents=True, exist_ok=True)

        with open(inc_path, 'w', encoding='utf-8') as fh:
            for inc in incidents:
                fh.write(json.dumps(inc) + '\n')

        with open(qrel_path, 'w', encoding='utf-8') as fh:
            for q in qrels:
                fh.write(json.dumps(q) + '\n')

        logger.info(f"Written: {inc_path}  ({len(incidents):,} incidents)")
        logger.info(f"Written: {qrel_path}  ({len(qrels):,} qrels)")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self):
        """Run the full ground truth construction pipeline."""
        logger.info("=" * 70)
        logger.info("THUNDERBIRD GROUND TRUTH BUILDER (Silver-Standard Protocol)")
        logger.info("=" * 70)
        logger.info(f"Parsed logs: {self.parsed_logs_file}")
        logger.info(f"Occ DB     : {self.occ_db_path}")
        logger.info(f"Output     : {self.output_file}")
        logger.info(f"Max inc.   : {self.max_incidents}")
        logger.info(f"Min relev. : {self.min_relevant}")
        logger.info(f"Seed       : {self.seed}")
        logger.info("=" * 70)

        node_anomalous = self.phase1_collect_anomalous()
        candidates     = self.phase2_cluster_incidents(node_anomalous)
        incidents, qrels = self.phase3_build_qrels(candidates)
        incidents, qrels = self.sample_incidents(incidents, qrels)
        self.write_output(incidents, qrels)

        logger.info("=" * 70)
        logger.info(f"Ground truth complete: {len(incidents):,} incidents")
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Thunderbird ground truth (silver-standard)"
    )
    parser.add_argument("--parsed-logs",
                        default=r"E:\Paper\LAR-RAG\data\processed\thunderbird\parsed.jsonl",
                        help="Path to parsed.jsonl")
    parser.add_argument("--occ-db",
                        default=r"E:\Paper\LAR-RAG\data\processed\thunderbird\occurrences.duckdb",
                        help="Path to DuckDB occurrence store")
    parser.add_argument("--output",
                        default="data/evaluation/thunderbird_ground_truth.jsonl",
                        help="Base output path (incidents + qrels derived from this)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--max-incidents", type=int, default=200)
    parser.add_argument("--min-relevant",  type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ThunderbirdGroundTruthBuilder(
        parsed_logs_file=args.parsed_logs,
        occ_db_path=args.occ_db,
        output_file=args.output,
        seed=args.seed,
        max_incidents=args.max_incidents,
        min_relevant=args.min_relevant,
    ).build()
