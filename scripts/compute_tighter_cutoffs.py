#!/usr/bin/env python3
"""
compute_tighter_cutoffs.py

Computes MRR@1, nDCG@3, and Recall@5 from stored per-query JSONL result files.
No experiments are re-run — ranked lists and relevance grades are read from disk.

Usage:
    python scripts/compute_tighter_cutoffs.py
"""

import json
import math
import glob
from pathlib import Path
from typing import Dict, List


def mrr_at_k(retrieved: List[str], relevant_grades: Dict[str, int], k: int) -> float:
    """MRR@k: reciprocal rank of the first relevant doc in top-k."""
    for rank, tid in enumerate(retrieved[:k], start=1):
        if relevant_grades.get(tid, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant_grades: Dict[str, int], k: int) -> float:
    """nDCG@k using graded relevance."""
    def dcg(items):
        score = 0.0
        for i, tid in enumerate(items, start=1):
            rel = relevant_grades.get(tid, 0)
            score += rel / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved[:k])

    # Ideal: top-k grades sorted descending
    ideal_items = sorted(relevant_grades.values(), reverse=True)[:k]
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_items))

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant_grades: Dict[str, int], k: int) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    n_relevant = sum(1 for g in relevant_grades.values() if g > 0)
    if n_relevant == 0:
        return 0.0
    hits = sum(1 for tid in retrieved[:k] if relevant_grades.get(tid, 0) > 0)
    return hits / n_relevant


def compute_for_dir(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Compute tighter-cutoff metrics for all methods in a results directory."""
    method_metrics = {}

    for jsonl_path in sorted(results_dir.glob("per_query_*.jsonl")):
        method_name = jsonl_path.stem.replace("per_query_", "")
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

        mrr1_scores, ndcg3_scores, recall5_scores = [], [], []

        for row in rows:
            retrieved = row.get("retrieved_template_ids", [])
            grades = row.get("relevance_grades", {})

            mrr1_scores.append(mrr_at_k(retrieved, grades, k=1))
            ndcg3_scores.append(ndcg_at_k(retrieved, grades, k=3))
            recall5_scores.append(recall_at_k(retrieved, grades, k=5))

        n = len(rows)
        method_metrics[method_name] = {
            "n_queries": n,
            "MRR@1":    sum(mrr1_scores)    / n if n else 0.0,
            "nDCG@3":   sum(ndcg3_scores)   / n if n else 0.0,
            "Recall@5": sum(recall5_scores) / n if n else 0.0,
        }

    return method_metrics


def print_table(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    col_w = 22
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Method':<{col_w}} {'MRR@1':>8} {'nDCG@3':>8} {'Recall@5':>10}  (n)")
    print(f"{'-'*col_w} {'-'*8} {'-'*8} {'-'*10}  ---")
    for method, m in metrics.items():
        print(
            f"{method:<{col_w}} {m['MRR@1']:>8.4f} {m['nDCG@3']:>8.4f} {m['Recall@5']:>10.4f}"
            f"  ({m['n_queries']})"
        )


def main():
    base = Path(__file__).resolve().parent.parent / "results"

    tables = {
        "Table 1 — Main Results  (bgl_full_run)":    base / "bgl_full_run",
        "Table 2 — Ablation Study (bgl_ablations)":  base / "bgl_ablations",
    }

    all_results = {}
    for title, results_dir in tables.items():
        if not results_dir.exists():
            print(f"⚠  Directory not found: {results_dir}")
            continue
        metrics = compute_for_dir(results_dir)
        print_table(title, metrics)
        all_results[title] = metrics

    print(f"\n{'='*70}")
    print("  Note: Recall@5 is cross-checked (already stored in results.json).")
    print("  Send MRR@1 and nDCG@3 values to co-author for Table 1 & Table 2.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
