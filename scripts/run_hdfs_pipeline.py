"""
HDFS Full-Scale Pipeline Orchestrator
======================================
End-to-end processing for the HDFS replication experiment (revision P1-A):

  Step 0: Verify raw HDFS data exists (HDFS_1.log + anomaly_label.csv)
  Step 1: Parse raw HDFS logs → data/processed/hdfs/parsed.jsonl
  Step 2: Extract templates   → data/processed/hdfs/templates.jsonl
                                 data/processed/hdfs/template_map.jsonl
  Step 3: Embed templates     → data/processed/hdfs/template_embeddings.jsonl/
  Step 4: Build occurrence store → data/processed/hdfs/occurrences.duckdb
  Step 5: Build FAISS index   → data/index/hdfs_faiss/
  Step 6: Build BM25 index    → data/processed/hdfs/bm25_index/
  Step 7: Build ground truth  → data/evaluation/hdfs_incidents.jsonl
                                 data/evaluation/hdfs_qrels.jsonl
  Step 8: Run experiments     → results/hdfs_full_run/results.json

Usage:
    python scripts/run_hdfs_pipeline.py
    python scripts/run_hdfs_pipeline.py --force-rebuild
    python scripts/run_hdfs_pipeline.py --skip-embed   # skip expensive API step
"""

import sys
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(description: str, cmd: list, check: bool = True) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP: {description}")
    logger.info(f"CMD : {' '.join(cmd)}")
    logger.info("=" * 70)
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    if ok:
        logger.info(f"OK  — {description}")
    else:
        logger.error(f"FAIL — {description} (exit {result.returncode})")
        if check:
            raise RuntimeError(f"Pipeline step failed: {description}")
    return ok


def _exists(*paths: str) -> bool:
    return all(Path(p).exists() for p in paths)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HDFSPipeline:
    RAW_LOG    = "data/raw/HDFS/HDFS_v1/HDFS.log"
    RAW_LABELS = "data/raw/HDFS/HDFS_v1/preprocessed/anomaly_label.csv"

    PARSED      = "data/processed/hdfs/parsed.jsonl"
    TEMPLATES   = "data/processed/hdfs/templates.jsonl"
    TMPL_MAP    = "data/processed/hdfs/template_map.jsonl"
    EMBED_MAN   = "data/processed/hdfs/template_embeddings.jsonl/manifest.json"
    OCC_DB      = "data/processed/hdfs/occurrences.duckdb"
    FAISS_IDX   = "data/index/hdfs_faiss/index.faiss"
    BM25_PKL    = "data/processed/hdfs/bm25_index/bm25_index.pkl"
    INCIDENTS   = "data/evaluation/hdfs_incidents.jsonl"
    QRELS       = "data/evaluation/hdfs_qrels.jsonl"
    GT_BASE     = "data/evaluation/hdfs_ground_truth.jsonl"

    def __init__(self, force_rebuild: bool = False, skip_embed: bool = False):
        self.force   = force_rebuild
        self.no_embed = skip_embed

    # ---- Step 0 ----
    def step0_check_raw_data(self):
        missing = [p for p in [self.RAW_LOG, self.RAW_LABELS] if not Path(p).exists()]
        if missing:
            logger.error("Missing raw HDFS data files:")
            for m in missing:
                logger.error(f"  {m}")
            logger.error(
                "\nDownload HDFS_v1.zip from Loghub and extract to data/raw/HDFS/:\n"
                "  https://zenodo.org/api/records/8196385/files/HDFS_v1.zip/content\n"
                "Expected: data/raw/HDFS/HDFS_v1/HDFS.log  +  data/raw/HDFS/HDFS_v1/preprocessed/anomaly_label.csv"
            )
            raise FileNotFoundError("Raw HDFS data not found")
        size_mb = Path(self.RAW_LOG).stat().st_size / (1024 * 1024)
        logger.info(f"Raw log: {size_mb:.0f} MB  — OK")
        logger.info(f"Anomaly labels: {self.RAW_LABELS}  — OK")

    # ---- Step 1 ----
    def step1_parse(self):
        if _exists(self.PARSED) and not self.force:
            logger.info(f"parsed.jsonl exists — skipping (--force-rebuild to redo)")
            return
        _run("Parse HDFS logs", [
            sys.executable, "src/data/parse_hdfs_local.py",
            "--input",  self.RAW_LOG,
            "--output", self.PARSED,
        ])

    # ---- Step 2 ----
    def step2_extract_templates(self):
        if _exists(self.TEMPLATES, self.TMPL_MAP) and not self.force:
            logger.info("templates.jsonl + template_map.jsonl exist — skipping")
            return
        _run("Extract HDFS templates", [
            sys.executable, "src/preprocess/template_hdfs.py",
            "--input",               self.PARSED,
            "--templates-output",    self.TEMPLATES,
            "--template-map-output", self.TMPL_MAP,
        ])

    # ---- Step 3 ----
    def step3_embed_templates(self):
        if self.no_embed:
            logger.info("--skip-embed: skipping embedding step")
            logger.warning(
                "Dense and LAR-RAG methods will fail without embeddings. "
                "Remove --skip-embed to generate them."
            )
            return
        if _exists(self.EMBED_MAN) and not self.force:
            logger.info("Embeddings manifest exists — skipping")
            return
        embed_dir = str(Path(self.EMBED_MAN).parent)
        _run("Embed HDFS templates (Azure OpenAI)", [
            sys.executable, "src/embeddings/embed_templates_safe_v2.py",
            "--templates", self.TEMPLATES,
            "--output", embed_dir,
            "--non-interactive",
            "--force",
        ])

    # ---- Step 4 ----
    def step4_build_occurrence_store(self):
        if _exists(self.OCC_DB) and not self.force:
            logger.info("occurrences.duckdb exists — skipping")
            return
        _run("Build HDFS occurrence store", [
            sys.executable, "scripts/build_occurrence_store.py",
            "--template-map", self.TMPL_MAP,
            "--output",       self.OCC_DB,
        ])

    # ---- Step 5 ----
    def step5_build_faiss_index(self):
        if _exists(self.FAISS_IDX) and not self.force:
            logger.info("FAISS index exists — skipping")
            return
        if not _exists(self.EMBED_MAN):
            logger.error("Embeddings not found — run step 3 first (or remove --skip-embed)")
            return
        embed_dir = str(Path(self.EMBED_MAN).parent)
        _run("Build HDFS FAISS index", [
            sys.executable, "scripts/build_faiss_index.py",
            "--embeddings-dir", embed_dir,
            "--out", str(Path(self.FAISS_IDX).parent),
            "--normalize",
        ])

    # ---- Step 6 ----
    def step6_build_bm25_index(self):
        if _exists(self.BM25_PKL) and not self.force:
            logger.info("BM25 index exists — skipping")
            return
        _run("Build HDFS BM25 index", [
            sys.executable, "scripts/build_bm25_index.py",
            "--templates", self.TEMPLATES,
            "--output",    str(Path(self.BM25_PKL).parent),
        ])

    # ---- Step 7 ----
    def step7_build_ground_truth(self):
        if _exists(self.INCIDENTS, self.QRELS) and not self.force:
            logger.info("Ground truth files exist — skipping")
            return
        _run("Build HDFS ground truth", [
            sys.executable, "src/evaluation/ground_truth_hdfs.py",
            "--parsed-logs",    self.PARSED,
            "--template-map",   self.TMPL_MAP,
            "--anomaly-labels", self.RAW_LABELS,
            "--output",         self.GT_BASE,
            "--seed",           "42",
            "--max-incidents",  "200",
            "--min-relevant",   "3",
        ])

    # ---- Step 8 ----
    def step8_run_experiments(self):
        _run("Run HDFS experiments", [
            sys.executable, "-m", "src.evaluation.run_experiments_v2",
            "--config", "config/hdfs_full_run.yaml",
        ])

    # ---- Orchestrate ----
    def run(self):
        t0 = datetime.now()
        logger.info("=" * 70)
        logger.info("HDFS PIPELINE — INFOCOM REVISION (P1-A)")
        logger.info("=" * 70)

        self.step0_check_raw_data()
        self.step1_parse()
        self.step2_extract_templates()
        self.step3_embed_templates()
        self.step4_build_occurrence_store()
        self.step5_build_faiss_index()
        self.step6_build_bm25_index()
        self.step7_build_ground_truth()
        self.step8_run_experiments()

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"\nPipeline complete in {elapsed:.0f}s")
        logger.info(f"Results: results/hdfs_full_run/results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDFS full evaluation pipeline")
    parser.add_argument("--config",        default="config/hdfs_full_run.yaml")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rerun all steps even if artifacts exist")
    parser.add_argument("--skip-embed",    action="store_true",
                        help="Skip embedding step (Azure OpenAI calls)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    HDFSPipeline(
        force_rebuild=args.force_rebuild,
        skip_embed=args.skip_embed,
    ).run()
