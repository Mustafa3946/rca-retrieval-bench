"""
Thunderbird Full-Scale Pipeline Orchestrator
=============================================
End-to-end processing for the Thunderbird replication experiment (revision P1-A):

  Step 0: Verify raw Thunderbird data exists on E: drive
  Step 1: Parse raw Thunderbird logs → E:/Paper/LAR-RAG/data/processed/thunderbird/parsed.jsonl
  Step 2: Extract templates          → data/processed/thunderbird/templates.jsonl
                                        E:/Paper/LAR-RAG/data/processed/thunderbird/template_map.jsonl
  Step 3: Embed templates            → data/processed/thunderbird/template_embeddings.jsonl/
  Step 4: Build occurrence store     → E:/Paper/LAR-RAG/data/processed/thunderbird/occurrences.duckdb
  Step 5: Build FAISS index          → data/index/thunderbird_faiss/
  Step 6: Build BM25 index           → data/processed/thunderbird/bm25_index/
  Step 7: Build ground truth         → data/evaluation/thunderbird_ground_truth_incidents.jsonl
                                        data/evaluation/thunderbird_ground_truth_qrels.jsonl
  Step 8: Run experiments            → results/thunderbird_full_run/results.json

Large artifacts (steps 1/2 template_map/4) live on E: drive to avoid C: space pressure.
Small artifacts (templates, embeddings, FAISS, BM25, results) stay in the project tree.

Usage:
    python scripts/run_thunderbird_pipeline.py
    python scripts/run_thunderbird_pipeline.py --force-rebuild
    python scripts/run_thunderbird_pipeline.py --skip-embed   # skip expensive API step
"""

import sys
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# E: drive paths  (large files)
# ---------------------------------------------------------------------------
E_BASE        = r"E:\Paper\LAR-RAG\data"
RAW_LOG       = rf"{E_BASE}\raw\Thunderbird\Thunderbird.log"
PARSED        = rf"{E_BASE}\processed\thunderbird\parsed.jsonl"
TMPL_MAP      = rf"{E_BASE}\processed\thunderbird\template_map.jsonl"
OCC_DB        = rf"{E_BASE}\processed\thunderbird\occurrences.duckdb"

# ---------------------------------------------------------------------------
# C: drive paths  (small files, project tree)
# ---------------------------------------------------------------------------
TEMPLATES     = "data/processed/thunderbird/templates.jsonl"
EMBED_DIR     = "data/processed/thunderbird/template_embeddings.jsonl"
EMBED_MAN     = "data/processed/thunderbird/template_embeddings.jsonl/manifest.json"
FAISS_IDX     = "data/index/thunderbird_faiss/index.faiss"
BM25_PKL      = "data/processed/thunderbird/bm25_index/bm25_index.pkl"
INCIDENTS     = "data/evaluation/thunderbird_ground_truth_incidents.jsonl"
QRELS         = "data/evaluation/thunderbird_ground_truth_qrels.jsonl"

# Lines to parse from the 211M-line file.
# First 30M lines contain the densest anomaly period (5% anomaly rate)
# and produce a manageable ~6 GB parsed.jsonl on E: drive.
MAX_PARSE_LINES = 30_000_000


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


def _ensure_e_dirs():
    """Create E: drive output directories if absent."""
    for d in [
        rf"{E_BASE}\processed\thunderbird",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ThunderbirdPipeline:

    def __init__(self, force_rebuild: bool = False, skip_embed: bool = False):
        self.force    = force_rebuild
        self.no_embed = skip_embed

    # ---- Step 0 ----
    def step0_check_raw_data(self):
        if not Path(RAW_LOG).exists():
            raise FileNotFoundError(
                f"Thunderbird.log not found at:\n  {RAW_LOG}\n"
                "Download from:\n"
                "  https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1\n"
                "and extract to E:\\Paper\\LAR-RAG\\data\\raw\\Thunderbird\\"
            )
        size_gb = Path(RAW_LOG).stat().st_size / (1024 ** 3)
        logger.info(f"Raw log: {size_gb:.1f} GB  ✓")
        _ensure_e_dirs()

    # ---- Step 1 ----
    def step1_parse(self):
        if _exists(PARSED) and not self.force:
            logger.info("parsed.jsonl exists on E: — skipping (--force-rebuild to redo)")
            return
        _run("Parse Thunderbird logs (first 30M lines)", [
            sys.executable, "src/data/parse_thunderbird_local.py",
            "--input",     RAW_LOG,
            "--output",    PARSED,
            "--max-lines", str(MAX_PARSE_LINES),
        ])

    # ---- Step 2 ----
    def step2_extract_templates(self):
        if _exists(TEMPLATES, TMPL_MAP) and not self.force:
            logger.info("templates.jsonl + template_map.jsonl exist — skipping")
            return
        _run("Extract Thunderbird templates", [
            sys.executable, "src/preprocess/template_thunderbird.py",
            "--input",               PARSED,
            "--templates-output",    TEMPLATES,
            "--template-map-output", TMPL_MAP,
        ])

    # ---- Step 3 ----
    def step3_embed_templates(self):
        if self.no_embed:
            logger.info("--skip-embed: skipping embedding step")
            return
        if _exists(EMBED_MAN) and not self.force:
            logger.info("Embeddings manifest exists — skipping")
            return
        _run("Embed Thunderbird templates (Azure OpenAI)", [
            sys.executable, "src/embeddings/embed_templates_safe_v2.py",
            "--templates", TEMPLATES,
            "--output",    EMBED_DIR,
            "--non-interactive",
            "--force",
        ])

    # ---- Step 4 ----
    def step4_build_occurrence_store(self):
        if _exists(OCC_DB) and not self.force:
            logger.info("occurrences.duckdb exists on E: — skipping")
            return
        _run("Build Thunderbird occurrence store (DuckDB)", [
            sys.executable, "scripts/build_occurrence_store.py",
            "--template-map", TMPL_MAP,
            "--output",       OCC_DB,
        ])

    # ---- Step 5 ----
    def step5_build_faiss_index(self):
        if _exists(FAISS_IDX) and not self.force:
            logger.info("FAISS index exists — skipping")
            return
        if not _exists(EMBED_MAN):
            logger.error("Embeddings not found — run step 3 first (or remove --skip-embed)")
            return
        _run("Build Thunderbird FAISS index", [
            sys.executable, "scripts/build_faiss_index.py",
            "--embeddings-dir", EMBED_DIR,
            "--out", str(Path(FAISS_IDX).parent),
            "--normalize",
        ])

    # ---- Step 6 ----
    def step6_build_bm25_index(self):
        if _exists(BM25_PKL) and not self.force:
            logger.info("BM25 index exists — skipping")
            return
        _run("Build Thunderbird BM25 index", [
            sys.executable, "scripts/build_bm25_index.py",
            "--templates", TEMPLATES,
            "--output",    str(Path(BM25_PKL).parent),
        ])

    # ---- Step 7 ----
    def step7_build_ground_truth(self):
        if _exists(INCIDENTS, QRELS) and not self.force:
            logger.info("Ground truth files exist — skipping")
            return
        _run("Build Thunderbird ground truth", [
            sys.executable, "src/evaluation/ground_truth_thunderbird.py",
            "--parsed-logs",    PARSED,
            "--occ-db",         OCC_DB,
            "--output",         "data/evaluation/thunderbird_ground_truth.jsonl",
            "--seed",           "42",
            "--max-incidents",  "200",
            "--min-relevant",   "3",
        ])

    # ---- Step 8 ----
    def step8_run_experiments(self):
        _run("Run Thunderbird experiments", [
            sys.executable, "-m", "src.evaluation.run_experiments_v2",
            "--config", "config/thunderbird_full_run.yaml",
        ])

    # ---- Orchestrate ----
    def run(self):
        t0 = datetime.now()
        logger.info("=" * 70)
        logger.info("THUNDERBIRD PIPELINE — INFOCOM REVISION (P1-A)")
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
        logger.info(f"\nPipeline complete in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
        logger.info("Results: results/thunderbird_full_run/results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thunderbird full evaluation pipeline")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rerun all steps even if artifacts exist")
    parser.add_argument("--skip-embed",    action="store_true",
                        help="Skip embedding step (Azure OpenAI calls)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    ThunderbirdPipeline(
        force_rebuild=args.force_rebuild,
        skip_embed=args.skip_embed,
    ).run()
