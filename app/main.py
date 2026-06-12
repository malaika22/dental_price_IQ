"""Pipeline orchestrator and FastAPI surface.

POST /orders/run   — upload a Henry Schein order PDF, get the three reports.
GET  /healthz
CLI: python -m app.main path/to/order.pdf  (or use run_pipeline.py)
"""
from __future__ import annotations
import logging
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import ai, db, matcher, parser, reports
from .models import EquivalencyFinding, ItemResult

_LOG_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _process_item_safe(item) -> ItemResult:
    """Run Stage 1 for one item; log and return empty result on failure."""
    try:
        return matcher.process_item(item)
    except Exception:
        log.exception(
            "[Stage 1] FAILED SKU %s — %s",
            item.schein_sku, item.description[:60],
        )
        return ItemResult(item=item)


def run_pipeline(pdf_path: str | Path, parallel: int = 4,
                 skip_search: bool = False) -> dict:
    """Full Stage 1+2 run for one order PDF. Returns paths of the 3 reports.

    skip_search=True runs intake + AI parsing + report scaffolding only
    (used for parser validation / dry runs without API spend).
    """
    pdf_path = Path(pdf_path)
    log.info("=== Pipeline start: %s ===", pdf_path.name)

    # Step 1 — PDF intake
    log.info("[Step 1/6] Parsing order PDF …")
    try:
        order = parser.parse_order_pdf(pdf_path)
    except Exception:
        log.exception("[Step 1/6] PDF parsing FAILED for %s", pdf_path)
        raise
    if not order.items:
        log.error("[Step 1/6] No line items extracted from %s", pdf_path)
        raise ValueError(f"No line items extracted from {pdf_path}")
    log.info(
        "[Step 1/6] Done — %d items from %s (ref=%s, total=$%.2f)",
        len(order.items), order.source_file, order.reference, order.computed_total,
    )

    results: List[ItemResult] = []
    findings: List[EquivalencyFinding] = []
    matched = 0

    if skip_search:
        log.info("[Step 2/6] Skipped — --skip-search (no Groq / search / scrape)")
        results = [ItemResult(item=i) for i in order.items]
    else:
        # Step 2 — Groq description parsing
        log.info("[Step 2/6] Groq batch parsing for %d items …", len(order.items))
        try:
            ai.parse_items_batch(order.items)
            log.info("[Step 2/6] Groq parsing complete")
        except Exception:
            log.exception("[Step 2/6] Groq batch parsing FAILED")
            raise

        # Step 3 — Stage 1: market sweep + scrape + match (parallel)
        log.info(
            "[Step 3/6] Stage 1 — price matching for %d items (%d workers) …",
            len(order.items), parallel,
        )
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(_process_item_safe, order.items))
        matched = sum(1 for r in results if r.best_exact)
        log.info(
            "[Step 3/6] Stage 1 done — %d/%d items have an exact match",
            matched, len(results),
        )

        # Step 4 — Stage 2: equivalency table
        log.info("[Step 4/6] Stage 2 — equivalency table lookup …")
        entries = matcher.load_equivalency_table(CONFIG_DIR)
        log.info("[Step 4/6] Loaded %d equivalency table entries", len(entries))
        for idx, r in enumerate(results, start=1):
            try:
                f = matcher.run_equivalency(r, entries)
                if f:
                    findings.append(f)
            except Exception:
                log.exception(
                    "[Step 4/6] Equivalency FAILED for SKU %s (%d/%d)",
                    r.item.schein_sku, idx, len(results),
                )
        log.info(
            "[Step 4/6] Stage 2 done — %d equivalency finding(s)",
            len(findings),
        )

    # Step 5 — reports
    stem = (order.reference or pdf_path.stem).replace("/", "_")
    log.info("[Step 5/6] Writing Excel reports (stem=%s) …", stem)
    try:
        p1 = reports.write_price_match_report(order, results, OUTPUT_DIR / f"{stem}_price_match.xlsx")
        p2 = reports.write_alternate_purchase_list(order, findings, OUTPUT_DIR / f"{stem}_alternate_purchases.xlsx")
        p3 = reports.write_evidence_file(order, results, findings, OUTPUT_DIR / f"{stem}_evidence.xlsx")
    except Exception:
        log.exception("[Step 5/6] Report generation FAILED")
        raise
    log.info("[Step 5/6] Reports written: %s, %s, %s", p1.name, p2.name, p3.name)

    # Step 6 — persistence
    log.info("[Step 6/6] Persisting run to SQLite …")
    try:
        conn = db.connect(OUTPUT_DIR / "dental_intel.sqlite3")
        order_id = db.persist_run(conn, order, results, findings)
        conn.close()
        log.info("[Step 6/6] Saved to database (order_id=%s)", order_id)
    except Exception:
        log.exception("[Step 6/6] Database persistence FAILED")
        raise

    log.info("=== Pipeline complete: %s — %d items, %d savings matches ===",
             order.reference or pdf_path.name, len(order.items), matched if not skip_search else 0)

    return {
        "reference": order.reference,
        "items": len(order.items),
        "total": order.total_price,
        "computed_total": order.computed_total,
        "price_match_report": str(p1),
        "alternate_purchase_list": str(p2),
        "evidence_file": str(p3),
    }


# ------------------------------------------------------------------ FastAPI

app = FastAPI(title="Dental Supply Price Intelligence", version="1.0")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/orders/run")
async def run_order(file: UploadFile):
    log.info("API upload received: %s", file.filename)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        summary = run_pipeline(tmp_path)
        log.info("API run complete for %s — %d items", file.filename, summary.get("items"))
        return JSONResponse(summary)
    except Exception as e:
        log.exception("API pipeline failed for %s", file.filename)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/reports/{name}")
def get_report(name: str):
    f = OUTPUT_DIR / name
    if not f.exists() or f.suffix != ".xlsx":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m app.main <order.pdf> [--skip-search]")
        sys.exit(1)
    out = run_pipeline(sys.argv[1], skip_search="--skip-search" in sys.argv)
    print(out)
