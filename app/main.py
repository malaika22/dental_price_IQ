"""Pipeline orchestrator and FastAPI surface.

POST /orders/run   — upload a Henry Schein order PDF, get the three reports.
GET  /healthz
CLI: python -m app.main path/to/order.pdf  (or use run_pipeline.py)
"""
from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import ai, db, matcher, parser, reports
from .models import EquivalencyFinding, ItemResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGIN",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]


def _unique_report_path(stem: str, kind: str) -> Path:
    """Unique xlsx path per run so prior outputs in output/ are never overwritten."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{stem}_{ts}_{kind}.xlsx"
    n = 1
    while path.exists():
        path = OUTPUT_DIR / f"{stem}_{ts}_{kind}_{n}.xlsx"
        n += 1
    return path


def _process_item_safe(item) -> ItemResult:
    """Isolate per-item failures so one bad SKU cannot 500 the whole run."""
    try:
        return matcher.process_item(item)
    except Exception:
        log.exception("Stage 1 failed for SKU %s", item.schein_sku)
        return ItemResult(item=item)


def run_pipeline(pdf_path: str | Path, parallel: int = 4,
                 skip_search: bool = False) -> dict:
    """Full Stage 1+2 run for one order PDF. Returns paths of the 3 reports.

    skip_search=True runs intake + AI parsing + report scaffolding only
    (used for parser validation / dry runs without API spend).
    """
    order = parser.parse_order_pdf(pdf_path)
    if not order.items:
        raise ValueError(f"No line items extracted from {pdf_path}")
    log.info("Parsed %d items from %s (ref %s)", len(order.items),
             order.source_file, order.reference)

    if not skip_search:
        ai.parse_items_batch(order.items)        # batched Groq calls (chunked)

    results: List[ItemResult] = []
    findings: List[EquivalencyFinding] = []

    if skip_search:
        results = [ItemResult(item=i) for i in order.items]
    else:
        # Stage 1 — market sweep + verification + 4-criteria matching,
        # parallel across items
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(_process_item_safe, order.items))

        # Stage 2 — equivalency, table-driven
        entries = matcher.load_equivalency_table(CONFIG_DIR)
        for r in results:
            try:
                f = matcher.run_equivalency(r, entries)
                if f:
                    findings.append(f)
            except Exception:
                log.exception("Stage 2 failed for SKU %s", r.item.schein_sku)

    stem = (order.reference or Path(pdf_path).stem).replace("/", "_")
    p1 = reports.write_price_match_report(order, results, _unique_report_path(stem, "price_match"))
    p2 = reports.write_alternate_purchase_list(order, findings, _unique_report_path(stem, "alternate_purchases"), results=results)
    p3 = reports.write_evidence_file(order, results, findings, _unique_report_path(stem, "evidence"))

    conn = db.connect(OUTPUT_DIR / "dental_intel.sqlite3")
    db.persist_run(conn, order, results, findings)
    conn.close()

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/orders/run")
async def run_order(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        summary = run_pipeline(tmp_path)
        return JSONResponse(summary)
    except Exception as e:
        log.exception("pipeline failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/reports/{name}")
def get_report(name: str):
    safe_name = Path(name).name
    if safe_name != name or ".." in name:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    f = OUTPUT_DIR / safe_name
    if not f.exists() or f.suffix != ".xlsx":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m app.main <order.pdf> [--skip-search]")
        sys.exit(1)
    out = run_pipeline(sys.argv[1], skip_search="--skip-search" in sys.argv)
    print(out)