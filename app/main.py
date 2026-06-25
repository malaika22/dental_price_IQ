"""Pipeline orchestrator and FastAPI surface.

POST /orders/run        — upload PDF, returns job_id (202), runs pipeline in background
GET  /orders/{id}/events — SSE live progress (Groq, Firecrawl, SerpAPI, stages)
GET  /orders/{id}        — job status + result when complete
GET  /healthz
GET  /reports/{name}
CLI: python -m app.main path/to/order.pdf
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import ai, db, jobs, matcher, parser, reports
from . import search as search_mod
from .models import EquivalencyFinding, ItemResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_FILE = OUTPUT_DIR / "dental_intel.sqlite3"
db.set_db_path(DB_FILE)


def run_pipeline(pdf_path: str | Path, parallel: Optional[int] = None,
                 skip_search: bool = False, job_id: Optional[str] = None) -> dict:
    """Full Stage 1+2 run for one order PDF. Returns paths of the 3 reports."""
    if job_id:
        jobs.bind_job(job_id)

    if parallel is None:
        parallel = int(os.environ.get("PIPELINE_PARALLEL", "1"))

    jobs.emit("pipeline_start", pdf=str(pdf_path), skip_search=skip_search)
    log.info("Pipeline starting — pdf=%s skip_search=%s parallel=%d",
             pdf_path, skip_search, parallel)

    jobs.emit("step_start", step="parse", label="Parsing order PDF")
    try:
        order = parser.parse_order_pdf(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}") from e
    search_mod.reset_firecrawl_budget()
    if not order.items:
        raise ValueError(f"No line items extracted from {pdf_path}")

    jobs.emit("step_complete", step="parse",
              reference=order.reference, items=len(order.items),
              total=order.total_price, message=f"Extracted {len(order.items)} line items")

    conn = db.connect(DB_FILE)
    db.set_scrape_db_path(DB_FILE)
    if not skip_search:
        search_mod.load_discovery_cache(conn)
    log.info("Parsed %d items from %s (ref %s)", len(order.items),
             order.source_file, order.reference)

    if not skip_search:
        jobs.emit("step_start", step="ai",
                  label="AI product enrichment",
                  provider=os.environ.get("LLM_PROVIDER", "groq"))
        try:
            ai.parse_items_batch(order.items)
            log.info("Groq parse complete — %d item(s)", len(order.items))
            jobs.emit("step_complete", step="ai", items=len(order.items),
                      message=f"AI enriched {len(order.items)} products")
        except Exception:
            log.exception("Groq parse failed — items will use raw descriptions as search queries")
            jobs.emit("step_warning", step="ai",
                      message="AI parse failed — using raw descriptions")

        jobs.emit("step_start", step="mpn", label="MPN lookup")
        try:
            nmpn = matcher.seed_and_apply_mpn(conn, order.items, CONFIG_DIR)
            if nmpn:
                log.info("MPN store — applied %d item(s)", nmpn)
            jobs.emit("step_complete", step="mpn", enriched=nmpn or 0,
                      message=f"MPN applied to {nmpn or 0} item(s)")
        except Exception:
            log.exception("MPN store apply failed — continuing without MPN enrichment")
            jobs.emit("step_warning", step="mpn", message="MPN lookup failed — continuing")

    results: List[ItemResult] = []
    findings: List[EquivalencyFinding] = []

    if skip_search:
        results = [ItemResult(item=i) for i in order.items]
    else:
        from concurrent.futures import TimeoutError as _FTimeout
        item_budget = int(os.environ.get("ITEM_TIMEOUT_SEC", "180"))
        jobs.emit("step_start", step="stage1",
                  label="Market price search & verification",
                  items_total=len(order.items), workers=parallel,
                  message=f"Processing {len(order.items)} items")
        log.info("Stage 1 — processing %d item(s) with %d worker(s), %ss cap per item",
                 len(order.items), parallel, item_budget)
        results_by_idx: dict[int, ItemResult] = {}
        items_total = len(order.items)

        def _work(n, it):
            jobs.bind_job(job_id)
            jobs.emit("item_start", index=n + 1, total=items_total,
                      sku=it.schein_sku,
                      description=(it.description or "")[:100])
            r = matcher.process_item(it)
            results_by_idx[n] = r
            best = None
            if r.best_exact and r.best_exact.price:
                best = f"${r.best_exact.price:.2f} @ {r.best_exact.source_site}"
            jobs.emit("item_complete", index=n + 1, total=items_total,
                      sku=it.schein_sku, candidates=len(r.candidates), best_exact=best)
            return r

        ex = ThreadPoolExecutor(max_workers=max(parallel, 2))
        try:
            futures = {n: ex.submit(_work, n, it)
                       for n, it in enumerate(order.items)}
            for n in range(len(order.items)):
                sku = order.items[n].schein_sku
                try:
                    futures[n].result(timeout=item_budget)
                    r = results_by_idx.get(n)
                    log.info("Stage 1 done — SKU %s: %d candidate(s), best_exact=%s",
                             sku, len(r.candidates) if r else 0,
                             f"${r.best_exact.price} @ {r.best_exact.source_site}"
                             if r and r.best_exact and r.best_exact.price else "none")
                except _FTimeout:
                    log.error("Stage 1 — SKU %s exceeded %ss wall-clock cap", sku, item_budget)
                    jobs.emit("item_timeout", sku=sku, seconds=item_budget)
                except Exception:
                    log.exception("Stage 1 — SKU %s crashed; continuing", sku)
                    results_by_idx.setdefault(n, ItemResult(item=order.items[n]))
        finally:
            ex.shutdown(wait=False)
        results = [results_by_idx.get(n) or ItemResult(item=order.items[n])
                   for n in range(len(order.items))]
        jobs.emit("step_complete", step="stage1", items=len(order.items),
                  message=f"Price search complete for {len(order.items)} items")

        entries = matcher.load_equivalency_table(CONFIG_DIR)
        jobs.emit("step_start", step="stage2", label="Equivalency analysis")
        if entries:
            log.info("Stage 2 — equivalency table has %d entr(ies)", len(entries))
        for r in results:
            try:
                f = matcher.run_equivalency(r, entries)
                if f:
                    findings.append(f)
                    log.info("Stage 2 — SKU %s equivalency: %s",
                             r.item.schein_sku, f.confidence_level)
                    jobs.emit("equivalency_found", sku=r.item.schein_sku,
                              level=f.confidence_level)
            except Exception:
                log.exception("Stage 2 failed for SKU %s", r.item.schein_sku)
        jobs.emit("step_complete", step="stage2", findings=len(findings),
                  message=f"Found {len(findings)} equivalency match(es)")

    jobs.emit("step_start", step="reports", label="Generating reports")
    log.info("Writing reports for ref=%s", order.reference)
    stem = (order.reference or Path(pdf_path).stem).replace("/", "_")
    p1 = reports.write_price_match_report(order, results, OUTPUT_DIR / f"{stem}_price_match.xlsx")
    p2 = reports.write_alternate_purchase_list(order, findings, OUTPUT_DIR / f"{stem}_alternate_purchases.xlsx", results=results)
    p3 = reports.write_evidence_file(order, results, findings, OUTPUT_DIR / f"{stem}_evidence.xlsx")

    if not skip_search:
        search_mod.flush_discovery_cache(conn)
    db.persist_run(conn, order, results, findings)
    conn.close()

    if not skip_search:
        search_mod.firecrawl_log_run_summary(order.reference)
        fc = search_mod.firecrawl_stats()
        log.info(
            "=== Pipeline complete: ref=%s items=%d firecrawl_scrapes=%d "
            "skipped=%d credits=%d exhausted=%s ===",
            order.reference, len(order.items), fc["scrapes"], fc["skipped"],
            fc["credits"], fc["exhausted"],
        )
        jobs.emit("firecrawl_summary", **fc)

    summary = {
        "reference": order.reference,
        "items": len(order.items),
        "total": order.total_price,
        "computed_total": order.computed_total,
        "price_match_report": str(p1),
        "alternate_purchase_list": str(p2),
        "evidence_file": str(p3),
    }
    jobs.emit("step_complete", step="reports", message="Reports ready for download")
    return summary


# ------------------------------------------------------------------ FastAPI

app = FastAPI(title="Dental Supply Price Intelligence", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _run_job_in_thread(job_id: str, tmp_path: str, filename: str) -> None:
    import time as _time
    started = _time.time()
    try:
        jobs.bind_job(job_id)
        jobs.set_status(job_id, "running")
        summary = run_pipeline(tmp_path, job_id=job_id)
        duration_ms = int((_time.time() - started) * 1000)
        jobs.complete_job(job_id, summary)
        db.update_order_run(
            job_id,
            reference=summary.get("reference"),
            status="completed",
            items=summary.get("items", 0),
            total_price=summary.get("total") or summary.get("computed_total"),
            duration_ms=duration_ms,
            price_match_report=summary.get("price_match_report"),
            alternate_purchase_list=summary.get("alternate_purchase_list"),
            evidence_file=summary.get("evidence_file"),
            completed_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        )
        log.info("Job %s — success ref=%s", job_id, summary.get("reference"))
    except ValueError as e:
        log.warning("Job %s — bad request: %s", job_id, e)
        jobs.fail_job(job_id, str(e))
        db.update_order_run(
            job_id,
            status="failed",
            error=str(e),
            duration_ms=int((_time.time() - started) * 1000),
            completed_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        )
    except Exception as e:
        log.exception("Job %s — pipeline failed", job_id)
        jobs.fail_job(job_id, str(e))
        db.update_order_run(
            job_id,
            status="failed",
            error=str(e),
            duration_ms=int((_time.time() - started) * 1000),
            completed_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/orders/run")
async def run_order(file: UploadFile):
    log.info("POST /orders/run — upload filename=%s", file.filename)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    job_id = jobs.create_job(file.filename or "order.pdf")
    fname = file.filename or "order.pdf"
    db.create_order_run(job_id, filename=fname)
    jobs.bind_job(job_id)
    jobs.emit("upload_complete", filename=fname, job_id=job_id)
    threading.Thread(target=_run_job_in_thread, args=(job_id, tmp_path, fname), daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/orders/history")
def order_history():
    db.backfill_order_runs_from_output(OUTPUT_DIR)
    runs = db.list_order_runs()
    out = []
    for r in runs:
        entry = {
            "id": r["id"],
            "fileName": r["filename"],
            "reference": r["reference"] or r["filename"].replace(".pdf", ""),
            "items": r["items"] or 0,
            "total": r["total_price"],
            "status": r["status"],
            "createdAt": r["created_at"],
            "completedAt": r["completed_at"],
            "durationMs": r["duration_ms"],
            "error": r["error"],
        }
        if r["status"] == "completed" and r["price_match_report"]:
            entry["reports"] = {
                "price_match_report": r["price_match_report"],
                "alternate_purchase_list": r["alternate_purchase_list"],
                "evidence_file": r["evidence_file"],
            }
        out.append(entry)
    return out


@app.get("/orders/{job_id}")
def get_order_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "job_id": job_id,
        "status": job.status,
        "filename": job.filename,
        "result": job.result,
        "error": job.error,
    }


@app.get("/orders/{job_id}/events")
async def order_events(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)

    async def stream():
        sent_done = False
        while not sent_done:
            while not job.events.empty():
                evt = job.events.get_nowait()
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("event") in ("pipeline_complete", "pipeline_error"):
                    sent_done = True
                    break
            if sent_done:
                break
            if job.status in ("complete", "failed") and job.events.empty():
                yield f"data: {json.dumps({'event': 'stream_end', 'data': {'status': job.status}})}\n\n"
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/reports/{name}")
def get_report(name: str):
    f = OUTPUT_DIR / Path(name).name
    if not f.exists() or f.suffix != ".xlsx":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f)


if __name__ == "__main__":
    argv = sys.argv[1:]
    for a in argv:
        if a == "--purge-discovery-all":
            os.environ["DISCOVERY_PURGE_ALL"] = "1"
        elif a.startswith("--purge-discovery="):
            os.environ["DISCOVERY_PURGE"] = a.split("=", 1)[1]
    pdf = next((a for a in argv if not a.startswith("--")), None)
    if not pdf:
        print("usage: python -m app.main <order.pdf> [--skip-search] "
              "[--purge-discovery=SKU,SKU | --purge-discovery-all]")
        sys.exit(1)
    out = run_pipeline(pdf, skip_search="--skip-search" in argv)
    print(out)
