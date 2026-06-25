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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

# On throttled hosts (e.g. Render free tier, 0.1 vCPU) Python's stdout is
# block-buffered when not a TTY, so log lines emitted right before a blocking
# network call (e.g. "scraping …") sit unflushed — making a slow scrape look
# like a total freeze with no output. Force line-buffering so every log line
# appears as it happens.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import ai, db, matcher, parser, reports
from . import search as search_mod
from .models import EquivalencyFinding, ItemResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_pipeline(pdf_path: str | Path, parallel: Optional[int] = None,
                 skip_search: bool = False) -> dict:
    """Full Stage 1+2 run for one order PDF. Returns paths of the 3 reports.

    skip_search=True runs intake + AI parsing + report scaffolding only
    (used for parser validation / dry runs without API spend).
    """
    # Resolve at call time (not at import via a default-arg expression) so a
    # PIPELINE_PARALLEL set after this module is imported still takes effect.
    if parallel is None:
        parallel = int(os.environ.get("PIPELINE_PARALLEL", "1"))
    log.info("Pipeline starting — pdf=%s skip_search=%s parallel=%d",
             pdf_path, skip_search, parallel)
    try:
        order = parser.parse_order_pdf(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}") from e
    search_mod.reset_firecrawl_budget()
    if not order.items:
        raise ValueError(f"No line items extracted from {pdf_path}")

    # open DB up front so the per-SKU discovery cache can be loaded before the
    # sweep (skips re-discovery of repeat items across runs)
    conn = db.connect(OUTPUT_DIR / "dental_intel.sqlite3")
    db.set_scrape_db_path(OUTPUT_DIR / "dental_intel.sqlite3")  # scrape cache uses same DB
    if not skip_search:
        search_mod.load_discovery_cache(conn)
    log.info("Parsed %d items from %s (ref %s)", len(order.items),
             order.source_file, order.reference)

    if not skip_search:
        try:
            ai.parse_items_batch(order.items)
            log.info("Groq parse complete — %d item(s)", len(order.items))
        except Exception:
            log.exception("Groq parse failed — items will use raw descriptions as search queries")
        # SKU→MPN enrichment from the persistent learned store (seeded from
        # config/mpn_table.txt, validated + grown from page consensus). Anchors
        # discovery on the exact part number; only page-VERIFIED MPNs are used to
        # reject conflicting listings (a seed is a discovery hint until confirmed).
        try:
            nmpn = matcher.seed_and_apply_mpn(conn, order.items, CONFIG_DIR)
            if nmpn:
                log.info("MPN store — applied %d item(s) (verified MPNs reject conflicts; "
                         "seeds are discovery hints until page-confirmed)", nmpn)
        except Exception:
            log.exception("MPN store apply failed — continuing without MPN enrichment")

    results: List[ItemResult] = []
    findings: List[EquivalencyFinding] = []

    if skip_search:
        results = [ItemResult(item=i) for i in order.items]
    else:
        # Stage 1 — market sweep + verification + 4-criteria matching,
        # parallel across items. Each item gets a hard wall-clock cap that
        # ACTUALLY fires: we wait on each future with its own timeout instead of
        # as_completed (which never yields a future that never completes, so the
        # old cap could not catch a truly-stuck item). A stuck scrape thread is
        # abandoned and the run continues.
        from concurrent.futures import TimeoutError as _FTimeout
        import time as _time
        item_budget = int(os.environ.get("ITEM_TIMEOUT_SEC", "180"))
        log.info("Stage 1 — processing %d item(s) with %d worker(s), %ss cap per item",
                 len(order.items), parallel, item_budget)
        results_by_idx: dict[int, ItemResult] = {}

        # SALVAGE: the worker stores its finished ItemResult into results_by_idx
        # itself (dict assignment is atomic under the GIL). So if an item blows the
        # per-item cap below, we do NOT immediately overwrite it with an empty
        # result — the worker keeps running and, when it finishes, its real result
        # (already populated with the deterministically-priced candidates) lands in
        # the dict and is picked up at the end. An item only ends up empty if its
        # worker truly never completes. This is what stops a slow-Groq item from
        # being discarded with all its scraped pages and prices.
        def _work(n, it):
            r = matcher.process_item(it)
            results_by_idx[n] = r
            return r

        # NOTE: deliberately NOT a `with ThreadPoolExecutor(...) as ex:` block.
        # The context-manager exit calls shutdown(wait=True), which blocks on any
        # worker still running — re-introducing the exact freeze the per-item
        # wall-clock cap exists to prevent. We shut down with wait=False in the
        # finally so the run returns promptly even if a worker is wedged; leftover
        # threads finish or die in the background (the inner scrape's own hard
        # deadline bounds their lifetime).
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
                    log.error("Stage 1 — SKU %s exceeded %ss wall-clock cap; leaving it "
                              "to finish in the background — its result is salvaged if "
                              "the worker completes before reports are written", sku, item_budget)
                except Exception:
                    log.exception("Stage 1 — SKU %s crashed; continuing", sku)
                    results_by_idx.setdefault(n, ItemResult(item=order.items[n]))
        finally:
            ex.shutdown(wait=False)
        # Fill any still-missing slots (workers that never completed) with an empty
        # result so the report still lists the item.
        results = [results_by_idx.get(n) or ItemResult(item=order.items[n])
                   for n in range(len(order.items))]

        # Stage 2 — equivalency, table-driven
        entries = matcher.load_equivalency_table(CONFIG_DIR)
        if entries:
            log.info("Stage 2 — equivalency table has %d entr(ies)", len(entries))
        for r in results:
            try:
                f = matcher.run_equivalency(r, entries)
                if f:
                    findings.append(f)
                    log.info("Stage 2 — SKU %s equivalency: %s",
                             r.item.schein_sku, f.confidence_level)
            except Exception:
                log.exception("Stage 2 failed for SKU %s", r.item.schein_sku)

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
    log.info("POST /orders/run — upload filename=%s", file.filename)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        summary = run_pipeline(tmp_path)
        log.info("POST /orders/run — success ref=%s", summary.get("reference"))
        return JSONResponse(summary)
    except ValueError as e:
        log.warning("POST /orders/run — bad request: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        log.exception("pipeline failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/reports/{name}")
def get_report(name: str):
    f = OUTPUT_DIR / Path(name).name
    if not f.exists() or f.suffix != ".xlsx":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f)


if __name__ == "__main__":
    argv = sys.argv[1:]
    # One-shot discovery-cache purge flags (translate to the env hook used by
    # load_discovery_cache, so the same path works for CLI and FastAPI):
    #   --purge-discovery-all          wipe the whole discovery cache
    #   --purge-discovery=SKU[,SKU]    clear specific SKUs (re-discovered fresh)
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