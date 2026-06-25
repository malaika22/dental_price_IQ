"""SQLite persistence. Schema covers Stages 1+2 (active) and Stages 3+4
(tables present, empty — PRD Section 10)."""
from __future__ import annotations
import json
import os
import sqlite3
from pathlib import Path

_DB_PATH: Path | None = None


def set_db_path(path: str | Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)


def resolve_db_path() -> Path:
    if _DB_PATH:
        return _DB_PATH
    return Path(os.environ.get("DENTAL_DB_PATH", "dental_intel.sqlite3"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    reference TEXT UNIQUE,
    order_date TEXT,
    source_file TEXT,
    total_price REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    qty INTEGER, schein_sku TEXT, description TEXT, uom TEXT,
    unit_price REAL, extended_price REAL,
    brand TEXT, product_name TEXT, size_form TEXT,
    pack_qty INTEGER, variant TEXT, mpn TEXT
);
CREATE TABLE IF NOT EXISTS price_findings (
    id INTEGER PRIMARY KEY,
    order_item_id INTEGER REFERENCES order_items(id),
    title TEXT, url TEXT, source_site TEXT,
    price REAL, pack_qty INTEGER, pack_condition TEXT,
    match_type TEXT, confidence INTEGER,
    criteria_json TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS equivalency_findings (
    id INTEGER PRIMARY KEY,
    order_item_id INTEGER REFERENCES order_items(id),
    equivalent_name TEXT, confidence_level TEXT, basis TEXT,
    supplier TEXT, url TEXT, price REAL, est_savings_total REAL
);
-- Per-SKU discovery cache (optimization): remembers which candidate URLs were
-- found for a SKU so repeat orders skip re-discovery. Prices are NOT cached
-- here — pages are always re-scraped fresh — so report prices never go stale
-- from this cache; only discovery API calls are saved.
CREATE TABLE IF NOT EXISTS discovery_cache (
    schein_sku TEXT PRIMARY KEY,
    urls_json TEXT,
    discovered_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Scrape cache: stores the scraped markdown per URL so a re-run (after a
-- freeze/crash, or a repeat order) skips re-scraping pages already fetched —
-- the single biggest Firecrawl credit saver. Short TTL keeps prices fresh.
CREATE TABLE IF NOT EXISTS scrape_cache (
    url TEXT PRIMARY KEY,
    markdown TEXT,
    scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Extraction cache: stores the Groq extract+validate JSON per (url + ordered
-- variant + pack) so a re-run of the SAME order skips the LLM for unchanged
-- pages. Scrapes are already cached (0 Firecrawl credits); this saves the
-- scarce Groq free-tier daily token budget on every debug re-run. Short TTL.
CREATE TABLE IF NOT EXISTS extract_cache (
    key TEXT PRIMARY KEY,
    payload TEXT,
    extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Learned MPN store: per Henry Schein SKU, the manufacturer part number.
-- Seeded from config/mpn_table.txt (source='manual'), then dynamically filled
-- and validated from the part numbers the real supplier pages declare
-- (source='page-consensus'). status: 'seed' (unverified) | 'verified' |
-- 'conflict' (seed disagrees with the pages → not trusted for rejection).
CREATE TABLE IF NOT EXISTS mpn_store (
    schein_sku TEXT PRIMARY KEY,
    mpn TEXT,
    manufacturer TEXT,
    source TEXT,
    status TEXT,
    page_mpn TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Stage 3 (order generation) — schema reserved, unused in Stage 1+2
CREATE TABLE IF NOT EXISTS order_history (
    id INTEGER PRIMARY KEY,
    schein_sku TEXT, order_date TEXT, qty INTEGER, unit_price REAL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS par_levels (
    schein_sku TEXT PRIMARY KEY, par_qty INTEGER, notes TEXT
);
-- Stage 4 (inventory intelligence) — schema reserved, unused in Stage 1+2
CREATE TABLE IF NOT EXISTS consumption_model (
    schein_sku TEXT PRIMARY KEY,
    weekly_rate REAL, last_computed TEXT, model_json TEXT
);
CREATE TABLE IF NOT EXISTS reorder_projections (
    id INTEGER PRIMARY KEY,
    schein_sku TEXT, projected_date TEXT, projected_qty INTEGER, basis TEXT
);
-- UI order history (persists across browser clears)
CREATE TABLE IF NOT EXISTS order_runs (
    id TEXT PRIMARY KEY,
    reference TEXT,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    items INTEGER DEFAULT 0,
    total_price REAL,
    error TEXT,
    duration_ms INTEGER,
    price_match_report TEXT,
    alternate_purchase_list TEXT,
    evidence_file TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);
"""


def connect(db_path: str | Path = "dental_intel.sqlite3") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def persist_run(conn, order, results, findings) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO orders(reference, order_date, source_file, total_price)"
        " VALUES (?,?,?,?)",
        (order.reference, order.order_date, order.source_file, order.total_price))
    order_id = cur.lastrowid
    for r in results:
        i = r.item
        cur.execute(
            "INSERT INTO order_items(order_id, qty, schein_sku, description, uom,"
            " unit_price, extended_price, brand, product_name, size_form, pack_qty,"
            " variant, mpn) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (order_id, i.qty, i.schein_sku, i.description, i.uom, i.unit_price,
             i.extended_price, i.brand, i.product_name, i.size_form, i.pack_qty,
             i.variant, i.mpn))
        item_id = cur.lastrowid
        for c in r.candidates:
            cur.execute(
                "INSERT INTO price_findings(order_item_id, title, url, source_site,"
                " price, pack_qty, pack_condition, match_type, confidence,"
                " criteria_json, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, c.title, c.url, c.source_site, c.price, c.pack_qty,
                 c.pack_condition, c.match_type, c.confidence,
                 json.dumps(c.criteria), c.rejected_reason or c.notes))
    sku_to_item = {}
    cur.execute("SELECT id, schein_sku FROM order_items WHERE order_id=?", (order_id,))
    for row in cur.fetchall():
        sku_to_item[row[1]] = row[0]
    for f in findings:
        cur.execute(
            "INSERT INTO equivalency_findings(order_item_id, equivalent_name,"
            " confidence_level, basis, supplier, url, price, est_savings_total)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (sku_to_item.get(f.item.schein_sku), f.equivalent_name,
             f.confidence_level, f.basis, f.supplier, f.url, f.price,
             f.est_savings_total))
    conn.commit()
    return order_id


# ------------------------------------------------------- order runs (UI) ----

def _runs_conn() -> sqlite3.Connection:
    return connect(resolve_db_path())


def create_order_run(run_id: str, filename: str, reference: str | None = None) -> None:
    conn = _runs_conn()
    try:
        conn.execute(
            "INSERT INTO order_runs(id, reference, filename, status) VALUES (?,?,?,?)",
            (run_id, reference, filename, "processing"),
        )
        conn.commit()
    finally:
        conn.close()


def update_order_run(run_id: str, **fields) -> None:
    allowed = {
        "reference", "filename", "status", "items", "total_price", "error",
        "duration_ms", "price_match_report", "alternate_purchase_list",
        "evidence_file", "completed_at",
    }
    cols = {k: v for k, v in fields.items() if k in allowed}
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols)
    conn = _runs_conn()
    try:
        conn.execute(
            f"UPDATE order_runs SET {sets} WHERE id=?",
            (*cols.values(), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def backfill_order_runs_from_output(output_dir: Path) -> None:
    """Register legacy completed runs from xlsx files already on disk."""
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    conn = _runs_conn()
    try:
        for p in output_dir.glob("*_price_match.xlsx"):
            stem = p.stem[: -len("_price_match")]
            run_id = f"legacy-{stem}"
            if conn.execute("SELECT 1 FROM order_runs WHERE id=?", (run_id,)).fetchone():
                continue
            alt = output_dir / f"{stem}_alternate_purchases.xlsx"
            ev = output_dir / f"{stem}_evidence.xlsx"
            row = conn.execute(
                "SELECT o.id, o.reference, o.source_file, o.total_price, o.created_at"
                " FROM orders o WHERE o.reference=? ORDER BY o.id DESC LIMIT 1",
                (stem,),
            ).fetchone()
            items = 0
            ref, src, total, created = stem, f"{stem}.pdf", None, None
            if row:
                oid, ref, src, total, created = row
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM order_items WHERE order_id=?", (oid,)
                ).fetchone()
                items = cnt[0] if cnt else 0
            if not created:
                created = datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute(
                "INSERT INTO order_runs(id, reference, filename, status, items,"
                " total_price, price_match_report, alternate_purchase_list,"
                " evidence_file, created_at, completed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, ref, Path(src).name if src else f"{stem}.pdf",
                    "completed", items, total,
                    str(p), str(alt) if alt.exists() else None,
                    str(ev) if ev.exists() else None,
                    created, created,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def list_order_runs(limit: int = 200) -> list[dict]:
    conn = _runs_conn()
    try:
        cur = conn.execute(
            "SELECT id, reference, filename, status, items, total_price, error,"
            " duration_ms, price_match_report, alternate_purchase_list,"
            " evidence_file, created_at, completed_at"
            " FROM order_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        keys = [
            "id", "reference", "filename", "status", "items", "total_price",
            "error", "duration_ms", "price_match_report", "alternate_purchase_list",
            "evidence_file", "created_at", "completed_at",
        ]
        return [dict(zip(keys, row)) for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------- discovery cache -------

def get_cached_discovery(conn, schein_sku: str, max_age_days: int = 7) -> list | None:
    """Return cached candidate URLs for a SKU if discovered within max_age_days,
    else None. Used to skip re-discovery of repeat items across runs."""
    cur = conn.cursor()
    cur.execute(
        "SELECT urls_json FROM discovery_cache WHERE schein_sku=? AND "
        "discovered_at >= datetime('now', ?)",
        (schein_sku, f"-{int(max_age_days)} days"))
    row = cur.fetchone()
    if not row:
        return None
    try:
        urls = json.loads(row[0])
        return urls if isinstance(urls, list) and urls else None
    except Exception:
        return None


def save_discovery(conn, schein_sku: str, urls: list) -> None:
    """Persist the discovered candidate URLs for a SKU (refreshes timestamp)."""
    if not urls:
        return
    conn.execute(
        "INSERT OR REPLACE INTO discovery_cache(schein_sku, urls_json, discovered_at)"
        " VALUES (?,?,datetime('now'))",
        (schein_sku, json.dumps(list(urls)[:40])))
    conn.commit()


def purge_discovery(conn, skus=None) -> tuple[int, int]:
    """Drop cached discovery for the given SKU(s) AND the scrape cache for the
    URLs those SKUs had cached, so the next run re-discovers + re-scrapes them
    fresh. Used to clear a wrong-variant URL that got pinned in the cache (e.g. a
    net32 'Watermelon' URL cached for a 'Flavorless' order). skus=None purges ALL
    discovery. Returns (discovery_rows_deleted, scrape_rows_deleted)."""
    cur = conn.cursor()
    if skus is not None:
        skus = [s.strip() for s in skus if s and str(s).strip()]
        if not skus:
            return (0, 0)
        marks = ",".join("?" * len(skus))
        cur.execute(f"SELECT urls_json FROM discovery_cache WHERE schein_sku IN ({marks})", skus)
    else:
        cur.execute("SELECT urls_json FROM discovery_cache")
    urls: list = []
    for (uj,) in cur.fetchall():
        try:
            urls.extend(json.loads(uj) or [])
        except Exception:
            pass
    n_scrape = 0
    for u in urls:
        try:
            cur.execute("DELETE FROM scrape_cache WHERE url=?", (u,))
            if cur.rowcount and cur.rowcount > 0:
                n_scrape += cur.rowcount
        except Exception:
            pass
    if skus is not None:
        cur.execute(f"DELETE FROM discovery_cache WHERE schein_sku IN ({marks})", skus)
    else:
        cur.execute("DELETE FROM discovery_cache")
    n_disc = cur.rowcount if (cur.rowcount and cur.rowcount > 0) else 0
    conn.commit()
    return (n_disc, n_scrape)


# ------------------------------------------------------- scrape cache -------
# Thread-safe across the pipeline's worker threads: each call opens its own
# short-lived SQLite connection (SQLite handles concurrent writers via its
# file lock; write volume here is low). This is what makes a re-run after a
# freeze cheap — already-scraped pages are read from disk, not re-fetched.

import threading as _threading
_scrape_lock = _threading.Lock()
_SCRAPE_DB_PATH = {"p": "dental_intel.sqlite3"}


def set_scrape_db_path(path) -> None:
    _SCRAPE_DB_PATH["p"] = str(path)


def get_cached_scrape(url: str, max_age_hours: int = 24) -> str | None:
    """Return cached markdown for a URL if scraped within max_age_hours, else
    None. Lets a re-run skip re-scraping pages already fetched this/last run."""
    if not url:
        return None
    try:
        with _scrape_lock:
            c = sqlite3.connect(_SCRAPE_DB_PATH["p"], timeout=10)
            try:
                cur = c.execute(
                    "SELECT markdown FROM scrape_cache WHERE url=? AND "
                    "scraped_at >= datetime('now', ?)",
                    (url, f"-{int(max_age_hours)} hours"))
                row = cur.fetchone()
            finally:
                c.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return None


def save_scrape(url: str, markdown: str) -> None:
    """Persist scraped markdown for a URL (refreshes timestamp). Called as soon
    as each scrape succeeds, so a freeze/crash never loses fetched pages."""
    if not url or not markdown:
        return
    try:
        with _scrape_lock:
            c = sqlite3.connect(_SCRAPE_DB_PATH["p"], timeout=10)
            try:
                c.execute(
                    "INSERT OR REPLACE INTO scrape_cache(url, markdown, scraped_at)"
                    " VALUES (?,?,datetime('now'))", (url, markdown))
                c.commit()
            finally:
                c.close()
    except Exception:
        pass


def get_cached_extract(key: str, max_age_hours: int = 24) -> dict | None:
    """Return the cached Groq extract+validate JSON for `key` if stored within
    max_age_hours, else None. Lets a re-run of the same order skip the LLM for
    pages whose scrape + ordered variant/pack are unchanged."""
    if not key:
        return None
    try:
        with _scrape_lock:
            c = sqlite3.connect(_SCRAPE_DB_PATH["p"], timeout=10)
            try:
                cur = c.execute(
                    "SELECT payload FROM extract_cache WHERE key=? AND "
                    "extracted_at >= datetime('now', ?)",
                    (key, f"-{int(max_age_hours)} hours"))
                row = cur.fetchone()
            finally:
                c.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception:
        pass
    return None


def save_extract(key: str, payload: dict) -> None:
    """Persist one page's Groq extract+validate JSON (refreshes timestamp)."""
    if not key or payload is None:
        return
    try:
        with _scrape_lock:
            c = sqlite3.connect(_SCRAPE_DB_PATH["p"], timeout=10)
            try:
                c.execute(
                    "INSERT OR REPLACE INTO extract_cache(key, payload, extracted_at)"
                    " VALUES (?,?,datetime('now'))", (key, json.dumps(payload)))
                c.commit()
            finally:
                c.close()
    except Exception:
        pass


# --------------------------------------------------------- MPN store --------

def get_mpn_store(conn) -> dict:
    """Return {schein_sku: {mpn, manufacturer, source, status, page_mpn}}."""
    out = {}
    try:
        for sku, mpn, mfr, src, st, pm in conn.execute(
                "SELECT schein_sku, mpn, manufacturer, source, status, page_mpn FROM mpn_store"):
            out[sku] = {"mpn": mpn, "manufacturer": mfr, "source": src,
                        "status": st, "page_mpn": pm}
    except Exception:
        pass
    return out


def upsert_mpn(conn, schein_sku, mpn, manufacturer=None, source="manual",
               status="seed", page_mpn=None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO mpn_store(schein_sku, mpn, manufacturer, source, "
        "status, page_mpn, updated_at) VALUES (?,?,?,?,?,?,datetime('now'))",
        (schein_sku, mpn, manufacturer, source, status, page_mpn))
    conn.commit()


def upsert_mpn_now(schein_sku, mpn, manufacturer=None, source="page-consensus",
                   status="verified", page_mpn=None) -> None:
    """Thread-safe immediate upsert (own connection) — called from worker threads
    as page-consensus MPNs are learned during the sweep."""
    if not schein_sku or not mpn:
        return
    try:
        with _scrape_lock:
            c = sqlite3.connect(_SCRAPE_DB_PATH["p"], timeout=10)
            try:
                c.execute(
                    "INSERT OR REPLACE INTO mpn_store(schein_sku, mpn, manufacturer, "
                    "source, status, page_mpn, updated_at) VALUES (?,?,?,?,?,?,datetime('now'))",
                    (schein_sku, mpn, manufacturer, source, status, page_mpn))
                c.commit()
            finally:
                c.close()
    except Exception:
        pass


def save_discovery_now(schein_sku: str, urls: list) -> None:
    """Thread-safe immediate discovery save (own connection), so discovery URLs
    are persisted as each item's sweep finishes — not held in memory until the
    end of the run where a freeze would lose them."""
    if not schein_sku or not urls:
        return
    try:
        with _scrape_lock:
            c = sqlite3.connect(_SCRAPE_DB_PATH["p"], timeout=10)
            try:
                c.execute(
                    "INSERT OR REPLACE INTO discovery_cache(schein_sku, urls_json, discovered_at)"
                    " VALUES (?,?,datetime('now'))",
                    (schein_sku, json.dumps(list(urls)[:40])))
                c.commit()
            finally:
                c.close()
    except Exception:
        pass