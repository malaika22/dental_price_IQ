"""SQLite persistence. Schema covers Stages 1+2 (active) and Stages 3+4
(tables present, empty — PRD Section 10)."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

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