# Dental Supply Price Intelligence System (Stages 1 + 2)

Automated Henry Schein order analysis: parses the weekly order PDF, runs a
broad public-web price sweep (no hardcoded supplier list), verifies candidate
pages with full JS rendering, applies the four trust criteria
(brand + product name + size/form + pack qty), evaluates the client
equivalency table, and produces the three output files:

1. `*_price_match.xlsx` — exact matches only, sorted by total savings desc
2. `*_alternate_purchases.xlsx` — exact/close equivalents (table-driven)
3. `*_evidence.xlsx` — every URL, price, condition, rejection + parsed-order audit

Product URLs are included in **all three** files.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
export SERPAPI_API_KEY=...
export FIRECRAWL_API_KEY=fc-...
```

## Run

```bash
# CLI — one order
python -m app.main path/to/order.pdf

# Dry run (parse + reports only, no API spend) — used for parser validation
python -m app.main path/to/order.pdf --skip-search

# API server
uvicorn app.main:app --port 8000
# POST a PDF to /orders/run, download from /reports/{filename}
```

## Architecture (modular per PRD §10)

| Layer | File | Notes |
|---|---|---|
| Intake / parsing | `app/parser.py` | PyMuPDF token state machine; OCR fallback hook (Tesseract) for scanned PDFs |
| AI reasoning | `app/ai.py` | Groq `llama-3.3-70b-versatile` (free tier): batched description parsing (brand/variant/pack/MPN), 4-criteria match validation, equivalency scoring. Built-in request pacing (~28 req/min) + backoff for free-tier rate limits. `GROQ_MODEL` / `GROQ_MIN_INTERVAL` env-configurable. |
| Market sweep | `app/search.py` | SerpAPI `google_shopping` + organic backfill — whole-web, no supplier list |
| Page verification | `app/search.py` | Firecrawl v2 JS-rendered scrape with JSON schema (price, pack, variant, login-wall detection) |
| Exact matching | `app/matcher.py` | Deterministic pre-checks (price sanity, volume normalization, exact pack equality) + AI validation |
| Equivalency (Stage 2) | `app/matcher.py` | Driven only by `config/equivalency_table.txt`; stream separation enforced |
| Reports | `app/reports.py` | openpyxl, exact PRD §4.7 columns |
| Persistence | `app/db.py` | SQLite schema includes empty Stage 3/4 tables (order_history, par_levels, consumption_model, reorder_projections) |

## Client-editable config (no code changes)

- `config/excluded_domains.txt` — Henry Schein, eBay, Amazon, marketplaces
  excluded; client can add/remove lines
- `config/equivalency_table.txt` — `SKU | equivalent name | brand | note`

## Hard filtering rules

- Excluded domains and all subdomains dropped
- `.pdf` / document links dropped
- Category/taxonomy/search URLs (`/category/`, `/collections/`, `/search`,
  `?q=`, root paths) can never become product matches
- Scraped page content is authoritative over search-result titles
  (shade/pack mismatch between title and page → demoted)
- Pack-mismatched candidates never reach the primary report regardless of
  price; they appear in evidence with the condition shown explicitly
- Login-walled pricing → rejected + listed on the Flagged Sites sheet

## Parser validation (run against the 4 real sample orders)

`python validate_parser.py`

| Order | Items | Total reconciles | Phantom rows |
|---|---|---|---|
| OR202602190851373110 (02/19) | 11/11 | $2,849.58 ✓ | 0 |
| OR202602251732249700 (02/25) | 16/16 | $5,528.06 ✓ | 0 |
| OR202603121104075460 (03/12) | 12/12 | $3,951.58 ✓ | 0 |
| OR202605280844126040 (05/28) | 25/25 | $10,610.55 ✓ | 0 |

Every row passes the `qty × unit ≈ extended` reconciliation guard, so column
misalignment (PRD Issue 7) is structurally impossible.

## Cost note (per ~20-item weekly run)

- SerpAPI: ~2 searches/item ≈ 40-50 searches
- Firecrawl: ~5 page scrapes/item ≈ 100 scrapes
- Groq: ~2 chunked parse calls + 1 validation call/item + equivalency calls — all free tier; a 20-item run is ~25 requests, paced to stay under 30 req/min (~1 minute of AI time)
