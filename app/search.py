"""Market sweep and page verification.

- SerpAPI (google_shopping + google organic) performs the broad public-web
  sweep. No hardcoded supplier list — the whole web is searched (Issue 5
  fixed: real searches, real URLs, real prices).
- Firecrawl scrapes each candidate product page with full JS rendering and
  returns structured JSON (price, pack qty, product name, variant, stock).
  This satisfies the PRD's headless-browser requirement (PRD 4.3) without
  self-managed Playwright.

Filtering rules:
- Excluded domains (henryschein, ebay, amazon, marketplaces, search engines)
  come from config/excluded_domains.txt — client-editable, no code changes.
- PDF / document links are dropped.
- Taxonomy/category/search URLs are dropped: only product detail pages may
  become price candidates.
"""
from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

from .models import OrderLineItem, PriceCandidate

log = logging.getLogger(__name__)

SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))

PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
# URL path fragments that indicate a category/taxonomy/search page, not a product
CATEGORY_PATH_RE = re.compile(
    r"/(category|categories|collections?|catalog|search|shop|browse|brands?|c|s|tag|filter)(/|$|\?)",
    re.I,
)
DOC_EXT_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip)(\?|$)", re.I)


def load_excluded_domains() -> List[str]:
    f = CONFIG_DIR / "excluded_domains.txt"
    if not f.exists():
        return ["henryschein.com", "ebay.com", "amazon.com"]
    out = []
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip().lower()
        if line:
            out.append(line)
    return out


EXCLUDED = load_excluded_domains()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def is_excluded(url: str) -> bool:
    d = _domain(url)
    return any(d == ex or d.endswith("." + ex) for ex in EXCLUDED)


def is_category_url(url: str) -> bool:
    """Taxonomy/category/search URLs must never surface as product matches."""
    path = urlparse(url).path or "/"
    if path in ("", "/"):
        return True
    if CATEGORY_PATH_RE.search(path):
        return True
    if "?q=" in url or "search" in (urlparse(url).query or "").lower():
        return True
    return False


def usable(url: str) -> bool:
    return bool(url) and not is_excluded(url) and not DOC_EXT_RE.search(url) and not is_category_url(url)


def _parse_price(s) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = PRICE_RE.search(str(s))
    return float(m.group(1).replace(",", "")) if m else None


# ----------------------------------------------------------------- SerpAPI --

def _serpapi(params: dict) -> dict:
    params = {**params, "api_key": SERPAPI_KEY}
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=40)
    r.raise_for_status()
    return r.json()


def market_sweep(item: OrderLineItem, max_candidates: int = 8) -> tuple[List[PriceCandidate], List[str]]:
    """Broad public sweep: Google Shopping first, organic as backfill.

    Returns (candidates, flagged_sites). Runs on EVERY item EVERY run
    (PRD 4.2 — never skipped).
    """
    query = item.search_query or item.description
    log.info(
        "SKU %s — market sweep starting (query=%r)",
        item.schein_sku, query[:120],
    )
    cands: List[PriceCandidate] = []
    flagged: List[str] = []
    seen_urls: set[str] = set()
    shopping_count = 0
    organic_count = 0

    # 1. Google Shopping — structured prices
    if not SERPAPI_KEY:
        log.warning("SKU %s — SerpAPI key missing; shopping sweep skipped", item.schein_sku)
    else:
        try:
            log.info("SKU %s — SerpAPI google_shopping …", item.schein_sku)
            data = _serpapi({"engine": "google_shopping", "q": query, "gl": "us", "hl": "en", "num": 40})
            for r in data.get("shopping_results", []):
                url = r.get("product_link") or r.get("link") or ""
                # prefer direct merchant link when SerpAPI provides it
                if "google.com" in _domain(url) and r.get("link"):
                    url = r["link"]
                if not usable(url) or url in seen_urls:
                    continue
                seen_urls.add(url)
                shopping_count += 1
                cands.append(PriceCandidate(
                    title=r.get("title", ""),
                    url=url,
                    source_site=r.get("source") or _domain(url),
                    price=_parse_price(r.get("extracted_price") or r.get("price")),
                ))
            log.info(
                "SKU %s — google_shopping: %d usable result(s)",
                item.schein_sku, shopping_count,
            )
        except Exception as e:
            log.error("SKU %s — google_shopping sweep FAILED: %s", item.schein_sku, e)

    # 2. Organic backfill — catches dental suppliers without shopping feeds
    if SERPAPI_KEY:
        try:
            log.info("SKU %s — SerpAPI google organic backfill …", item.schein_sku)
            data = _serpapi({"engine": "google", "q": f"{query} price buy", "gl": "us", "hl": "en", "num": 20})
            for r in data.get("organic_results", []):
                url = r.get("link", "")
                if not usable(url) or url in seen_urls:
                    continue
                seen_urls.add(url)
                organic_count += 1
                cands.append(PriceCandidate(
                    title=r.get("title", ""),
                    url=url,
                    source_site=_domain(url),
                    price=_parse_price(r.get("snippet", "")),  # provisional; Firecrawl confirms
                ))
            log.info(
                "SKU %s — organic backfill: %d usable result(s)",
                item.schein_sku, organic_count,
            )
        except Exception as e:
            log.error("SKU %s — organic sweep FAILED: %s", item.schein_sku, e)

    # Cheapest-first, keep a shortlist for verification
    cands.sort(key=lambda c: (c.price is None, c.price or 0))
    shortlist = cands[:max_candidates]
    if shortlist:
        log.info(
            "SKU %s — market sweep done: %d candidate(s) shortlisted (cheapest=$%s at %s)",
            item.schein_sku, len(shortlist),
            shortlist[0].price, shortlist[0].source_site,
        )
    else:
        log.warning("SKU %s — market sweep done: no candidates found", item.schein_sku)
    return shortlist, flagged


# ---------------------------------------------------------------- Firecrawl --

FIRECRAWL_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string"},
        "price": {"type": "number", "description": "current selling price in USD, the price a buyer pays now"},
        "pack_quantity": {"type": "integer", "description": "units per pack/box as sold on this page"},
        "variant": {"type": "string", "description": "shade, color, or size variant shown on this page"},
        "in_stock": {"type": "boolean"},
        "requires_login_for_price": {"type": "boolean"},
        "minimum_order_condition": {"type": "string", "description": "any case-lot or minimum quantity condition attached to the price"},
    },
    "required": ["product_name"],
}


def firecrawl_verify(c: PriceCandidate, sku: str = "") -> PriceCandidate:
    """Render the candidate page (JS executed) and extract authoritative data.

    The scraped page is the source of truth — if it disagrees with the search
    result title (wrong shade, wrong pack), the scraped values win.
    """
    tag = f"SKU {sku} — " if sku else ""
    if not FIRECRAWL_KEY:
        log.warning("%sFirecrawl key missing; skipping scrape of %s", tag, c.url[:80])
        c.notes = "Firecrawl API key not configured"
        return c
    log.info("%sscraping %s (%s) …", tag, c.source_site, c.url[:100])
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}",
                     "Content-Type": "application/json"},
            json={
                "url": c.url,
                "formats": [{"type": "json", "schema": FIRECRAWL_SCHEMA}],
                "onlyMainContent": True,
                "waitFor": 3000,            # let JS pricing render
                "timeout": 45000,
            },
            timeout=70,
        )
        r.raise_for_status()
        payload = r.json().get("data", {})
        extracted = payload.get("json") or payload.get("extract") or {}
    except Exception as e:
        log.error("%sscrape FAILED for %s: %s", tag, c.url[:80], e)
        c.notes = f"page verification failed: {e}"
        return c

    if extracted.get("requires_login_for_price"):
        log.warning("%sscrape: login wall at %s", tag, c.url[:80])
        c.match_type = "rejected"
        c.rejected_reason = "login required for pricing (public pricing only)"
        return c

    c.scraped_product_name = extracted.get("product_name")
    c.scraped_variant = extracted.get("variant")
    verified_price = _parse_price(extracted.get("price"))
    if verified_price:
        c.price = verified_price            # page price overrides search snippet
    if extracted.get("pack_quantity"):
        c.pack_qty = int(extracted["pack_quantity"])
    if extracted.get("in_stock") is not None:
        c.in_stock = bool(extracted["in_stock"])
    if extracted.get("minimum_order_condition"):
        c.pack_condition = extracted["minimum_order_condition"]
    log.info(
        "%sscrape SUCCESS — price=$%s pack=%s product=%r",
        tag, c.price, c.pack_qty,
        (c.scraped_product_name or c.title)[:60],
    )
    return c
