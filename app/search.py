"""Market sweep and page verification.

- SerpAPI (google_shopping + google organic + PAID ADS) performs the broad
  public-web sweep. No hardcoded supplier list (Issue 5 fixed).
- Firecrawl scrapes each candidate product page with full JS rendering
  (satisfies PRD 4.3 without self-managed Playwright).

v3 changes:
1. MULTI-QUERY: each item sweeps with its brand query AND a generic query
   (brand stripped) produced by the AI parse step — essential for Henry
   Schein house brands (Acclean, Maxima, Criterion…) that no other supplier
   sells under that name.
2. ADS INGESTION: SerpAPI `ads` and `inline_shopping_results` arrays are now
   read alongside organic/shopping, and gclid/utm/srsltid tracking params are
   stripped (canonical URLs) before dedupe and scraping.
3. DEEPER, FAIRER SHORTLIST: max_candidates 8→14, ≤2 candidates per domain,
   and up to 3 slots reserved for dental-supplier-looking domains whose
   snippet had no price (Firecrawl confirms price on-page anyway).

Filtering rules unchanged: excluded domains (client-editable config), .pdf /
document links, and taxonomy/category/search URLs never become candidates.
"""
from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

import requests

from .models import OrderLineItem, PriceCandidate

log = logging.getLogger(__name__)

SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))

PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
CATEGORY_PATH_RE = re.compile(
    r"/(category|categories|collections?|catalog|search|shop|browse|brands?|c|s|tag|filter)(/|$|\?)",
    re.I,
)
DOC_EXT_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip)(\?|$)", re.I)

# Tracking params stripped before dedupe/scraping (gclid etc. = paid-ad clickthroughs)
TRACKING_KEYS = {"gclid", "gbraid", "wbraid", "gclsrc", "dclid", "msclkid",
                 "fbclid", "srsltid", "gad_source", "gad_campaignid"}
TRACKING_PREFIXES = ("utm_", "hsa_", "gad_")

# Heuristic for "looks like a dental supplier" — used only to reserve
# shortlist slots for price-less snippets worth verifying on-page.
KNOWN_SUPPLIERS = {
    "net32.com", "pricenex.com", "supplyclinic.com", "safcodental.com",
    "frontierdental.com", "curio.dental", "tdsc.com", "crazydentalprices.com",
    "dentalcity.com", "scottsdental.com", "darbydental.com", "medexsupply.com",
    "mvpdentalsupply.com", "ddsdentalsupplies.com", "thedentalmarket.net",
    "skydentalsupply.com", "ansondental.com", "amtouch.com", "zoro.com",
}


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


def canonical_url(url: str) -> str:
    """Strip ad/tracking parameters so paid-ad clickthrough URLs dedupe and
    scrape as their real product page."""
    if not url:
        return url
    try:
        p = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() not in TRACKING_KEYS
             and not k.lower().startswith(TRACKING_PREFIXES)]
        return urlunparse(p._replace(query=urlencode(q, doseq=True), fragment=""))
    except Exception:
        return url


def is_excluded(url: str) -> bool:
    d = _domain(url)
    return any(d == ex or d.endswith("." + ex) for ex in EXCLUDED)


def is_category_url(url: str) -> bool:
    path = urlparse(url).path or "/"
    if path in ("", "/"):
        return True
    if CATEGORY_PATH_RE.search(path):
        return True
    # net32-style taxonomy slugs (…-t-515-614) are listing pages, not products
    if re.search(r"-t-\d+(?:-\d+)*/?$", path):
        return True
    if "?q=" in url or "search" in (urlparse(url).query or "").lower():
        return True
    return False


def usable(url: str) -> bool:
    return bool(url) and not is_excluded(url) and not DOC_EXT_RE.search(url) and not is_category_url(url)


def _looks_supplier(domain: str) -> bool:
    return domain in KNOWN_SUPPLIERS or "dent" in domain


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


def _item_queries(item: OrderLineItem) -> List[str]:
    """Brand query + generic (brand-stripped) query from the AI parse step."""
    queries = []
    for q in (item.search_query or item.description, getattr(item, "generic_query", None)):
        q = (q or "").strip()
        if q and q.lower() not in (x.lower() for x in queries):
            queries.append(q)
    return queries


def _collect(results: list, source_hint: str, cands: list, seen: set,
             price_keys=("extracted_price", "price")) -> int:
    added = 0
    for r in results or []:
        url = canonical_url(r.get("product_link") or r.get("link")
                            or r.get("tracking_link") or "")
        if "google.com" in _domain(url) and r.get("link"):
            url = canonical_url(r["link"])
        if not usable(url) or url in seen:
            continue
        seen.add(url)
        price = None
        for k in price_keys:
            price = _parse_price(r.get(k))
            if price:
                break
        if price is None:
            price = _parse_price(r.get("snippet", ""))
        cands.append(PriceCandidate(
            title=r.get("title", ""),
            url=url,
            source_site=r.get("source") or _domain(url),
            price=price,
        ))
        added += 1
    return added


def _shortlist(cands: List[PriceCandidate], max_candidates: int) -> List[PriceCandidate]:
    """≤2 per domain; priced cheapest-first; up to 3 reserved slots for
    price-less snippets on dental-supplier domains (verified on-page later)."""
    by_dom: dict[str, int] = {}
    capped: List[PriceCandidate] = []
    for c in sorted(cands, key=lambda c: (c.price is None, c.price or 0)):
        d = _domain(c.url)
        if by_dom.get(d, 0) >= 2:
            continue
        by_dom[d] = by_dom.get(d, 0) + 1
        capped.append(c)
    priced = [c for c in capped if c.price is not None]
    reserve = [c for c in capped
               if c.price is None and _looks_supplier(_domain(c.url))][:3]
    out = priced[:max(0, max_candidates - len(reserve))] + reserve
    return out[:max_candidates]


def market_sweep(item: OrderLineItem, max_candidates: int = 14) -> tuple[List[PriceCandidate], List[str]]:
    """Broad public sweep. Runs on EVERY item EVERY run (PRD 4.2)."""
    queries = _item_queries(item)
    log.info("SKU %s — market sweep starting (%d quer%s: %s)",
             item.schein_sku, len(queries), "y" if len(queries) == 1 else "ies",
             " | ".join(q[:60] for q in queries))
    cands: List[PriceCandidate] = []
    flagged: List[str] = []
    seen: set[str] = set()

    if not SERPAPI_KEY:
        log.warning("SKU %s — SerpAPI key missing; sweep skipped", item.schein_sku)
        return [], flagged

    for qi, query in enumerate(queries, start=1):
        # 1. Google Shopping — structured prices
        try:
            data = _serpapi({"engine": "google_shopping", "q": query,
                             "gl": "us", "hl": "en", "num": 40})
            n = _collect(data.get("shopping_results"), "shopping", cands, seen)
            n += _collect(data.get("shopping_ads"), "shopping_ads", cands, seen)
            log.info("SKU %s — q%d google_shopping: %d usable result(s)",
                     item.schein_sku, qi, n)
        except Exception as e:
            log.error("SKU %s — q%d google_shopping FAILED: %s", item.schein_sku, qi, e)

        # 2. Google organic + PAID ADS + inline shopping ads
        try:
            data = _serpapi({"engine": "google", "q": f"{query} price buy",
                             "gl": "us", "hl": "en", "num": 20})
            n = _collect(data.get("organic_results"), "organic", cands, seen)
            n += _collect(data.get("ads"), "ads", cands, seen)
            n += _collect(data.get("inline_shopping_results"), "inline_ads", cands, seen)
            log.info("SKU %s — q%d organic+ads: %d usable result(s)",
                     item.schein_sku, qi, n)
        except Exception as e:
            log.error("SKU %s — q%d organic/ads FAILED: %s", item.schein_sku, qi, e)

    shortlist = _shortlist(cands, max_candidates)
    if shortlist:
        log.info("SKU %s — sweep done: %d/%d shortlisted (cheapest=$%s at %s)",
                 item.schein_sku, len(shortlist), len(cands),
                 shortlist[0].price, shortlist[0].source_site)
    else:
        log.warning("SKU %s — sweep done: no candidates found", item.schein_sku)
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
    The scraped page is the source of truth over the search-result title."""
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
                "waitFor": 3000,
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
        c.price = verified_price
    if extracted.get("pack_quantity"):
        c.pack_qty = int(extracted["pack_quantity"])
    if extracted.get("in_stock") is not None:
        c.in_stock = bool(extracted["in_stock"])
    if extracted.get("minimum_order_condition"):
        c.pack_condition = extracted["minimum_order_condition"]
    log.info("%sscrape SUCCESS — price=$%s pack=%s product=%r",
             tag, c.price, c.pack_qty, (c.scraped_product_name or c.title)[:60])
    return c