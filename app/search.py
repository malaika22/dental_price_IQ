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
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

import requests

from .models import OrderLineItem, PriceCandidate

log = logging.getLogger(__name__)

SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_MAX_SCRAPES = int(os.environ.get("FIRECRAWL_MAX_SCRAPES_PER_RUN", "150"))
_fc_state = {"exhausted": False, "scrapes": 0, "credits": 0}
_fc_lock = threading.Lock()
# Total Firecrawl credit budget for a run and the slice reserved exclusively
# for the stage-3 supplier fallback (so open-web /search can't drain it all).
FIRECRAWL_RUN_CREDITS = int(os.environ.get("FIRECRAWL_RUN_CREDITS", "1000"))
FIRECRAWL_STAGE3_RESERVE = int(os.environ.get("FIRECRAWL_STAGE3_RESERVE", "50"))


def _fc_credits_left() -> int:
    with _fc_lock:
        return FIRECRAWL_RUN_CREDITS - _fc_state["credits"]


def _fc_add_credits(n: int):
    with _fc_lock:
        _fc_state["credits"] += max(0, int(n or 0))


def reset_firecrawl_budget():
    """Call at the start of each pipeline run."""
    with _fc_lock:
        _fc_state["exhausted"] = False
        _fc_state["scrapes"] = 0
        _fc_state["credits"] = 0
    _gp_state["disabled"] = False
    _gp_state["failures"] = 0
    _serp_state["exhausted"] = False
    _serp_state["consecutive_failures"] = 0
    _shop_state["disabled"] = False
    _shop_state["zero_streak"] = 0
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))

PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
CATEGORY_PATH_RE = re.compile(
    r"/(category|categories|collections?|catalog|search|shop|browse|brands?|c|s|tag|filter)(/|$|\?)",
    re.I,
)
DOC_EXT_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip)(\?|$)", re.I)
CONTENT_PATH_RE = re.compile(r"/(blog|blogs|forum|forums|community|news|article|articles|reviews?)(/|$)", re.I)
CONTENT_SUBDOMAIN_RE = re.compile(r"^(forum|forums|community|blog|blogs|news)\.", re.I)

# Tracking params stripped before dedupe/scraping (gclid etc. = paid-ad clickthroughs)
TRACKING_KEYS = {"gclid", "gbraid", "wbraid", "gclsrc", "dclid", "msclkid",
                 "fbclid", "srsltid", "gad_source", "gad_campaignid",
                 "promo", "_pos", "_sid", "_ss", "queryid", "ref", "affid"}
TRACKING_PREFIXES = ("utm_", "hsa_", "gad_")

# Heuristic for "looks like a dental supplier" — used only to reserve
# shortlist slots for price-less snippets worth verifying on-page.
KNOWN_SUPPLIERS = {
    "net32.com", "pricenex.com", "supplyclinic.com", "safcodental.com",
    "frontierdental.com", "curio.dental", "tdsc.com", "crazydentalprices.com",
    "dentalcity.com", "scottsdental.com", "darbydental.com", "medexsupply.com",
    "mvpdentalsupply.com", "ddsdentalsupplies.com", "thedentalmarket.net",
    "skydentalsupply.com", "ansondental.com", "amtouch.com", "zoro.com",
    "benco.com", "surgimac.com", "practicon.com", "pearsondental.com",
    "medicalgassupplier.com", "vitalitymedical.com", "ciamedical.com",
    "amerdental.com", "usdentaldepot.com", "chasedentalsupply.com",
    "optimusdentalsupply.com", "davisdentalsupply.com",
    "myddssupply.com", "dentalwholesaledirect.com", "primedentalsupply.com",
    "odshc.com", "supplydoc.com", "pstshop.com", "dentalaccessories.org",
    "curio.dental", "dentalsky.com", "dhpsupply.com",
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


def load_supplier_sites() -> List[str]:
    """Trusted supplier domains searched directly per item (config-driven)."""
    f = CONFIG_DIR / "supplier_sites.txt"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip().lower()
        if line:
            out.append(line)
    return out


SUPPLIER_SITES = load_supplier_sites()

# Per-SKU discovery cache (loaded from DB at run start, flushed at run end).
# Maps schein_sku -> list[url]. Skips re-discovery of repeat items.
_disc_cache = {}          # sku -> [urls]      (read-only during the run)
_disc_new = {}            # sku -> [urls]      (discovered this run, to persist)
_disc_lock = threading.Lock()
DISCOVERY_CACHE_DAYS = int(os.environ.get("DISCOVERY_CACHE_DAYS", "7"))
DISCOVERY_CACHE_ENABLED = os.environ.get("DISCOVERY_CACHE", "1") not in ("0", "false", "False")
# Scrape cache: reuse scraped markdown for SCRAPE_CACHE_HOURS so a re-run after
# a freeze (or a repeat order) skips re-scraping — the biggest credit saver.
SCRAPE_CACHE_HOURS = int(os.environ.get("SCRAPE_CACHE_HOURS", "24"))
SCRAPE_CACHE_ENABLED = os.environ.get("SCRAPE_CACHE", "1") not in ("0", "false", "False")


def load_discovery_cache(conn) -> None:
    """Populate the in-memory discovery cache from the DB once per run."""
    _disc_cache.clear(); _disc_new.clear()
    if not (DISCOVERY_CACHE_ENABLED and conn):
        return
    try:
        from . import db
        cur = conn.cursor()
        cur.execute("SELECT schein_sku FROM discovery_cache WHERE "
                    "discovered_at >= datetime('now', ?)",
                    (f"-{DISCOVERY_CACHE_DAYS} days",))
        for (sku,) in cur.fetchall():
            urls = db.get_cached_discovery(conn, sku, DISCOVERY_CACHE_DAYS)
            if urls:
                _disc_cache[sku] = urls
        if _disc_cache:
            log.info("Discovery cache: %d SKU(s) loaded (<%d days old) — repeat "
                     "items will skip re-discovery", len(_disc_cache), DISCOVERY_CACHE_DAYS)
    except Exception as e:
        log.debug("discovery cache load skipped: %s", e)


def flush_discovery_cache(conn) -> None:
    """Persist URLs discovered this run back to the DB."""
    if not (DISCOVERY_CACHE_ENABLED and conn and _disc_new):
        return
    try:
        from . import db
        for sku, urls in _disc_new.items():
            db.save_discovery(conn, sku, urls)
        log.info("Discovery cache: persisted %d SKU(s) for future runs", len(_disc_new))
    except Exception as e:
        log.debug("discovery cache flush skipped: %s", e)
if SUPPLIER_SITES:
    log.info("Trusted-supplier sweep ENABLED — %d domains from %s",
             len(SUPPLIER_SITES), CONFIG_DIR / "supplier_sites.txt")
else:
    log.warning("Trusted-supplier sweep DISABLED — no supplier_sites.txt found at %s "
                "(cheapest pages on Curio/Surgimac/Frontier may be missed). "
                "Deploy config/supplier_sites.txt to enable.", CONFIG_DIR)


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
    if "henryschein" in d:        # henryschein.com, henryscheindental.com, etc.
        return True
    return any(d == ex or d.endswith("." + ex) for ex in EXCLUDED)


def is_category_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path or "/"
    d = parsed.netloc.lower().removeprefix("www.")
    if path in ("", "/"):
        return True
    if CONTENT_SUBDOMAIN_RE.match(parsed.netloc.lower()):
        return True                       # forum./community./blog. subdomains
    if CONTENT_PATH_RE.search(path):
        return True                       # /blog/, /forum/, /news/ content pages
    # net32: product detail slugs end in -d-<id>; -t-/-l- slugs are taxonomy
    if d.endswith("net32.com") and not re.search(r"-d-\d", path):
        return True
    if CATEGORY_PATH_RE.search(path):
        # …unless it is clearly a product detail page within that section
        # (e.g. pearsondental.com/catalog/product.asp?pid=110259)
        q = (urlparse(url).query or "").lower()
        if "product" not in path.lower() and "pid=" not in q and "sku=" not in q:
            return True
    if re.search(r"-[tl]-\d+(?:-\d+)*/?$", path):
        return True                       # generic -t-/-l- taxonomy slugs
    if "?q=" in url or "search" in (urlparse(url).query or "").lower():
        return True
    return False


def usable(url: str) -> bool:
    return bool(url) and not is_excluded(url) and not DOC_EXT_RE.search(url) and not is_category_url(url)


def _looks_supplier(domain: str) -> bool:
    return (domain in KNOWN_SUPPLIERS or domain in SUPPLIER_SITES
            or any(domain.endswith(s) for s in SUPPLIER_SITES) or "dent" in domain)


def _parse_price(s) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = PRICE_RE.search(str(s))
    return float(m.group(1).replace(",", "")) if m else None


# ----------------------------------------------------------------- SerpAPI --

import time as _time

_serp_lock = threading.Lock()
_serp_last = [0.0]
SERP_MIN_INTERVAL = float(os.environ.get("SERPAPI_MIN_INTERVAL", "1.2"))  # serialize ~1/1.2s to avoid burst 429s
SERP_MAX_RETRIES = int(os.environ.get("SERPAPI_MAX_RETRIES", "5"))
_serp_state = {"exhausted": False, "consecutive_failures": 0}
SERP_FAIL_GIVEUP = int(os.environ.get("SERPAPI_GIVEUP_AFTER", "8"))  # consecutive failed calls = real wall


def _serp_pace():
    with _serp_lock:
        wait = SERP_MIN_INTERVAL - (_time.monotonic() - _serp_last[0])
        if wait > 0:
            _time.sleep(wait)
        _serp_last[0] = _time.monotonic()


def _serpapi(params: dict) -> dict:
    """SerpAPI call with pacing + 429 backoff. Once the monthly/credit quota is
    exhausted (persistent 429), a run-level flag short-circuits further calls so
    we fail fast instead of stalling every remaining item."""
    # Only give up entirely after MANY consecutive failures (a real quota wall),
    # NOT after one item — SerpAPI throttling is intermittent and recovers.
    if _serp_state["exhausted"]:
        raise RuntimeError("SerpAPI quota wall hit earlier this run")
    params = {**params, "api_key": SERPAPI_KEY}
    last_err = None
    for attempt in range(SERP_MAX_RETRIES):
        _serp_pace()
        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=40)
            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                backoff = float(ra) if ra and ra.replace(".", "").isdigit() else min(4 * (attempt + 1), 30)
                log.warning("SerpAPI 429 — backing off %ss (attempt %d/%d)",
                            round(backoff), attempt + 1, SERP_MAX_RETRIES)
                _time.sleep(backoff)
                last_err = requests.HTTPError("429 Too Many Requests")
                continue
            r.raise_for_status()
            with _serp_lock:
                _serp_state["consecutive_failures"] = 0   # success resets the wall counter
            return r.json()
        except requests.HTTPError:
            raise
        except Exception as e:
            last_err = e
            _time.sleep(min(2 * (attempt + 1), 10))
    # this call failed after all retries
    with _serp_lock:
        _serp_state["consecutive_failures"] += 1
        if _serp_state["consecutive_failures"] >= SERP_FAIL_GIVEUP:
            _serp_state["exhausted"] = True
            log.error("SerpAPI failed %d calls in a row — likely a real quota/rate "
                      "wall. Remaining items this run will be skipped. Raise "
                      "SERPAPI_MIN_INTERVAL, lower SUPPLIER_SWEEP_MAX_SITES, or upgrade plan.",
                      SERP_FAIL_GIVEUP)
    raise last_err or RuntimeError("SerpAPI call failed")


def _item_queries(item: OrderLineItem) -> List[str]:
    """Brand query + generic (brand-stripped) query from the AI parse step."""
    queries = []
    for q in (item.search_query or item.description,
              getattr(item, "generic_query", None),
              getattr(item, "mpn_query", None)):
        q = (q or "").strip()
        if q and q.lower() not in (x.lower() for x in queries):
            queries.append(q)
    return queries


_gp_state = {"disabled": False, "failures": 0}
# google_shopping auto-skip: if it yields 0 usable candidates for this many
# items in a row, stop calling it for the rest of the run (logs show it
# consistently returning 0 usable on this SerpAPI plan). Saves 1 call/query/item.
_shop_state = {"disabled": False, "zero_streak": 0}
SHOP_DISABLE_AFTER = int(os.environ.get("GOOGLE_SHOPPING_DISABLE_AFTER", "3"))


def _resolve_google_product(product_id: str, cands: list, seen: set,
                            max_sellers: int = 5) -> int:
    if _gp_state["disabled"]:
        return 0
    """SerpAPI google_product sellers lookup: turns a google.com product_link
    into direct merchant links with prices. This recovers the entire Google
    Shopping channel (previously 100% discarded as google.com URLs)."""
    try:
        data = _serpapi({"engine": "google_product", "product_id": product_id,
                         "google_domain": "google.com", "gl": "us", "hl": "en",
                         "offers": "1"})
    except Exception as e:
        _gp_state["failures"] += 1
        if _gp_state["failures"] >= 5 and not _gp_state["disabled"]:
            _gp_state["disabled"] = True
            log.warning("google_product engine disabled for this run after 5 "
                        "failures (likely not in your SerpAPI plan) — relying on "
                        "direct merchant links + organic/ads, which is sufficient")
        log.debug("google_product resolution failed for %s: %s", product_id, e)
        return 0
    if data.get("error"):
        log.debug("google_product no offers for %s: %s", product_id, data["error"])
        return 0
    sellers = ((data.get("sellers_results") or {}).get("online_sellers")
               or (data.get("product_results") or {}).get("sellers") or [])
    added = 0
    for s in sellers:
        url = canonical_url(s.get("direct_link") or s.get("link") or "")
        if not usable(url) or url in seen or "google.com" in _domain(url):
            continue
        seen.add(url)
        price = _parse_price(s.get("base_price") or s.get("total_price")
                             or s.get("price"))
        cands.append(PriceCandidate(
            title=s.get("product_title") or s.get("name") or "",
            url=url,
            source_site=s.get("name") or _domain(url),
            price=price,
        ))
        added += 1
        if added >= max_sellers:
            break
    return added


def _direct_merchant_link(r: dict) -> str:
    """Pull a real merchant URL out of a shopping result without a 2nd API
    call. SerpApi exposes merchant links under several keys depending on the
    result type."""
    for key in ("link", "product_link", "tracking_link", "merchant_link",
                "seller_link", "offer_link"):
        u = r.get(key)
        if u and "google.com" not in _domain(canonical_url(u)):
            return canonical_url(u)
    # nested seller objects on some shopping/ads results
    for box in (r.get("offers"), r.get("sellers")):
        if isinstance(box, list):
            for s in box:
                u = s.get("link") or s.get("direct_link")
                if u and "google.com" not in _domain(canonical_url(u)):
                    return canonical_url(u)
    return ""


def _collect(results: list, source_hint: str, cands: list, seen: set,
             price_keys=("extracted_price", "price"),
             resolve_budget: list | None = None) -> int:
    added = 0
    for r in results or []:
        url = _direct_merchant_link(r)
        # no direct merchant link → optionally resolve via google_product
        if not url:
            url = canonical_url(r.get("product_link") or r.get("link") or "")
            if "google.com" in _domain(url):
                pid = r.get("product_id")
                if pid and resolve_budget and resolve_budget[0] > 0:
                    resolve_budget[0] -= 1
                    added += _resolve_google_product(str(pid), cands, seen)
                continue
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


def _supplier_site_sweep(item: OrderLineItem, cands: list, seen: set,
                         max_sites: int = int(os.environ.get("SUPPLIER_SWEEP_MAX_SITES", "8"))) -> int:
    """Run a targeted `site:domain <query>` search for each trusted supplier so
    their product pages are pulled in even when absent from generic SerpAPI
    results. One organic call per domain (cheap), capped at max_sites."""
    if not SERPAPI_KEY or not SUPPLIER_SITES:
        return 0
    base = (item.mpn_query or item.search_query or item.description or "").strip()
    # bias the query toward the ordered variant so the correct variant page
    # ranks (e.g. "...Flavorless"/"No Flavor" instead of the default Mint)
    variant = (item.variant or "").strip()
    if variant and variant.lower() not in base.lower():
        base = f"{base} {variant}"
    if not base:
        return 0
    added = 0
    for domain in SUPPLIER_SITES[:max_sites]:
        try:
            data = _serpapi({"engine": "google",
                             "q": f"site:{domain} {base}",
                             "gl": "us", "hl": "en", "num": 5})
            n = _collect(data.get("organic_results"), f"site:{domain}", cands, seen)
            added += n
        except Exception as e:
            log.debug("SKU %s — site:%s search failed: %s", item.schein_sku, domain, e)
    log.info("SKU %s — trusted-supplier sweep: +%d candidate(s) from %d site(s)",
             item.schein_sku, added, min(len(SUPPLIER_SITES), max_sites))
    return added


def _firecrawl_search(query: str, cands: list, seen: set, *,
                      include_domains: list | None = None, limit: int = 10,
                      stage: str = "fc") -> int:
    """Discovery-only Firecrawl /v2/search (no scrapeOptions → 2 credits/10
    results). Returns url/title/description which we map into PriceCandidates;
    the existing shortlist + early-stop then gates which ones get scraped, so
    this never triggers a scrape explosion. Used for SerpAPI fallback (open
    web) and the stage-3 supplier fallback (includeDomains filter)."""
    if not FIRECRAWL_KEY:
        return 0
    body = {"query": query, "limit": limit, "sources": ["web"]}
    if include_domains:
        body["includeDomains"] = include_domains   # restrict to trusted suppliers
    try:
        r = requests.post("https://api.firecrawl.dev/v2/search",
                           headers={"Authorization": f"Bearer {FIRECRAWL_KEY}",
                                    "Content-Type": "application/json"},
                           json=body, timeout=(15, 60))
        if r.status_code == 402:
            _mark_firecrawl_exhausted("402", tag=f"{stage} search ")
            return 0
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        log.warning("Firecrawl /search (%s) failed for %r: %s", stage, query[:50], e)
        return 0
    # account credits actually billed (fallback to the documented 2/10 rule)
    used = payload.get("creditsUsed")
    if used is None:
        used = max(2, -(-limit // 10) * 2)
    _fc_add_credits(used)
    web = (payload.get("data") or {}).get("web") or []
    added = 0
    for r0 in web:
        url = canonical_url(r0.get("url") or r0.get("metadata", {}).get("sourceURL") or "")
        if not usable(url) or url in seen:
            continue
        seen.add(url)
        cands.append(PriceCandidate(
            title=r0.get("title", ""), url=url,
            source_site=_domain(url),
            price=_parse_price(r0.get("description", "")),   # snippet may carry a price
        ))
        added += 1
    if added:
        log.info("Firecrawl /search (%s) — +%d candidate(s) for %r (credits used=%s, left=%d)",
                 stage, added, query[:40], used, _fc_credits_left())
    return added


def _firecrawl_supplier_gap_sweep(item: OrderLineItem, cands: list, seen: set) -> int:
    """Replaces the SerpAPI `site:` supplier sweep. After organic discovery has
    run, find which trusted suppliers ALREADY appeared in the candidate pool and
    search Firecrawl ONLY for the suppliers still missing (the "gap"). One
    Firecrawl /search with includeDomains covers the whole gap in a single call
    (limit=20 → ~4 credits), versus 8+ separate SerpAPI site: searches.

    Aggressive skip rule: a supplier counts as "covered" if ANY of its pages is
    already in the pool, so it's dropped from the gap list."""
    if not (FIRECRAWL_KEY and SUPPLIER_SITES):
        return 0
    # which trusted suppliers are already represented in the candidate pool?
    found_domains = set()
    for c in cands:
        d = _domain(c.url)
        for s in SUPPLIER_SITES:
            if d == s or d.endswith("." + s) or s.endswith(d):
                found_domains.add(s)
    gap = [s for s in SUPPLIER_SITES if s not in found_domains]
    if not gap:
        log.info("SKU %s — supplier gap-sweep skipped: all %d trusted suppliers "
                 "already covered by organic results", item.schein_sku, len(SUPPLIER_SITES))
        return 0
    if _fc_state["exhausted"]:
        return 0
    base = (item.mpn_query or item.search_query or item.description or "").strip()
    variant = (item.variant or "").strip()
    if variant and variant.lower() not in base.lower():
        base = f"{base} {variant}"
    if not base:
        return 0
    log.info("SKU %s — supplier gap-sweep: %d/%d suppliers missing from organic, "
             "1 Firecrawl /search (limit=20) to fill gap", item.schein_sku,
             len(gap), len(SUPPLIER_SITES))
    return _firecrawl_search(base, cands, seen, include_domains=gap, limit=20,
                             stage="supplier-gap")


def market_sweep(item: OrderLineItem, max_candidates: int = 14) -> tuple[List[PriceCandidate], List[str]]:
    """Broad public sweep. Runs on EVERY item EVERY run (PRD 4.2)."""
    queries = _item_queries(item)
    log.info("SKU %s — market sweep starting (%d quer%s: %s)",
             item.schein_sku, len(queries), "y" if len(queries) == 1 else "ies",
             " | ".join(q[:60] for q in queries))
    cands: List[PriceCandidate] = []
    flagged: List[str] = []
    seen: set[str] = set()

    # --- DISCOVERY CACHE HIT: reuse known URLs, skip all discovery API calls ---
    # (Prices are still scraped fresh downstream, so report prices never go
    #  stale from this cache — only discovery calls are saved.)
    cached_urls = _disc_cache.get(item.schein_sku) if DISCOVERY_CACHE_ENABLED else None
    if cached_urls:
        for url in cached_urls:
            cu = canonical_url(url)
            if usable(cu) and cu not in seen:
                seen.add(cu)
                cands.append(PriceCandidate(title="", url=cu, source_site=_domain(cu), price=None))
        if cands:
            log.info("SKU %s — discovery CACHE HIT: %d known URL(s), skipping search "
                     "(scrapes still run fresh)", item.schein_sku, len(cands))
            # Cached candidates are ALL price-less (cache stores URLs only), so
            # the normal priced/reserve split in _shortlist would starve them to
            # just 3 reserve slots. Instead cap ≤2 per domain and take up to
            # max_candidates — giving cached items the same scrape breadth as
            # fresh ones (prices are read on-page during the fresh scrape).
            by_dom: dict[str, int] = {}
            shortlist = []
            for c in cands:
                d = _domain(c.url)
                if by_dom.get(d, 0) >= 2:
                    continue
                by_dom[d] = by_dom.get(d, 0) + 1
                shortlist.append(c)
            shortlist = shortlist[:max_candidates]
            return shortlist, flagged

    serp_ok = bool(SERPAPI_KEY) and not _serp_state["exhausted"]
    if not SERPAPI_KEY:
        log.warning("SKU %s — SerpAPI key missing; will rely on Firecrawl /search",
                    item.schein_sku)

    # ---- STAGE 1: SerpAPI discovery (primary) ---------------------------
    for qi, query in enumerate(queries, start=1) if serp_ok else []:
        # 1. Google Shopping — structured prices. Auto-skipped once it proves
        #    unproductive (0 usable for SHOP_DISABLE_AFTER items in a row),
        #    saving one SerpAPI call per query on the rest of the run.
        if not _shop_state["disabled"]:
            try:
                data = _serpapi({"engine": "google_shopping", "q": query,
                                 "gl": "us", "hl": "en", "num": 40})
                budget = [3]
                raw = len(data.get("shopping_results") or []) + len(data.get("shopping_ads") or [])
                n = _collect(data.get("shopping_results"), "shopping", cands, seen,
                             resolve_budget=budget)
                n += _collect(data.get("shopping_ads"), "shopping_ads", cands, seen,
                              resolve_budget=budget)
                log.info("SKU %s — q%d google_shopping: %d raw → %d usable "
                         "(resolved %d google product link(s))",
                         item.schein_sku, qi, raw, n, 3 - budget[0])
                # track productivity: only the FIRST query per item votes, to
                # avoid double-counting; a single usable result resets the streak
                if qi == 1:
                    if n == 0:
                        _shop_state["zero_streak"] += 1
                        if _shop_state["zero_streak"] >= SHOP_DISABLE_AFTER:
                            _shop_state["disabled"] = True
                            log.warning("google_shopping disabled for this run after "
                                        "%d items with 0 usable results — relying on "
                                        "organic+ads (saves 1 SerpAPI call/query).",
                                        SHOP_DISABLE_AFTER)
                    else:
                        _shop_state["zero_streak"] = 0
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

    # Trusted-supplier coverage: gap-based Firecrawl sweep. Only searches for
    # suppliers NOT already surfaced by organic discovery (1 Firecrawl call vs
    # 8+ SerpAPI site: calls). Runs regardless of SerpAPI state.
    _firecrawl_supplier_gap_sweep(item, cands, seen)

    # ---- STAGE 2: Firecrawl /search open-web fallback -------------------
    # Fires when SerpAPI is unavailable/exhausted. Discovery-only (2 credits per
    # 10 results); the shortlist + early-stop still gate scraping. Stops short
    # of the reserved floor so stage 3 always has credits.
    serp_dead = (not SERPAPI_KEY) or _serp_state["exhausted"]
    if serp_dead and not _fc_state["exhausted"]:
        if _fc_credits_left() > FIRECRAWL_STAGE3_RESERVE:
            log.info("SKU %s — SerpAPI unavailable; STAGE 2 Firecrawl /search "
                     "(open web), credits left=%d", item.schein_sku, _fc_credits_left())
            for query in queries:
                if _fc_credits_left() <= FIRECRAWL_STAGE3_RESERVE:
                    log.info("SKU %s — hit stage-3 reserve floor (%d); stopping open-web search",
                             item.schein_sku, FIRECRAWL_STAGE3_RESERVE)
                    break
                _firecrawl_search(query, cands, seen, limit=10, stage="stage2-web")

        # ---- STAGE 3: Firecrawl /search restricted to trusted suppliers ----
        # Last resort, funded by the reserved credit floor. includeDomains keeps
        # results on the client's trusted supplier list only.
        if SUPPLIER_SITES and not _fc_state["exhausted"] and _fc_credits_left() > 0:
            base = (item.mpn_query or item.search_query or item.description or "").strip()
            variant = (item.variant or "").strip()
            if variant and variant.lower() not in base.lower():
                base = f"{base} {variant}"
            if base:
                log.info("SKU %s — STAGE 3 supplier fallback (reserve=%d credits): "
                         "Firecrawl /search restricted to %d trusted domains",
                         item.schein_sku, FIRECRAWL_STAGE3_RESERVE,
                         min(len(SUPPLIER_SITES), 10))
                # one combined includeDomains search is far cheaper than N site: calls
                _firecrawl_search(base, cands, seen,
                                  include_domains=SUPPLIER_SITES[:10],
                                  limit=10, stage="stage3-suppliers")

    shortlist = _shortlist(cands, max_candidates)
    # record discovered URLs for future runs (all candidates, not just shortlist,
    # so next run has the full set to re-rank against fresh prices)
    if DISCOVERY_CACHE_ENABLED and cands:
        with _disc_lock:
            _disc_new[item.schein_sku] = [c.url for c in cands]
        # also persist NOW (own thread-safe connection) so a freeze later in the
        # run doesn't lose the discovery work already paid for this item
        try:
            from . import db
            db.save_discovery_now(item.schein_sku, [c.url for c in cands])
        except Exception:
            pass
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
        "price": {"type": "number", "description": "current selling price in USD for the EXACT variant/pack size this URL points to. If the page offers multiple variants, flavors, or pack sizes, return the price of the option matching the URL/title — NEVER the default, lowest, or per-unit price of a different option"},
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
    # SCRAPE CACHE: if this URL was scraped recently (this run or a previous one
    # that froze/crashed), reuse the stored markdown — costs ZERO Firecrawl
    # credits. This is what makes a re-run after a freeze cheap.
    if SCRAPE_CACHE_ENABLED:
        try:
            from . import db
            cached_md = db.get_cached_scrape(c.url, SCRAPE_CACHE_HOURS)
        except Exception:
            cached_md = None
        if cached_md:
            c.scraped_markdown = cached_md[:int(os.environ.get("SCRAPE_MD_CAP", "2000"))]
            log.info("%sscrape CACHE HIT (%d chars, 0 credits) — %s",
                     tag, len(c.scraped_markdown), c.url[:60])
            return c
    with _fc_lock:
        if _fc_state["exhausted"]:
            c.notes = "scrape skipped — Firecrawl credits exhausted (402) earlier in this run"
            return c
        if _fc_state["scrapes"] >= FIRECRAWL_MAX_SCRAPES:
            c.notes = f"scrape skipped — per-run budget of {FIRECRAWL_MAX_SCRAPES} scrapes reached"
            return c
        _fc_state["scrapes"] += 1
    if not FIRECRAWL_KEY:
        log.warning("%sFirecrawl key missing; skipping scrape of %s", tag, c.url[:80])
        c.notes = "Firecrawl API key not configured"
        return c
    log.info("%sscraping %s (%s) …", tag, c.source_site, c.url[:100])
    try:
        # Hard total-time deadline: requests' read timeout resets on every byte,
        # so a server trickling data slowly (some net32 listing pages do) can
        # hang far past it. Run the POST in a daemon helper thread and enforce an
        # absolute ceiling. We deliberately do NOT join a thread that blew the
        # ceiling (a `with` executor would block on exit waiting for it) — it is
        # left as a leaked daemon so this function returns immediately and the
        # run never freezes. A handful of leaked threads per run is harmless.
        import concurrent.futures as _cf
        HARD_DEADLINE = int(os.environ.get("SCRAPE_HARD_DEADLINE_SEC", "90"))
        _ex = _cf.ThreadPoolExecutor(max_workers=1)

        def _do_post():
            return requests.post(
                "https://api.firecrawl.dev/v2/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "url": c.url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                    "waitFor": 3000,
                    "timeout": 45000,
                    "maxAge": int(os.environ.get("FIRECRAWL_MAX_AGE_MS", str(7 * 24 * 60 * 60 * 1000))),
                },
                timeout=(15, 60),
            )

        fut = _ex.submit(_do_post)
        try:
            r = fut.result(timeout=HARD_DEADLINE)
        except _cf.TimeoutError:
            log.error("%sscrape HARD TIMEOUT (%ss ceiling) — abandoning slow page %s",
                      tag, HARD_DEADLINE, c.url[:70])
            c.notes = "scrape abandoned — page exceeded hard time ceiling"
            _ex.shutdown(wait=False)   # do NOT block on the stuck thread
            return c
        _ex.shutdown(wait=False)
        r.raise_for_status()
        payload = r.json().get("data", {})
        markdown = payload.get("markdown") or ""
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        log.error("%sscrape FAILED (%s) for %s", tag, status, c.url[:80])
        if status == 402:
            with _fc_lock:
                if not _fc_state["exhausted"]:
                    _fc_state["exhausted"] = True
                    log.error("FIRECRAWL CREDITS EXHAUSTED (402) — all further "
                              "scrapes this run will be skipped; candidates will be "
                              "marked unverified. Top up the Firecrawl plan and re-run.")
            c.notes = "not verified — Firecrawl credits exhausted (402)"
            return c
        if status in (404, 410):
            c.match_type = "rejected"
            c.rejected_reason = "page not found (404) — dead or expired listing"
        else:
            c.notes = f"page verification failed: HTTP {status}"
        return c
    except requests.Timeout:
        log.error("%sscrape TIMEOUT (socket stalled) for %s — skipping candidate",
                  tag, c.url[:80])
        c.notes = "page verification timed out (slow/unresponsive site)"
        return c
    except Exception as e:
        log.error("%sscrape FAILED for %s: %s", tag, c.url[:80], e)
        c.notes = f"page verification failed: {e}"
        return c

    # cheap login-wall heuristic on the raw markdown (no AI call needed) so
    # gated pages are rejected before they reach the Groq extraction batch.
    low = markdown.lower()
    if not markdown.strip():
        c.notes = "scrape returned empty content"
        return c
    if (("log in" in low or "login" in low or "sign in" in low or "member price" in low
         or "call for price" in low or "request a quote" in low)
            and "$" not in markdown):
        log.warning("%sscrape: likely login wall at %s", tag, c.url[:80])
        c.match_type = "rejected"
        c.rejected_reason = "login required for pricing (public pricing only)"
        return c

    # store the markdown for batched Groq extraction (done per-item in matcher).
    # Truncate hard to keep the Groq batch prompt within token limits. A product
    # page's price/pack/variant lives in the first ~1500 chars of main content;
    # huge pages (380k-char category listings) are very unlikely to be product
    # pages — a 2000-char slice of one is just nav/listing junk that pollutes the
    # Groq batch and wastes tokens. Reject those outright. Moderately large pages
    # (a product page padded with reviews/related items) are still kept+truncated,
    # since their price lives in the first ~1500 chars.
    MD_CAP = int(os.environ.get("SCRAPE_MD_CAP", "2000"))
    OVERSIZE = int(os.environ.get("SCRAPE_OVERSIZE_REJECT", "150000"))
    if len(markdown) > OVERSIZE:
        log.info("%soversized page (%d chars) — almost certainly a category/search "
                 "listing, not a product page; skipping extraction", tag, len(markdown))
        c.notes = "skipped — page too large to be a product listing (category/search page)"
        return c
    c.scraped_markdown = markdown[:MD_CAP]
    # Persist immediately so a later freeze/crash never loses this fetched page
    # — a re-run reads it from cache for 0 credits.
    if SCRAPE_CACHE_ENABLED:
        try:
            from . import db
            db.save_scrape(c.url, c.scraped_markdown)
        except Exception:
            pass
    log.info("%sscrape SUCCESS (markdown, stored %d of %d chars) — %s",
             tag, len(c.scraped_markdown), len(markdown), c.url[:60])
    return c