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
import json
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

import requests

from .models import OrderLineItem, PriceCandidate

log = logging.getLogger(__name__)
fc_credit_log = logging.getLogger("firecrawl.credits")

SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "")
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_MAX_SCRAPES = int(os.environ.get("FIRECRAWL_MAX_SCRAPES_PER_RUN", "150"))
_fc_state = {
    "exhausted": False,
    "exhausted_reason": None,
    "scrapes": 0,
    "skipped": 0,
    "credits": 0,
}
_fc_lock = threading.Lock()


def _mark_firecrawl_exhausted(reason: str, *, url: str = "", tag: str = "") -> None:
    with _fc_lock:
        if _fc_state["exhausted"]:
            return
        _fc_state["exhausted"] = True
        _fc_state["exhausted_reason"] = reason
    where = f" (triggered by {tag}{url[:80]})" if url else ""
    if reason == "402":
        fc_credit_log.error(
            "========== FIRECRAWL CREDITS EXHAUSTED (HTTP 402) ==========\n"
            "First failing URL%s",
            where,
        )
    elif reason == "budget":
        fc_credit_log.warning(
            "========== FIRECRAWL SCRAPE BUDGET REACHED ==========\n"
            "Per-run cap FIRECRAWL_MAX_SCRAPES_PER_RUN=%d reached.",
            FIRECRAWL_MAX_SCRAPES,
        )
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
        _fc_state["exhausted_reason"] = None
        _fc_state["scrapes"] = 0
        _fc_state["skipped"] = 0
        _fc_state["credits"] = 0
    _gp_state["disabled"] = False
    _gp_state["failures"] = 0
    _serp_state["exhausted"] = False
    _serp_state["consecutive_failures"] = 0
    _shop_state["disabled"] = False
    _shop_state["zero_streak"] = 0
    log.info("Firecrawl budget reset (max %d scrapes, %d run credits)",
             FIRECRAWL_MAX_SCRAPES, FIRECRAWL_RUN_CREDITS)


def firecrawl_stats() -> dict:
    with _fc_lock:
        return {
            "scrapes": _fc_state["scrapes"],
            "skipped": _fc_state["skipped"],
            "exhausted": _fc_state["exhausted"],
            "exhausted_reason": _fc_state["exhausted_reason"],
            "credits": _fc_state["credits"],
        }


def firecrawl_log_run_summary(order_ref: str | None = None) -> None:
    fc = firecrawl_stats()
    ref = f" ref={order_ref}" if order_ref else ""
    if fc["exhausted"] and fc["exhausted_reason"] == "402":
        fc_credit_log.error(
            "========== FIRECRAWL RUN SUMMARY%s ==========\n"
            "CREDITS EXHAUSTED — %d scraped, %d skipped, %d credits billed.",
            ref, fc["scrapes"], fc["skipped"], fc["credits"],
        )
    elif fc["exhausted"] and fc["exhausted_reason"] == "budget":
        fc_credit_log.warning(
            "========== FIRECRAWL RUN SUMMARY%s ==========\n"
            "Scrape budget reached — %d scraped, %d skipped.",
            ref, fc["scrapes"], fc["skipped"],
        )
    else:
        fc_credit_log.info(
            "Firecrawl run summary%s: %d scrape(s), %d skipped, %d credits billed",
            ref, fc["scrapes"], fc["skipped"], fc["credits"],
        )


CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))

PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")

# Sections that list DIFFERENT products (cross-sell / recommendations). Their
# prices are a major source of wrong extractions — the model grabs a related
# item's price instead of the product's. We cut the markdown at the first such
# heading BEFORE sending it to Groq. NOTE: same-product seller blocks
# ("vendor options", "other sellers", "more buying choices") are deliberately
# NOT listed here — those carry the real (often lowest) price and must stay.
_CROSS_SELL_RE = re.compile(
    r"(?im)^\s*[#>*\-\s]*(?:"
    r"(?:related\s+products?|related\s+items?|related\s+categor\w*|"
    r"you\s+may\s+also\s+like|similar\s+items?|"
    r"best[\s\-]?sellers?\s+(?:related|for)|"
    r"customers\s+(?:who\s+bought|also\s+(?:bought|viewed))|"
    r"people\s+also\s+(?:bought|viewed)|frequently\s+bought\s+together|"
    r"recommended\s+(?:for\s+you|products?|items?)|more\s+like\s+this|"
    r"you\s+might\s+also\s+like)\b"     # phrase at line start, trailing text OK
    r"|related\s*:?\s*$"                 # bare "Related" heading (anchored, safe)
    r")"
)
# Explicit "price is behind login" phrasing — must reject EVEN IF other "$"
# values appear elsewhere on the page (promos, related items). This is what the
# old `"$" not in markdown` heuristic missed (e.g. dhpsupply "Login to see price").
_LOGIN_PRICE_RE = re.compile(
    r"(?i)(?:log\s?in|login|sign\s?in|register|create\s+an?\s+account)\s+"
    r"(?:to|for|and|to\s+view|to\s+see)?\s*(?:see|view|reveal|unlock|get)?\s*"
    r"(?:the\s+)?(?:price|pricing|cost|your\s+price|net\s+price|member\s+price)"
)

# HARD GATE: phrasing that means the product's price slot itself is hidden, so any
# "$" left on the page is a promo, related item, or metadata — NOT the buy price.
# Two shapes:
#   (a) "log in TO SEE / VIEW / REVEAL the price"        (dhpsupply Clinpro)
#   (b) "sign in / log in FOR (your/our) pricing"        (midwestdental floss)
# Deliberately does NOT match qualified upsell phrasing — "...for your CONTRACT /
# MEMBER / SPECIAL / WHOLESALE / NET / BETTER price" — where a public list price is
# normally shown right alongside and should still be used.
_LOGIN_HARDGATE_RE = re.compile(
    r"(?i)(?:log\s?in|login|sign\s?in|register)\s+"
    r"(?:"
    r"(?:to|and)?\s*(?:see|view|reveal|unlock|check)\s+(?:the\s+|our\s+)?(?:price|pricing|cost)"
    r"|for\s+(?:your\s+|our\s+)?(?!contract|member|special|wholesale|net|better|account)"
    r"(?:price|pricing|cost)"
    r")\b"
)


def _strip_cross_sell(markdown: str) -> str:
    """Cut markdown at the first 'related products / similar items / customers
    also bought' heading so cross-sell prices never reach the extractor. Keeps
    same-product seller sections (vendor options / other sellers) intact. Only
    cuts when the heading appears comfortably past the buy box (> MIN chars) so
    we never truncate the main price block itself."""
    if not markdown:
        return markdown
    MIN = int(os.environ.get("CROSS_SELL_MIN_KEEP", "400"))
    earliest = None
    for m in _CROSS_SELL_RE.finditer(markdown):
        if m.start() >= MIN:
            earliest = m.start()
            break
    if earliest is not None:
        return markdown[:earliest]
    return markdown


# ---- Deterministic multi-seller (aggregator) price parser -------------------
# Net32 and SupplyClinic list MANY sellers for the SAME product in a table. The
# correct price is the LOWEST AVAILABLE seller's landed total (price + shipping),
# NOT the big headline "$X/ea". A small free-tier model cannot reliably scan such
# a table for the minimum (this is exactly why net32 kept returning the headline
# /ea instead of the Lowest-Price row), so we parse it deterministically here and
# LOCK the price. Client rule: "Special Order" counts as AVAILABLE; only
# out-of-stock / backordered rows are excluded. Falls back to the LLM (returns
# None) when the page isn't a parseable seller table.
_AGG_DOMAINS = ("net32.com", "supplyclinic.com")
_OOS_RE = re.compile(
    r"(out\s+of\s+stock|back[\s\-]?order(?:ed)?|sold\s+out|unavailable|"
    r"notify\s+me|discontinued|no\s+longer\s+available)", re.I)
_STRIKE_RE = re.compile(r"~~\s*\$\s?\d[\d,]*(?:\.\d{2})?\s*~~")
# Net32 renders the per-seller price as "**$19.00**/ea" — markdown bold puts a
# `**` between the number and "/ea", which broke the old `\.\d{2}\s*/\s*ea`
# anchor (it matched ZERO rows on every net32 page, forcing the fragile legacy
# fallback). Tolerate optional bold/space before the unit so _net32_options and
# headline detection actually engage.
_EA_RE = re.compile(r"\$\s?([\d,]+\.\d{2})\s*\*{0,2}\s*/\s*ea", re.I)
_SHIP_ADD_RE = re.compile(r"\+\s*\$\s?([\d,]+\.\d{2})\b")
# Net32 quantity-break sub-rows ("1 @ $28.13, 2 @ $21.25, 4 @ $18.53") are BULK
# tier prices, never the single-unit price. The legacy any-$ fallback would
# otherwise read the deepest tier ($18.53) as the seller's price. Strip them.
_QTY_BREAK_RE = re.compile(r"\d+\s*@\s*\$\s?[\d,]+\.\d{2}")


def _agg_include_shipping() -> bool:
    """Whether to fold flat-rate shipping into the reported aggregator price.
    Default OFF: report the ITEM/LIST price so it compares apples-to-apples with
    Schein's ex-shipping unit price. SupplyClinic's '+ $5.12 Flat Rate Shipping'
    was being added in, turning $7.93 masks into $13.05 and $49.54 zirconia into
    $54.66. Set AGG_INCLUDE_SHIPPING=1 to restore landed-total behaviour."""
    return os.environ.get("AGG_INCLUDE_SHIPPING", "0") == "1"
_FREE_SHIP_RE = re.compile(r"free\s+(?:standard\s+)?shipping|free\s+ship", re.I)
_LOWEST_RE = re.compile(r"lowest\s+price", re.I)
# Net32 glues the badge straight onto the lowest seller's Total ("$26.73LOWEST
# PRICE"). Reading THIS literal is far more robust than inferring the badge from
# per-row parsing — the row can be dropped for unrelated reasons (a neighbouring
# out-of-stock note, a missing /vendor/ anchor), which is exactly how XLight's
# badged $26.73 kept getting lost no matter how the row scan was tuned. A price
# is only ever written immediately before "LOWEST PRICE" on the genuinely badged
# row, so this can't false-match the generic "Lowest Price guarantee" blurb
# (no price precedes that one).
_BADGE_PRICE_RE = re.compile(r"\$\s?([\d,]+\.\d{2})\s*LOWEST\s+PRICE", re.I)
_PRICE_NUM_RE = re.compile(r"\$\s?([\d,]+\.\d{2})")
# Per-seller availability marker (one per row) — used to segment SupplyClinic's
# LIST PRICE table when 'Add to Cart' button text is dropped by the scraper.
_AVAIL_RE = re.compile(
    r"(?i)\b(in\s+stock|in-stock|special\s+order|available|"
    r"back[\s\-]?order(?:ed)?|out\s+of\s+stock|sold\s+out|unavailable|discontinued)\b")


def _to_f(s) -> Optional[float]:
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _aggregator_domain(url: str) -> Optional[str]:
    d = _domain(url)
    for a in _AGG_DOMAINS:
        if d == a or d.endswith("." + a):
            return a
    return None


def _seller_rows(section: str) -> list:
    """Split a vendor/seller table into per-seller chunks. Every seller row ends
    in an 'Add to Cart' affordance on both Net32 and SupplyClinic, so we split on
    that and drop the trailing page-footer fragment after the last one."""
    parts = re.split(r"(?i)add\s+to\s+cart", section)
    return [p for p in parts[:-1] if p.strip()] if len(parts) > 1 else []


def _vendor_section(markdown: str, dom: Optional[str] = None) -> str:
    """Slice from the vendor/seller-table heading to the first cross-sell/footer
    heading so neighbouring product prices never enter the min().

    NET32 EXCEPTION: net32 linearizes a "Similar items" sidebar INTO THE MIDDLE
    of its seller list (between the featured buy-box rows and the "See All
    Options" expansion). Cutting at that first cross-sell heading truncated the
    table and dropped sellers net32 deprioritizes to the bottom — notably the
    genuine LOWEST when it has a long handling time (XLight: Shine $26.73 ships
    in 30+ days, so net32 buries it past the sidebar, and the cut lost it →
    locked $41.91 instead). For net32 we therefore DON'T apply the cross-sell
    cut: badge-trust already returns the per-seller "Lowest Price" row first, so
    pages whose lowest is in the featured block are unaffected, while pages whose
    lowest is buried now keep it. Tunable via NET32_FULL_SECTION (default on)."""
    low = markdown.lower()
    starts = [low.find(h) for h in
              ("vendor options", "seller options", "other sellers", "all options",
               "price + shipping", "list price")
              if h in low]
    start = min([s for s in starts if s >= 0], default=0)
    tail = markdown[start:]
    if dom == "net32.com" and os.environ.get("NET32_FULL_SECTION", "1") == "1":
        return tail                                   # keep buried lowest sellers
    m = _CROSS_SELL_RE.search(tail)
    return tail[:m.start()] if m else tail


def _net32_vendor(section: str, vstart: int) -> str:
    """Vendor display name from a '/vendor/<slug>/<id>' link start, e.g.
    '/vendor/shine-dental/154' → 'Shine Dental'. '' when not resolvable."""
    if vstart is None or vstart < 0:
        return ""
    seg = section[vstart + len("/vendor/"): vstart + 80].split("/", 1)[0]
    seg = seg.strip().strip(")").strip("(")
    return " ".join(w.capitalize() for w in seg.replace("-", " ").split()) if seg else ""


def _net32_options(section: str):
    """((landed_total, has_lowest_badge) per AVAILABLE seller, [excluded backorder]).
    Anchors on each '$X.XX/ea' (exactly one per seller) instead of an 'Add to
    Cart' delimiter — Firecrawl often drops button text, which left the old
    split-based parser with zero rows. The stock note PRECEDES a seller's price
    (after its name); shipping + 'Lowest Price' badge FOLLOW it, so look-behind
    is bounded to the previous price (only this seller's stock) and look-ahead to
    the next price. Client rule: 'Special Order' is AVAILABLE — _OOS_RE excludes
    only out-of-stock / backordered / sold-out / discontinued rows. Those excluded
    rows are RETURNED SEPARATELY (price, status, vendor) so the report can show the
    cheapest one as a labelled secondary row rather than dropping it silently."""
    eas = list(_EA_RE.finditer(section))
    opts, excluded = [], []
    for i, m in enumerate(eas):
        base = _to_f(m.group(1))
        if base is None:
            continue
        prev_end = eas[i - 1].end() if i > 0 else 0
        nxt = eas[i + 1].start() if i + 1 < len(eas) else len(section)
        # Bound the stock look-behind to THIS seller's OWN block. Each net32
        # seller row starts with a "/vendor/<name>" link, then its stock note,
        # then its "/ea" price. Scanning all the way back to the previous price
        # let a foreign "Backordered"/"Out of Stock" — from a price-less row or
        # the interleaved "Similar items" sidebar sitting between two priced
        # sellers — bleed in and wrongly drop a valid seller. That is exactly why
        # XLight's Shine $26.73 (the genuine lowest, matched by _EA_RE) never
        # reached options: an out-of-stock row upstream poisoned its look-behind.
        vstart = section.rfind("/vendor/", prev_end, m.start())
        b0 = vstart if vstart != -1 else max(prev_end, m.start() - 400)
        behind = section[b0:m.start()]                        # this seller's name + stock
        ahead = section[m.start():min(nxt, m.start() + 220)]  # shipping
        if _FREE_SHIP_RE.search(ahead):
            ship = 0.0
        else:
            sm = _SHIP_ADD_RE.search(ahead)
            ship = (_to_f(sm.group(1)) or 0.0) if sm else 0.0
        total = round(base + (ship if _agg_include_shipping() else 0.0), 2)
        if total <= 0:
            continue
        oos = _OOS_RE.search(behind)
        if oos:                                               # unavailable → capture, don't lock
            excluded.append({"price": total,
                             "status": oos.group(0).strip().rstrip(".").title(),
                             "vendor": _net32_vendor(section, vstart)})
            continue
        opts.append((total, bool(_LOWEST_RE.search(section[m.start():nxt]))))
    return opts, excluded


def _supplyclinic_options(section: str) -> list:
    """(landed_total, has_lowest_badge) per seller from a SupplyClinic LIST PRICE
    table. Anchors on each availability marker (one per seller); the current
    price follows it. Struck-through 'was' originals are dropped; flat-rate
    shipping is added, not read as the price. 'Special Order' counts as
    available; backordered / out-of-stock rows are skipped."""
    avs = list(_AVAIL_RE.finditer(section))
    opts = []
    for i, m in enumerate(avs):
        if _OOS_RE.search(m.group(0)):              # this seller's availability is OOS
            continue
        nxt = avs[i + 1].start() if i + 1 < len(avs) else len(section)
        block = _STRIKE_RE.sub(" ", section[m.start():nxt])   # drop struck originals
        if _FREE_SHIP_RE.search(block):
            ship = 0.0
        else:
            sm = _SHIP_ADD_RE.search(block)
            ship = (_to_f(sm.group(1)) or 0.0) if sm else 0.0
        clean = _SHIP_ADD_RE.sub(" ", block)        # so flat-rate isn't read as the price
        nums = [n for n in (_to_f(x) for x in _PRICE_NUM_RE.findall(clean)) if n and n > 0]
        if not nums:
            continue
        opts.append((round(min(nums) + (ship if _agg_include_shipping() else 0.0), 2),
                     bool(_LOWEST_RE.search(block))))
    return opts


def _legacy_seller_options(section: str) -> list:
    """Fallback: original 'Add to Cart'-split row parser → [(total, badge)]."""
    options = []
    for row in _seller_rows(section):
        if _OOS_RE.search(row):
            continue
        clean = _STRIKE_RE.sub(" ", row)
        clean = _QTY_BREAK_RE.sub(" ", clean)       # drop "4 @ $18.53" bulk tiers
        ea = _EA_RE.search(clean)
        if ea:
            base = _to_f(ea.group(1))
            if base is None:
                continue
            if _FREE_SHIP_RE.search(clean):
                ship = 0.0
            else:
                sm = _SHIP_ADD_RE.search(clean)
                ship = (_to_f(sm.group(1)) or 0.0) if sm else 0.0
            total = round(base + (ship if _agg_include_shipping() else 0.0), 2)
        else:
            sm = _SHIP_ADD_RE.search(clean)
            ship = (_to_f(sm.group(1)) or 0.0) if (sm and not _FREE_SHIP_RE.search(clean)) else 0.0
            clean_no_ship = _SHIP_ADD_RE.sub(" ", clean)
            nums = [n for n in (_to_f(x) for x in _PRICE_NUM_RE.findall(clean_no_ship))
                    if n and n > 0]
            if not nums:
                continue
            total = round(min(nums) + (ship if _agg_include_shipping() else 0.0), 2)
        if total and total > 0:
            options.append((total, bool(_LOWEST_RE.search(row))))
    return options


_agg_debug_lock = threading.Lock()


def _agg_debug_dump(url, dom, section, markdown, options, backorder,
                    headline, badge_price, min_total, result) -> None:
    """AGG_DEBUG=1 artifact writer. Dumps everything the seller-table parser saw
    and decided for ONE aggregator page so a wrong lock can be traced offline:
      <dom>_<slug>_<hash>.json   — parsed result + options/backorder/badge/headline
                                   + raw _EA_RE and _BADGE_PRICE_RE matches
      <...>.section.md           — the vendor section the parser actually scanned
      <...>.full.md              — the full markdown handed to the parser
    Files are keyed by a URL hash, so a re-run overwrites the same page's dump.
    Best-effort: never raises into the scrape path."""
    try:
        import hashlib
        from .paths import resolve_agg_debug_dir
        dbg_dir = resolve_agg_debug_dir()
        dbg_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:10]
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", (url or "").split("//")[-1])[:60].strip("-")
        base = dbg_dir / f"{dom}_{slug}_{h}"
        meta = {
            "url": url,
            "domain": dom,
            "returned_result": result,
            "headline_per_ea": headline,
            "badge_price": badge_price,
            "min_total": min_total,
            "options_(total,has_badge)": options,
            "excluded_backorder": backorder,
            "ea_matches": [m.group(0) for m in _EA_RE.finditer(section)],
            "badge_matches": [m.group(0) for m in _BADGE_PRICE_RE.finditer(section)],
            "section_len": len(section or ""),
            "markdown_len": len(markdown or ""),
        }
        # NOTE: with_name(+suffix), NOT with_suffix — the domain's own dot
        # (net32.com) would make with_suffix(".json") clobber everything after it,
        # collapsing every net32 page to one colliding "net32.json".
        with _agg_debug_lock:
            base.with_name(base.name + ".json").write_text(json.dumps(meta, indent=2, default=str))
            base.with_name(base.name + ".section.md").write_text(section or "")
            base.with_name(base.name + ".full.md").write_text(markdown or "")
        log.info("AGG_DEBUG wrote %s.{json,section.md,full.md} (returned $%s)",
                 base.name, (result or {}).get("price"))
    except Exception as e:
        log.warning("AGG_DEBUG dump failed for %s: %s", (url or "")[:60], e)


def parse_aggregator_price(url: str, markdown: str) -> Optional[dict]:
    """Pick the lowest AVAILABLE seller's landed total (price + shipping) from a
    Net32 / SupplyClinic multi-seller table. Returns
    {'price', 'lowest_badge_matched', 'n_rows'} or None when the page isn't a
    parseable aggregator table (caller then falls back to the LLM). Anchor-based
    per-seller parsing (robust to dropped 'Add to Cart' text) runs first; the
    older split-based parser is the final fallback."""
    dom = _aggregator_domain(url)
    if not dom or not markdown:
        return None
    section = _vendor_section(markdown, dom)
    backorder = []
    if dom == "net32.com":
        options, backorder = _net32_options(section)
    elif dom == "supplyclinic.com":
        options = _supplyclinic_options(section)
    else:
        options = []
    if not options:                                  # robustness fallback
        options = _legacy_seller_options(section)
    if not options:
        return None
    min_total = min(t for t, _ in options)
    badge = [t for t, lo in options if lo]
    badge_price = min(badge) if badge else None
    # DIRECT BADGE RESCUE (net32 only): net32 prints the lowest seller's Total
    # with the tag glued straight on ("$26.73LOWEST PRICE"). Reading that literal
    # recovers a badged in-stock row that per-row parsing dropped by mistake.
    # BUT net32 badges the lowest price even when that seller is BACKORDERED
    # (XLight: Shine $26.73 is "Backordered.Ships ... Jul 16"), which the client
    # rule EXCLUDES. So only trust the badge when the badged seller's OWN block
    # (back to its /vendor/ link) is not out-of-stock — otherwise the rescue
    # would wrongly resurrect a backordered price as the lock.
    if dom == "net32.com":
        direct = []
        for bm in _BADGE_PRICE_RE.finditer(section):
            p = _to_f(bm.group(1))
            if not p or p <= 0:
                continue
            vstart = section.rfind("/vendor/", 0, bm.start())
            blk = section[(vstart if vstart != -1 else max(0, bm.start() - 400)):bm.start()]
            if _OOS_RE.search(blk):
                continue                   # badged seller is backordered → ignore
            direct.append(p)
        if direct:
            d = min(direct)
            badge_price = d if badge_price is None else min(badge_price, d)
    # Headline = the featured buy-box "$X/ea" at the top of the page. It is NOT
    # the lowest seller, but it bounds plausibility: a real lowest sits between
    # ~30% and ~105% of it. Used to reject stray sub-unit / sidebar numbers.
    hm = _EA_RE.search(markdown)
    headline = _to_f(hm.group(1)) if hm else None

    def _plausible(p) -> bool:
        if p is None or p <= 0:
            return False
        if headline and not (0.30 * headline <= p <= 1.05 * headline):
            return False
        return True

    # Build the final result (precedence UNCHANGED — badge first, then a
    # plausible sub-floor row, then the raw min). Computed into `result` so the
    # AGG_DEBUG dump below can record exactly what was returned alongside the raw
    # rows that produced it.
    result = None
    # Net32/SupplyClinic explicitly badge the single lowest AVAILABLE seller as
    # "LOWEST PRICE". That badge is authoritative — more reliable than our own
    # min(), which a quantity-break tier ("4 @ $18.53") or a linearized "Similar
    # items" sidebar can drag off. So when a plausible badge exists, report IT,
    # even if our parsed min disagrees (the old code did the opposite and lost
    # the correct price whenever min was polluted).
    if badge_price is not None and _plausible(badge_price):
        result = {"price": badge_price, "lowest_badge_matched": True,
                  "n_rows": len(options), "badge_price": badge_price,
                  "backorder_options": backorder}
    # No usable badge → min of the table, but drop rows that fall below the
    # plausibility floor (sub-unit / sidebar artifacts like $6.01 under a $14.94
    # buy box) when at least one plausible row remains.
    if result is None and headline:
        plausible_rows = sorted(t for t, _ in options if _plausible(t))
        if plausible_rows and min_total < 0.30 * headline:
            result = {"price": plausible_rows[0], "lowest_badge_matched": False,
                      "n_rows": len(options), "badge_price": badge_price,
                      "backorder_options": backorder}
    if result is None:
        result = {"price": min_total, "lowest_badge_matched": False,
                  "n_rows": len(options), "badge_price": badge_price,
                  "backorder_options": backorder}

    # DEBUG (AGG_DEBUG=1): write a full per-page artifact (parsed result + every
    # seller row + the FULL markdown the parser saw) to output/agg_debug/, so a
    # wrong lock (e.g. net32 Fuji $209 truth → $174 returned) can be traced
    # offline against exactly what the scraper produced vs. what the page renders.
    if os.environ.get("AGG_DEBUG") == "1":
        _agg_debug_dump(url, dom, section, markdown, options, backorder,
                        headline, badge_price, min_total, result)
    return result


def _apply_aggregator_lock(c: PriceCandidate, full_markdown: str, tag: str = "") -> bool:
    """If the candidate URL is a Net32/SupplyClinic seller table, set its price
    deterministically from the lowest available seller and LOCK it so the AI
    layer won't overwrite it. Returns True when a price was locked."""
    try:
        agg = parse_aggregator_price(c.url, full_markdown)
    except Exception as e:                         # parser must never break a scrape
        log.warning("%saggregator parse error (%s) — falling back to AI for %s",
                    tag, e, c.url[:60])
        return False
    if not agg:
        return False
    c.price = agg["price"]
    c.in_stock = True
    c.price_locked = True
    c.backorder_options = agg.get("backorder_options") or []
    src = "Lowest Price row" if agg["lowest_badge_matched"] else "min of Total column"
    note = (f"price set from {_domain(c.url)} seller table: lowest available "
            f"(incl. shipping) of {agg['n_rows']} seller(s), via {src}")
    c.notes = ((c.notes + " · ") if c.notes else "") + note
    # Transparency flag: a "Lowest Price" badge was detected but we ended up
    # pricing a DIFFERENT (lower) row. That's sometimes legitimate — a seller
    # with a higher /ea but free shipping can beat the badged /ea on landed
    # total — but it's also the signature of a stray parse, so surface it.
    bp = agg.get("badge_price")
    if bp and c.price + 0.02 < bp:
        c.notes += (f" · NOTE: net32/supplyclinic 'Lowest Price' row was ${bp:.2f}; "
                    f"parsed lowest landed is ${c.price:.2f} — verify")
    log.info("%saggregator price LOCKED $%.2f (%d sellers, badge_match=%s) — %s",
             tag, c.price, agg["n_rows"], agg["lowest_badge_matched"], c.url[:60])
    return True


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
    for line in f.read_text(encoding="utf-8").splitlines():
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
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip().lower()
        if line:
            out.append(line)
    return out


SUPPLIER_SITES = load_supplier_sites()


def load_seed_urls() -> dict:
    """Per-SKU 'known-good' product URLs (config/seed_urls.txt, lines 'SKU | url').
    These are injected as candidates and scraped DIRECTLY every run regardless of
    discovery or cache — a guaranteed backstop (Option 2) for supplier pages no
    search engine indexes. Built once from a manual-search sheet; purely additive,
    never required."""
    f = CONFIG_DIR / "seed_urls.txt"
    out: dict = {}
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or "|" not in line:
            continue
        sku, url = line.split("|", 1)
        sku, url = sku.strip(), url.strip()
        if sku and url.startswith("http"):
            out.setdefault(sku, []).append(url)
    return out


SEED_URLS = load_seed_urls()

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
# Aggregator (net32 / supplyclinic) pages are LIVE marketplaces: the lowest-seller
# price and the "Lowest Price" badge move intraday as sellers reprice, so even a
# few-hours-old snapshot can be off by a few cents (e.g. cached $39.47 vs live
# $39.35). Give them their own, much shorter TTL so the deterministic lock tracks
# the live page. Default 0 = never serve a cached aggregator page; always re-scrape.
AGG_SCRAPE_CACHE_HOURS = int(os.environ.get("AGG_SCRAPE_CACHE_HOURS", "0"))
SCRAPE_CACHE_ENABLED = os.environ.get("SCRAPE_CACHE", "1") not in ("0", "false", "False")


def load_discovery_cache(conn) -> None:
    """Populate the in-memory discovery cache from the DB once per run."""
    with _disc_lock:
        _disc_cache.clear(); _disc_new.clear()
    if not (DISCOVERY_CACHE_ENABLED and conn):
        return
    # One-shot purge hook: DISCOVERY_PURGE_ALL=1 wipes the whole discovery cache;
    # DISCOVERY_PURGE="sku1,sku2" clears specific SKUs. Purged SKUs (and their
    # cached scraped pages) are removed BEFORE loading, so they're re-discovered
    # and re-scraped fresh this run — the clean fix for a pinned wrong-variant URL.
    purge_all = os.environ.get("DISCOVERY_PURGE_ALL", "0") not in ("0", "false", "False", "")
    purge_skus = [s for s in re.split(r"[,\s]+", os.environ.get("DISCOVERY_PURGE", "").strip()) if s]
    if purge_all or purge_skus:
        try:
            from . import db
            nd, ns = db.purge_discovery(conn, None if purge_all else purge_skus)
            log.info("Discovery cache PURGE (%s): removed %d discovery row(s) and "
                     "%d cached page(s) — these will be re-discovered this run",
                     "ALL" if purge_all else ",".join(purge_skus), nd, ns)
        except Exception as e:
            log.warning("discovery purge failed: %s", e)
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


def purge_discovery_cache(conn, skus=None) -> tuple[int, int]:
    """Programmatic purge: clear cached discovery (+ scraped pages) for skus (or
    ALL when skus is None) and drop them from the in-memory cache. Returns
    (discovery_rows_deleted, scrape_rows_deleted)."""
    from . import db
    nd, ns = db.purge_discovery(conn, skus)
    with _disc_lock:
        for s in (list(skus) if skus else list(_disc_cache.keys())):
            _disc_cache.pop(s, None)
    return nd, ns


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
        q = (urlparse(url).query or "").lower()
        pl = path.lower()
        # A "/c/<id>" or "/s/<id>" segment is a HARD category-listing signal
        # (Hybris/SFCC category pages — e.g. tdsc .../Fluoride-Products/c/215).
        # The old test exempted this as "product detail" because the CATEGORY
        # name "Fluoride-Products" contains the substring "product"; that mistook
        # a listing of many products for one product page and priced the first
        # (wrong) tile on it.
        if re.search(r"/[cs]/\d+(?:/|$|\?)", pl):
            return True
        # Otherwise treat as a product-detail page ONLY on a real product-path
        # token or an id query param — never on a bare "product(s)" substring
        # that may just be part of a category name.
        product_detail = (
            re.search(r"/(?:products?|item|prod|dp)/[^/]+", pl)
            or re.search(r"/p/[\w-]*\d", pl)
            or "pid=" in q or "sku=" in q or "productid=" in q or "itemid=" in q)
        if not product_detail:
            return True
    if re.search(r"-[tl]-\d+(?:-\d+)*/?$", path):
        return True                       # generic -t-/-l- taxonomy slugs
    if "?q=" in url or "search" in (urlparse(url).query or "").lower():
        return True
    return False


_BAD_URL_RE = re.compile(r"/goto\?|/redirect\?|[?&]url=", re.I)
# Domains that are clearly not dental suppliers (appeared as junk candidates):
# generic marketplaces, churches, unrelated retailers. Snippet prices from these
# are noise (vintage jars, etc.). Extend as needed via config later.
_OFFTOPIC_DOMAINS = {
    "etsy.com", "pinterest.com", "youtube.com", "facebook.com", "reddit.com",
    "stadtkirche-michelstadt.de", "shelllumber.com",
}


def usable(url: str) -> bool:
    if not url or is_excluded(url) or DOC_EXT_RE.search(url) or is_category_url(url):
        return False
    if _BAD_URL_RE.search(url):
        return False                       # /goto redirect / tracking wrapper, not a product page
    d = _domain(url)
    if any(d == o or d.endswith("." + o) for o in _OFFTOPIC_DOMAINS):
        return False                       # clearly non-dental-supplier domain
    return True


def _looks_supplier(domain: str) -> bool:
    return (domain in KNOWN_SUPPLIERS or domain in SUPPLIER_SITES
            or any(domain.endswith(s) for s in SUPPLIER_SITES) or "dent" in domain)


def is_trusted_supplier(url: str) -> bool:
    """True when a candidate URL is on the client's trusted supplier list
    (config/supplier_sites.txt). Used to scrape these pages first within the
    per-item scrape cap, so a known supplier's price is never crowded out by a
    cheaper-looking (often wrong-variant) generic snippet. Matching mirrors the
    gap-sweep: exact, sub-domain, or parent-domain (shop.benco ↔ benco)."""
    d = _domain(url)
    if not d:
        return False
    return any(d == s or d.endswith("." + s) or s.endswith(d) for s in SUPPLIER_SITES)


# --- discovery relevance filter ------------------------------------------------
# SerpAPI sometimes returns wholly unrelated products from a supplier (Clinpro
# fluoride discovery pulled 17 crazydental CROWN URLs). Those waste shortlist and
# scrape slots. This guard drops a URL ONLY when its slug clearly describes a
# DIFFERENT product, and is deliberately conservative so valid differently-worded
# equivalents (e.g. a generic "enzymatic cleaner" for "ReSURGE … Cleaning
# Solution") are never dropped.
DISCOVERY_RELEVANCE_FILTER = os.environ.get("DISCOVERY_RELEVANCE_FILTER", "1") not in ("0", "false", "False")
_REL_STOP = {"with", "and", "for", "the", "pack", "each", "box", "case", "bottle",
             "tube", "tubes", "unit", "dose", "size", "pkg", "count", "qty",
             "dental", "refill", "new", "item", "product", "products", "shop",
             "category", "treatment", "supply", "supplies"}


def _rel_tok(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if len(t) >= 4 and not t.isdigit() and t not in _REL_STOP}


def _anchor_tokens(item) -> set:
    parts = " ".join(str(x or "") for x in (
        getattr(item, "product_name", ""), getattr(item, "brand", ""),
        getattr(item, "variant", ""), item.description))
    return _rel_tok(parts)


def _tok_match(a: str, b: str) -> bool:
    # exact, or share a 5-char prefix so "cleaner"~"cleaning", "glove"~"gloves"
    return a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])


def _url_relevant(url: str, item, anchors: Optional[set] = None) -> bool:
    """Keep a discovered URL unless its slug clearly describes a DIFFERENT product.
    Keeps: opaque/SKU slugs, and anything sharing a product word with the order
    (prefix-relaxed). Drops only: slugs with >=2 real product words, none of which
    match the order — e.g. 'Ion-Polycarbonate-Crowns-P22' for a Clinpro item."""
    if not DISCOVERY_RELEVANCE_FILTER:
        return True
    anchors = anchors if anchors is not None else _anchor_tokens(item)
    if len(anchors) < 2:
        return True                                  # too little to judge → keep
    path = re.sub(r"https?://[^/]+", "", url).split("?")[0]   # slug only, no host/query
    slug = _rel_tok(path)
    if len(slug) < 2:
        return True                                  # opaque / one-word slug → keep
    if any(_tok_match(s, a) for s in slug for a in anchors):
        return True                                  # shares a product word → keep
    return False                                     # >=2 real words, zero overlap → drop


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
            from .jobs import emit
            emit("serpapi", action="search",
                 engine=params.get("engine", "google"),
                 query=(params.get("q") or "")[:120])
            return r.json()
        except requests.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            # 5xx is a transient server-side wall — stop retrying this call but let
            # it count toward the give-up breaker below (a sustained 5xx storm then
            # trips `exhausted` instead of every item grinding the full retry loop).
            # 4xx (bad query etc.) stays an immediate raise — retrying won't help.
            if status and status >= 500:
                last_err = e
                break
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
_shop_lock = threading.Lock()   # guards _shop_state under the parallel item workers
SHOP_DISABLE_AFTER = int(os.environ.get("GOOGLE_SHOPPING_DISABLE_AFTER", "3"))


def _resolve_google_product(product_id: str, cands: list, seen: set,
                            max_sellers: int = 5) -> int:
    """SerpAPI google_product sellers lookup: turns a google.com product_link
    into direct merchant links with prices. This recovers the entire Google
    Shopping channel (previously 100% discarded as google.com URLs)."""
    if _gp_state["disabled"]:
        return 0
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


SUPPLIER_RESERVE_SLOTS = int(os.environ.get("SUPPLIER_RESERVE_SLOTS", "6"))


def _shortlist(cands: List[PriceCandidate], max_candidates: int) -> List[PriceCandidate]:
    """≤2 per domain; priced cheapest-first; reserved slots for price-less
    snippets on dental-supplier domains (verified on-page later).

    The reserve prioritises TRUSTED supplier_sites.txt domains over generic
    'looks-like-a-supplier' domains, and is sized by SUPPLIER_RESERVE_SLOTS
    (default 6, up from 3) so a known supplier's price-less listing can't be
    crowded out of the shortlist by cheaper-looking generic snippets — the page
    is priced on-scrape, and the scrape cap (separate) still bounds cost."""
    # SEEDS: manually-curated known-good URLs — always kept, exempt from the
    # ≤2-per-domain cap (otherwise two cheap WRONG listings on the same store can
    # crowd out the curated correct one before it is ever scraped — the Oral-B
    # crazydentalprices case: $6.20 Healthy Gums + $11.25 trial buried the seeded
    # $44.99 Pro-Health). Seeds lead the shortlist and never count against the cap.
    seeds = [c for c in cands if getattr(c, "_seed", False)]
    seed_urls = {c.url for c in seeds}
    by_dom: dict[str, int] = {}
    capped: List[PriceCandidate] = []
    for c in sorted(cands, key=lambda c: (c.price is None, c.price or 0)):
        if c.url in seed_urls:
            continue                      # handled separately, always kept
        d = _domain(c.url)
        if by_dom.get(d, 0) >= 2:
            continue
        by_dom[d] = by_dom.get(d, 0) + 1
        capped.append(c)
    priced = [c for c in capped if c.price is not None]
    priceless_supplier = [c for c in capped if c.price is None and _looks_supplier(_domain(c.url))]
    # trusted supplier_sites.txt domains first, then generic 'dent'/known suppliers
    priceless_supplier.sort(key=lambda c: 0 if is_trusted_supplier(c.url) else 1)
    reserve = priceless_supplier[:SUPPLIER_RESERVE_SLOTS]
    body = priced[:max(0, max_candidates - len(reserve) - len(seeds))] + reserve
    out = seeds + [c for c in body if c.url not in seed_urls]
    return out[:max(max_candidates, len(seeds))]


def _supplier_site_sweep(item: OrderLineItem, cands: list, seen: set,
                         domains: list | None = None,
                         max_sites: int = int(os.environ.get("SUPPLIER_SWEEP_MAX_SITES", "10"))) -> int:
    """Run a targeted `site:domain <query>` search for each trusted supplier so
    their product pages are pulled in even when absent from generic SerpAPI
    results. One organic call per domain (cheap), capped at max_sites.

    This is the Firecrawl-INDEPENDENT fallback for supplier discovery: it spends
    SerpAPI's OWN credit pool, so the trusted suppliers in supplier_sites.txt (the
    manual-sheet niche shops — curio, thedentalmarket, dentaldash, …) are still
    discovered when the Firecrawl gap-sweep can't run because Firecrawl is
    exhausted. `domains` defaults to the whole list; the fallback passes only the
    GAP (suppliers not already in the pool) so no call is wasted on a covered one."""
    if not SERPAPI_KEY or not SUPPLIER_SITES:
        return 0
    sites = domains if domains is not None else SUPPLIER_SITES
    if not sites:
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
    swept = sites[:max_sites]
    for domain in swept:
        if _serp_state["exhausted"]:
            break
        try:
            data = _serpapi({"engine": "google",
                             "q": f"site:{domain} {base}",
                             "gl": "us", "hl": "en", "num": 5})
            n = _collect(data.get("organic_results"), f"site:{domain}", cands, seen)
            added += n
        except Exception as e:
            log.debug("SKU %s — site:%s search failed: %s", item.schein_sku, domain, e)
    log.info("SKU %s — trusted-supplier site: sweep (SerpAPI fallback): +%d "
             "candidate(s) from %d/%d gap site(s)",
             item.schein_sku, added, len(swept), len(sites))
    return added


_storefront_cache: dict = {}   # domain -> [{title,url,price}] | None (per-run, lazy)
STOREFRONT_ENABLED = os.environ.get("STOREFRONT_FEED", "1") not in ("0", "false", "False", "")
STOREFRONT_MAX_SITES = int(os.environ.get("STOREFRONT_MAX_SITES", "12"))
_STORE_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _shopify_products(domain: str):
    """Fetch a Shopify store's PUBLIC catalog feed (/products.json — no auth, free)
    once per run, cached. Returns [{title,url,price}] or None when the store isn't
    Shopify / the feed is unavailable. Finds products on tiny shops that no
    external search engine indexes."""
    if domain in _storefront_cache:
        return _storefront_cache[domain]
    products = None
    try:
        r = requests.get(f"https://{domain}/products.json?limit=250",
                         headers={"User-Agent": _STORE_UA, "Accept": "application/json"},
                         timeout=(8, 20), allow_redirects=True)
        if r.status_code == 200 and "json" in (r.headers.get("content-type", "").lower()):
            items = (r.json() or {}).get("products")
            if isinstance(items, list):
                products = []
                for p in items:
                    handle, title = p.get("handle"), p.get("title")
                    if not (handle and title):
                        continue
                    price = None
                    for v in (p.get("variants") or []):
                        price = _to_price(v.get("price"))
                        if price:
                            break
                    products.append({"title": str(title), "price": price,
                                     "url": f"https://{domain}/products/{handle}"})
    except Exception:
        products = None
    _storefront_cache[domain] = products
    return products


def _storefront_feed_search(item: OrderLineItem, cands: list, seen: set,
                            domains: list, max_sites: int = STOREFRONT_MAX_SITES) -> int:
    """Final discovery fallback: for trusted suppliers STILL missing after organic
    + Firecrawl gap-sweep + SerpAPI site: search, query the store's OWN storefront
    catalog (Shopify /products.json today) and fuzzy-match the ordered product.
    Free plain GETs, cached per domain — the only path that reaches tiny shops no
    search engine indexes."""
    if not STOREFRONT_ENABLED or not domains:
        return 0
    anchors = _anchor_tokens(item)
    base = (item.search_query or item.description or "").lower()
    qtokens = set(re.findall(r"[a-z0-9]{2,}", base))
    if not qtokens:
        return 0
    need = max(3, len(qtokens) // 3)
    added = 0
    for domain in domains[:max_sites]:
        products = _shopify_products(domain)
        if not products:
            continue
        scored = []
        for p in products:
            ptoks = set(re.findall(r"[a-z0-9]{2,}", p["title"].lower()))
            overlap = len(qtokens & ptoks)
            # require BOTH enough overlap AND at least one anchor token (brand /
            # part # / variant) so we don't grab an unrelated product off the feed
            if overlap >= need and ((not anchors) or (anchors & ptoks)):
                scored.append((overlap, p))
        scored.sort(key=lambda t: -t[0])
        for _, p in scored[:2]:
            url = canonical_url(p["url"])
            if not usable(url) or url in seen:
                continue
            seen.add(url)
            cands.append(PriceCandidate(title=p["title"][:120], url=url,
                                        source_site=domain, price=p.get("price")))
            added += 1
    if added:
        log.info("SKU %s — storefront-feed fallback: +%d candidate(s) from shop catalogs",
                 item.schein_sku, added)
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
        from .jobs import emit
        emit("firecrawl", action="search", stage=stage, query=query[:120],
             results=added, credits=used)
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
    # DEEPENED gap-sweep: one limit=20 search over 70+ domains returns only the 20
    # best-ranked URLs, so most suppliers (pstshop, dentaldash, …) never surface.
    # Split the gap into batches and search each with a higher limit so every
    # supplier batch gets its own ranked slots — far better coverage for a few
    # extra credits. Tunable: GAP_SWEEP_LIMIT (per-batch results), GAP_SWEEP_BATCH
    # (domains per batch). Each batch search ≈ ceil(limit/10)*2 credits.
    gap_limit = int(os.environ.get("GAP_SWEEP_LIMIT", "40"))
    gap_batch = int(os.environ.get("GAP_SWEEP_BATCH", "25"))
    n_batches = -(-len(gap) // gap_batch)
    log.info("SKU %s — supplier gap-sweep: %d/%d suppliers missing from organic; "
             "%d Firecrawl /search batch(es) (limit=%d, %d domains each) to fill gap",
             item.schein_sku, len(gap), len(SUPPLIER_SITES), n_batches, gap_limit, gap_batch)
    added = 0
    for i in range(0, len(gap), gap_batch):
        if _fc_state["exhausted"]:
            break
        added += _firecrawl_search(base, cands, seen, include_domains=gap[i:i + gap_batch],
                                   limit=gap_limit, stage="supplier-gap")
    return added


def market_sweep(item: OrderLineItem, max_candidates: int = 14) -> tuple[List[PriceCandidate], List[str]]:
    """Broad public sweep. Runs on EVERY item EVERY run (PRD 4.2)."""
    queries = _item_queries(item)
    from .jobs import emit
    emit("discovery_start", sku=item.schein_sku, queries=queries,
         message=f"Searching prices for SKU {item.schein_sku}")
    log.info("SKU %s — market sweep starting (%d quer%s: %s)",
             item.schein_sku, len(queries), "y" if len(queries) == 1 else "ies",
             " | ".join(q[:60] for q in queries))
    cands: List[PriceCandidate] = []
    flagged: List[str] = []
    seen: set[str] = set()

    # SEED URLS (Option 2): inject this SKU's known-good product URLs FIRST, so
    # they are always scraped regardless of discovery or cache state — a
    # guaranteed backstop for supplier pages no search engine indexes. Added
    # before the cache-hit return below so seeds survive a warm cache, and they
    # lead the candidate list so they claim scrape slots ahead of generic results.
    _seeds = SEED_URLS.get(item.schein_sku, [])
    for su in _seeds:
        cu = canonical_url(su)
        if cu not in seen and not is_excluded(cu) and not DOC_EXT_RE.search(cu):
            seen.add(cu)
            pc = PriceCandidate(title="", url=cu, source_site=_domain(cu), price=None)
            pc._seed = True   # known-good: exempt from the ≤2-per-domain shortlist cap
            cands.append(pc)
    if _seeds:
        log.info("SKU %s — seeded %d known-good URL(s) from seed_urls.txt",
                 item.schein_sku, len(_seeds))

    # --- DISCOVERY CACHE HIT: reuse known URLs, skip all discovery API calls ---
    # (Prices are still scraped fresh downstream, so report prices never go
    #  stale from this cache — only discovery calls are saved.)
    cached_urls = _disc_cache.get(item.schein_sku) if DISCOVERY_CACHE_ENABLED else None
    if cached_urls:
        _anchors = _anchor_tokens(item)
        _dropped = 0
        for url in cached_urls:
            cu = canonical_url(url)
            if usable(cu) and cu not in seen:
                if not _url_relevant(cu, item, _anchors):
                    _dropped += 1
                    continue
                seen.add(cu)
                cands.append(PriceCandidate(title="", url=cu, source_site=_domain(cu), price=None))
        if _dropped:
            log.info("SKU %s — discovery relevance filter dropped %d off-topic cached URL(s)",
                     item.schein_sku, _dropped)
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
                    with _shop_lock:
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

    # Firecrawl-INDEPENDENT fallback: the gap-sweep above is the ONLY mechanism
    # that discovers trusted suppliers missing from organic results, and it
    # returns nothing when Firecrawl is exhausted/absent — silently dropping the
    # manual-sheet niche suppliers (curio, thedentalmarket, dentaldash, …). When
    # Firecrawl can't run, fall back to the SerpAPI `site:domain` sweep over the
    # STILL-MISSING suppliers (SerpAPI's separate credit pool), so those suppliers
    # are still found. Skipped entirely when Firecrawl handled the gap (no double
    # spend) or SerpAPI is itself unavailable.
    if (_fc_state["exhausted"] or not FIRECRAWL_KEY) and SERPAPI_KEY and not _serp_state["exhausted"]:
        present = set()
        for c in cands:
            d = _domain(c.url)
            for s in SUPPLIER_SITES:
                if d == s or d.endswith("." + s) or s.endswith(d):
                    present.add(s)
        gap = [s for s in SUPPLIER_SITES if s not in present]
        if gap:
            log.info("SKU %s — Firecrawl unavailable; SerpAPI supplier fallback over "
                     "%d missing trusted supplier(s)", item.schein_sku, len(gap))
            _supplier_site_sweep(item, cands, seen, domains=gap)

    # Storefront-feed fallback (Option 1): for trusted suppliers STILL missing
    # after organic + Firecrawl gap-sweep + SerpAPI site: search, query each
    # store's OWN catalog feed (Shopify /products.json) directly — free, and the
    # only path that reaches tiny shops no search engine indexes (curio,
    # dentaldash, pstshop, …). Runs over whatever suppliers remain uncovered.
    if STOREFRONT_ENABLED and SUPPLIER_SITES:
        present = set()
        for c in cands:
            d = _domain(c.url)
            for s in SUPPLIER_SITES:
                if d == s or d.endswith("." + s) or s.endswith(d):
                    present.add(s)
        gap2 = [s for s in SUPPLIER_SITES if s not in present]
        if gap2:
            _storefront_feed_search(item, cands, seen, gap2)

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

    # Off-topic guard: drop candidates whose slug clearly describes a different
    # product (e.g. crazydental CROWN URLs surfaced for a Clinpro fluoride item)
    # BEFORE shortlisting/caching, so junk never eats a scrape slot or pollutes
    # the discovery cache for future runs.
    if DISCOVERY_RELEVANCE_FILTER and cands:
        _anchors = _anchor_tokens(item)
        kept = [c for c in cands if _url_relevant(c.url, item, _anchors)]
        if len(kept) < len(cands):
            log.info("SKU %s — discovery relevance filter dropped %d off-topic URL(s) "
                     "(%d → %d)", item.schein_sku, len(cands) - len(kept), len(cands), len(kept))
            cands = kept
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
    emit("discovery_complete", sku=item.schein_sku,
         shortlisted=len(shortlist), total_found=len(cands))
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


STRUCTURED_PRICE_ENABLED = os.environ.get("STRUCTURED_PRICE", "1") not in ("0", "false", "False", "")
STRUCTURED_PRICE_MAX = float(os.environ.get("STRUCTURED_PRICE_MAX", "100000"))

_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
# <meta property="product:price:amount" content="44"> (either attribute order)
_META_PRICE_RE_A = re.compile(
    r'<meta[^>]+(?:property|name|itemprop)=["\'](?:product:price:amount|og:price:amount|price)["\']'
    r'[^>]*\bcontent=["\']\s*\$?([0-9][0-9,]*\.?[0-9]*)["\']', re.I)
_META_PRICE_RE_B = re.compile(
    r'<meta[^>]+\bcontent=["\']\s*\$?([0-9][0-9,]*\.?[0-9]*)["\']'
    r'[^>]*(?:property|name|itemprop)=["\'](?:product:price:amount|og:price:amount|price)["\']', re.I)
# schema.org MICRODATA price: <span itemprop="price" data-rate="44.99"> $44.99 </span>
# (crazydentalprices and other stores expose price ONLY as microdata — no JSON-LD,
# no og:price — so a plain GET can read it for $0 instead of spending a scrape).
_MICRODATA_PRICE_RE = re.compile(
    r'itemprop=["\']price["\'][^>]*?'
    r'(?:data-rate|content)=["\']\s*\$?([0-9][0-9,]*\.[0-9]{1,2})', re.I)
_MICRODATA_PRICE_TEXT_RE = re.compile(
    r'itemprop=["\']price["\'][^>]*>\s*\$?([0-9][0-9,]*\.[0-9]{2})\s*<', re.I)


def _microdata_price(raw_html: str):
    """(price, single_offer) from schema.org itemprop=price microdata, or (None,
    False). single_offer is True only when the page declares ONE price — several
    distinct itemprop=price values mean a variant table (handled elsewhere)."""
    vals = []
    for rx in (_MICRODATA_PRICE_RE, _MICRODATA_PRICE_TEXT_RE):
        for m in rx.finditer(raw_html or ""):
            p = _to_price(m.group(1))
            if p:
                vals.append(p)
        if vals:
            break
    if not vals:
        return None, False
    return min(vals), len({round(v, 2) for v in vals}) == 1


def _to_price(v) -> Optional[float]:
    try:
        f = float(str(v).replace(",", "").replace("$", "").strip())
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _jsonld_price(raw_html: str):
    """Return (price, name, single_offer) from JSON-LD Product data.

    single_offer is True ONLY when the page declares effectively ONE price — a
    single plain Offer, or an AggregateOffer whose low == high. That is an
    unambiguous, trustworthy price for the one product on the page, so the caller
    may use it directly. A genuine AggregateOffer RANGE (low < high) or several
    distinct offer prices means a MULTI-VARIANT page whose JSON-LD low price is
    usually the CHEAPEST variant — NOT the ordered one — so single_offer is False
    and the caller must defer price extraction to the (variant-aware) LLM.
    Handles a dict, a list, or an @graph; offers as Offer / AggregateOffer / list.
    """
    priced = []   # (price, name, is_unambiguous)
    for m in _JSONLD_RE.finditer(raw_html or ""):
        try:
            data = json.loads(m.group(1).strip())
        except Exception:
            continue
        nodes = []
        for d in (data if isinstance(data, list) else [data]):
            if isinstance(d, dict):
                g = d.get("@graph")
                if isinstance(g, list):
                    nodes.extend(x for x in g if isinstance(x, dict))
                nodes.append(d)
        for node in nodes:
            t = node.get("@type")
            types = t if isinstance(t, list) else [t]
            if not any(str(x).lower() == "product" for x in types):
                continue
            name = node.get("name") if isinstance(node.get("name"), str) else None
            offers = node.get("offers")
            for off in (offers if isinstance(offers, list) else [offers] if offers else []):
                if not isinstance(off, dict):
                    continue
                cur = str(off.get("priceCurrency") or "").upper()
                if cur and cur != "USD":          # don't grab a EUR/GBP price
                    continue
                otype = str(off.get("@type") or "").lower()
                if "aggregateoffer" in otype:
                    lo = _to_price(off.get("lowPrice") or off.get("price"))
                    hi = _to_price(off.get("highPrice"))
                    if lo:
                        # a range (low < high) is genuinely multi-priced → ambiguous
                        priced.append((lo, name, hi is None or abs(hi - lo) < 0.01))
                else:
                    p = _to_price(off.get("price")
                                  or (off.get("priceSpecification") or {}).get("price"))
                    if p:
                        priced.append((p, name, True))
    if not priced:
        return None, None, False
    distinct = {round(pp, 2) for pp, _, _ in priced}
    lowest = min(priced, key=lambda o: o[0])
    # unambiguous only when every offer is a plain/equal price AND they all agree
    single = len(distinct) == 1 and all(unamb for _, _, unamb in priced)
    return lowest[0], lowest[1], single


_OOS_AVAIL_RE = re.compile(
    r'availability"\s*:\s*"[^"]*(?:OutOfStock|SoldOut|Discontinued|BackOrder|PreOrder)'
    r'|itemprop=["\']availability["\'][^>]*(?:OutOfStock|SoldOut)'
    r'|og:availability["\'][^>]*content=["\'](?:oos|outofstock|out\s*of\s*stock)', re.I)


def _struct_out_of_stock(raw_html: str) -> bool:
    """Deterministic stock signal from structured data: JSON-LD offers.availability,
    schema.org itemprop, or og:availability declaring the product unavailable. This
    is robust on nav-heavy pages (e.g. dds's BigCommerce page, whose 'Out of stock'
    buy-box text sits far past the markdown head that the text-scan guard reads)."""
    return bool(_OOS_AVAIL_RE.search(raw_html or ""))


def structured_price(metadata: dict, raw_html: str):
    """Price, product name, and single_offer flag from a page's structured data,
    independent of the nav-heavy markdown. JSON-LD first (Shopify/BigCommerce/Woo
    all emit it), then Firecrawl's metadata dict, then raw <meta> price tags.
    Returns (price, name, single_offer). single_offer=True means the price is the
    page's own unambiguous declared price and is safe to use directly."""
    if not STRUCTURED_PRICE_ENABLED:
        return None, None, False
    p, name, single = _jsonld_price(raw_html)
    meta = metadata if isinstance(metadata, dict) else {}
    if p is None:
        for k in ("product:price:amount", "og:price:amount", "priceAmount", "price"):
            if meta.get(k) is not None:
                p = _to_price(meta.get(k))
                if p:
                    single = True        # one declared og/product price tag = unambiguous
                    break
    if p is None and raw_html:
        mm = _META_PRICE_RE_A.search(raw_html) or _META_PRICE_RE_B.search(raw_html)
        if mm:
            p = _to_price(mm.group(1))
            if p:
                single = True
    if p is None and raw_html:                       # schema.org microdata (crazydental)
        mp, mp_single = _microdata_price(raw_html)
        if mp:
            p, single = mp, mp_single
    name = name or (meta.get("og:title") if isinstance(meta.get("og:title"), str) else None) \
        or (meta.get("title") if isinstance(meta.get("title"), str) else None)
    if p is None or p > STRUCTURED_PRICE_MAX:
        return None, name, False
    return p, name, single


# ---- FREE-FETCH tier: read a server-rendered structured price with a plain HTTP
# GET (no Firecrawl credits). Works for the many supplier sites (Shopify /
# BigCommerce / WooCommerce / most server-rendered stores) that embed a single
# JSON-LD or og:price in the raw HTML. Falls through to Firecrawl when the page is
# JS-only, bot-blocked, or has no trustworthy single-offer price. Aggregators
# (net32 / supplyclinic) are skipped — their seller tables need JS rendering.
FREE_FETCH_ENABLED = os.environ.get("FREE_FETCH", "1") not in ("0", "false", "False", "")
FREE_FETCH_TIMEOUT = (int(os.environ.get("FREE_FETCH_CONNECT", "8")),
                      int(os.environ.get("FREE_FETCH_READ", "15")))
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>")


def _html_to_text(html: str) -> str:
    """Crude HTML→text for variant validation: drop script/style, strip tags,
    unescape entities, collapse whitespace. Good enough for token matching."""
    import html as _htmlmod
    h = _SCRIPT_STYLE_RE.sub(" ", html or "")
    h = _TAG_RE.sub(" ", h)
    return re.sub(r"\s+", " ", _htmlmod.unescape(h)).strip()


# ---- VARIANT MATRIX: a single product page that sells many SKU variants, each
# with its OWN price (amtouch/BigCommerce Clinpro lists 7200FF/50 @ $144.14 and
# 7210FF/100 @ $288.27 on one page). The page's default/base structured price is
# whichever variant BigCommerce renders first — usually NOT the one ordered. The
# embedded per-variant JSON ("sku":"7210FF",..."prices":{"price":{"value":288.27}})
# IS server-rendered, so when the order's MPN matches a variant SKU we can take
# that variant's exact price deterministically (no JS, no LLM, 0 credits). The
# whole matrix is cached as a compact marker line so cache hits stay correct.
_VARIANT_BLOCK_RE = re.compile(
    r'"sku"\s*:\s*"([^"]+)"\s*,\s*"upc"\s*:[^,]*,\s*"prices"\s*:\s*\{\s*'
    r'"price"\s*:\s*\{\s*"currencyCode"\s*:\s*"[^"]*"\s*,\s*"value"\s*:\s*'
    r'([0-9]+(?:\.[0-9]+)?)')
_VMATRIX_MARKER = "[[VMATRIX "


def _json_array_after(html: str, key: str, start: int = 0):
    """Bracket-match and json.loads the array literal that follows "key": in html,
    starting at/after `start`. Returns (list_or_None, end_index). String-aware so
    brackets inside JSON strings don't throw off the depth count."""
    import json as _json
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[', html[start:])
    if not m:
        return None, len(html)
    i = start + m.end() - 1                    # index of '['
    depth, instr, esc, j = 0, False, False, i
    while j < len(html):
        ch = html[j]
        if esc:
            esc = False
        elif ch == '\\':
            esc = True
        elif ch == '"':
            instr = not instr
        elif not instr:
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    break
        j += 1
    try:
        return _json.loads(html[i:j + 1]), j + 1
    except Exception:
        return None, j + 1


def _variant_records(html: str) -> list:
    """Every priced variant embedded in the page as {sku, price, title}, across the
    common storefronts. Title (Shopify/Woo) lets the caller pick the ordered variant
    when no SKU/MPN matches (newarkdentalpemco 'Bottle 33oz' vs 'Packets 24Pkg'):
      • BigCommerce GraphQL — \\"sku\\":\\"7210FF\\"...\\"value\\":288.27 (escaped).
      • WooCommerce — data-product_variations array {sku, display_price, attributes}.
      • Shopify — "variants":[{"price":{"amount":N},"sku":"X","title":"Bottle 33oz"}]."""
    recs, seen = [], set()

    def add(sku, price, title=None):
        sku = (str(sku) if sku is not None else "").strip()
        try:
            p = float(price)
        except (TypeError, ValueError):
            return
        if p <= 0:
            return
        key = (sku, round(p, 2), (title or "").strip().lower())
        if key in seen:
            return
        seen.add(key)
        recs.append({"sku": sku, "price": p, "title": (title or "").strip()})

    # BigCommerce
    norm = (html or "").replace('\\u0022', '"').replace('\\"', '"')
    for m in _VARIANT_BLOCK_RE.finditer(norm):
        add(m.group(1), m.group(2))
    # WooCommerce
    mm = re.search(r'data-product_variations="([^"]+)"', html or "")
    if mm:
        import html as _htmlmod
        import json as _json
        try:
            for v in _json.loads(_htmlmod.unescape(mm.group(1))):
                attrs = v.get("attributes") if isinstance(v, dict) else None
                title = " ".join(str(x) for x in attrs.values()) if isinstance(attrs, dict) else None
                add(v.get("sku"), v.get("display_price"), title)
        except Exception:
            pass
    # Shopify — parse each "variants":[...] array by bracket matching (a fragile
    # regex bridge mis-pairs SKU↔price across adjacent variant objects).
    pos = 0
    for _ in range(40):                        # bound: a page may embed many arrays
        arr, pos = _json_array_after(html or "", "variants", pos)
        if arr is None and pos >= len(html or ""):
            break
        for v in (arr or []):
            if not isinstance(v, dict):
                continue
            price = v.get("price")
            amt = price.get("amount") if isinstance(price, dict) else price
            add(v.get("sku"), amt, v.get("title") or v.get("public_title"))
    return recs


def _extract_variant_matrix(html: str) -> dict:
    """{sku: price} view of _variant_records (kept for the SKU/MPN match path)."""
    out = {}
    for r in _variant_records(html):
        if r["sku"] and r["sku"] not in out:
            out[r["sku"]] = r["price"]
    return out


_SPRICE_MARKER = "[[SPRICE "


def _sprice_marker(price: float) -> str:
    return f"{_SPRICE_MARKER}{price:.2f}]]"


def _split_sprice(text: str):
    """Pull an embedded single-structured-price marker off the front of cached text.
    Returns (price_or_None, clean_text). Lets a JS-rendered page that only exposes
    its price in JSON-LD/microdata (alpinedentalshop $495.95) keep that price on a
    cache hit instead of falling back to a stale search-snippet price."""
    if not text or not text.startswith(_SPRICE_MARKER):
        return None, text
    try:
        j = text.index("]]")
        price = float(text[len(_SPRICE_MARKER):j])
        clean = text[j + 2:].lstrip("\n")
        return (price if price > 0 else None), clean
    except Exception:
        return None, text


def _vmatrix_marker(records) -> str:
    import json as _json
    if isinstance(records, dict):                  # legacy {sku:price}
        records = [{"sku": k, "price": v, "title": ""} for k, v in records.items()]
    compact = [[r.get("sku", ""), r.get("price"), r.get("title", "")] for r in records]
    return _VMATRIX_MARKER + _json.dumps(compact, separators=(",", ":")) + "]]"


def _split_vmatrix(text: str):
    """Pull the embedded variant marker (if any) off the front of cached text.
    Returns (records_list_or_None, clean_text). Accepts both the current
    [[sku,price,title],...] form and the legacy {sku:price} dict form. The marker
    is one whole line (page text is newline-free after _html_to_text), so split on
    the first newline — the records JSON itself ends in ']]', which would collide
    with a ']]' terminator."""
    if not text or not text.startswith(_VMATRIX_MARKER):
        return None, text
    import json as _json
    try:
        nl = text.find("\n")
        line = text if nl < 0 else text[:nl]
        clean = "" if nl < 0 else text[nl + 1:]
        payload = line[len(_VMATRIX_MARKER):]
        if payload.endswith("]]"):
            payload = payload[:-2]
        data = _json.loads(payload)
        recs = []
        if isinstance(data, dict):
            recs = [{"sku": k, "price": v, "title": ""} for k, v in data.items()]
        elif isinstance(data, list):
            for x in data:
                if isinstance(x, list) and len(x) >= 2:
                    recs.append({"sku": x[0], "price": x[1],
                                 "title": x[2] if len(x) > 2 else ""})
                elif isinstance(x, dict):
                    recs.append({"sku": x.get("sku", ""), "price": x.get("price"),
                                 "title": x.get("title", "")})
        return (recs or None), clean
    except Exception:
        return None, text


_SIZE_TOKEN_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(fl\s*oz|oz|ml|gallon|gal|gr|mg|cc)\b', re.I)


def _size_tokens(text: str) -> set:
    return {(m.group(1) + m.group(2)).replace(" ", "").lower()
            for m in _SIZE_TOKEN_RE.finditer(text or "")}


def _title_match_variant(c: PriceCandidate, records: list):
    """When SKU/MPN can't identify the variant, pick the one whose TITLE carries the
    ordered size (newarkdentalpemco 'Bottle 33oz' $105.39 vs 'Packets 24Pkg' $66.29
    for a '33oz/Bt' order). Conservative: needs ≥2 titled variants and the ordered
    size token to land on EXACTLY ONE of them — otherwise no match."""
    titled = [r for r in records if (r.get("title") or "").strip()]
    if len(titled) < 2:
        return None
    otok = _size_tokens((getattr(c, "_order_sizeform", "") or "") + " " +
                        (getattr(c, "_order_desc", "") or ""))
    if not otok:
        return None
    matches = [r for r in titled if otok & _size_tokens(r["title"])]
    return matches[0] if len(matches) == 1 else None


def _apply_variant_matrix(c: PriceCandidate, records, tag: str = "") -> bool:
    """Set the price of the variant that matches the ORDER, deterministically:
      1. page SKU == order Schein SKU / MPN (exact), or page SKU = vendor-prefix +
         MPN — same-product identity (sets mpn_confirmed).
      2. else the variant whose TITLE uniquely carries the ordered size.
    `records` is a list of {sku, price, title} (a legacy {sku:price} dict is also
    accepted). Returns True when a variant price was set."""
    if not records:
        return False
    if isinstance(records, dict):
        records = [{"sku": k, "price": v, "title": ""} for k, v in records.items()]
    mpn = (getattr(c, "_order_mpn", "") or "").strip()
    osku = (getattr(c, "_order_sku", "") or "").strip()   # Henry Schein SKU
    want = re.sub(r"[^a-z0-9]", "", mpn.lower())
    want_sku = re.sub(r"[^a-z0-9]", "", osku.lower())      # resellers often key by HS SKU
    has_alpha = any(ch.isalpha() for ch in want)
    exact = prefixed = None
    if len(want) >= 5 or len(want_sku) >= 5:
        # MATCH RULES (precise on purpose — a wrong variant locks a wrong price):
        # exact SKU/MPN, or page SKU = <vendor/mfr prefix> + order MPN (3M-7210FF
        # for 7210FF). NEVER match when the order MPN merely STARTS WITH the page
        # SKU (dropped suffix = usually the variant: 9736H-150-HP-TRQ ≠ ...HP).
        for r in records:
            norm = re.sub(r"[^a-z0-9]", "", str(r["sku"]).lower())
            if not norm:
                continue
            if (len(want_sku) >= 5 and norm == want_sku) or (len(want) >= 5 and norm == want):
                exact = r; break
            if (prefixed is None and has_alpha and len(want) >= 5 and len(norm) > len(want)
                    and norm.endswith(want)):
                b = norm[len(norm) - len(want) - 1]
                if not (b.isdigit() and want[0].isdigit()):
                    prefixed = r
    hit, kind = (exact or prefixed), "SKU/MPN"
    if not hit:
        hit, kind = _title_match_variant(c, records), "size/title"
    if not hit:
        return False
    try:
        price = float(hit["price"])
    except (TypeError, ValueError):
        return False
    if price <= 0:
        return False
    c.price = price
    c.price_structured = True
    c.price_locked = True            # deterministic — AI must not overwrite
    if kind == "SKU/MPN":
        c.mpn_confirmed = True       # page SKU == order identity → definitive product
    label = (hit.get("title") or hit.get("sku") or mpn or osku)
    c.notes = ((c.notes + " · ") if c.notes else "") + (
        f"variant-matrix price ${price:.2f} (ordered variant by {kind}: {label})")
    log.info("%sVARIANT MATRIX — $%.2f via %s (%s)", tag, price, kind, label)
    return True


def free_fetch_structured(c: PriceCandidate, tag: str = "") -> bool:
    """Try a free HTTP GET and set a SINGLE-offer structured price (0 Firecrawl
    credits). Returns True when a trustworthy price was set (caller skips
    Firecrawl), else False (caller falls back to Firecrawl). Never raises."""
    if not FREE_FETCH_ENABLED or _aggregator_domain(c.url):
        return False
    try:
        r = requests.get(c.url, headers={"User-Agent": _BROWSER_UA,
                                         "Accept": "text/html,application/xhtml+xml"},
                         timeout=FREE_FETCH_TIMEOUT, allow_redirects=True)
        if r.status_code != 200 or not r.text:
            return False
        html = r.text
    except Exception:
        return False
    try:
        sp, sname, sp_single = structured_price({}, html)
    except Exception:
        sp, sname, sp_single = None, None, False
    # VARIANT MATRIX: capture every SKU→price from the raw HTML before tags are
    # stripped. A multi-variant product page (BigCommerce matrix) is EXACTLY the
    # case where the single-offer gate below bails or picks the WRONG variant, so
    # try the deterministic MPN→SKU match FIRST. An MPN match is authoritative even
    # when no single structured price exists.
    records = _variant_records(html)
    matrix_hit = _apply_variant_matrix(c, records, tag)
    if not matrix_hit and not (sp_single and sp and sp > 0):
        return False                       # no variant match and no trustworthy single price → Firecrawl
    if not matrix_hit:
        c.price = sp
        c.price_structured = True
    if sname and not c.structured_name:
        c.structured_name = sname.strip()[:200]
    # Deterministic stock status from structured data — flag OOS even when the
    # buy-box 'Out of stock' text is buried past the markdown head (dds/BigCommerce).
    if _struct_out_of_stock(html):
        c.in_stock = False
        c.notes = ((c.notes + " · ") if c.notes else "") + "structured data: OUT OF STOCK"
    main = _strip_cross_sell(_html_to_text(html))
    # Prepend markers so cache hits re-apply the deterministic price for $0:
    #   VMATRIX — full {sku:price} variant matrix (matched by MPN/SKU at apply time)
    #   SPRICE  — the single unambiguous structured price (JSON-LD/og/microdata), so
    #             a JS-rendered page (alpinedentalshop $495.95) doesn't fall back to
    #             a stale search-snippet price ($199.99) on the next cache hit.
    markers = ""
    if records:
        markers += _vmatrix_marker(records) + "\n"
    if not matrix_hit and sp_single and sp and sp > 0:
        markers += _sprice_marker(sp) + "\n"
    cache_text = markers + main
    c.scraped_markdown = main[:int(os.environ.get("SCRAPE_MD_CAP", "10000"))]
    if not matrix_hit:
        c.notes = ((c.notes + " · ") if c.notes else "") + (
            "price $%.2f via free HTTP fetch (page structured data — 0 Firecrawl credits)" % sp)
    if SCRAPE_CACHE_ENABLED:
        try:
            from . import db
            db.save_scrape(c.url, cache_text[:int(os.environ.get("SCRAPE_CACHE_MD_CAP", "150000"))])
        except Exception:
            pass
    log.info("%sFREE FETCH (0 credits) — structured $%.2f from %s", tag, c.price or sp or 0.0, c.url[:60])
    return True


def firecrawl_verify(c: PriceCandidate, sku: str = "", allow_paid: bool = True) -> PriceCandidate:
    """Render the candidate page (JS executed) and extract authoritative data.
    The scraped page is the source of truth over the search-result title.

    allow_paid (Change #1): cache hits and the free HTTP structured fetch are
    always attempted (they cost $0). A PAID Firecrawl scrape only happens when
    allow_paid is True; otherwise the candidate is left un-scraped. The caller
    counts only paid scrapes against the per-item cap, so free scrapes are
    uncapped. c._paid_scrape is set True iff a real Firecrawl scrape was spent."""
    tag = f"SKU {sku} — " if sku else ""
    c._paid_scrape = False
    # SCRAPE CACHE: if this URL was scraped recently (this run or a previous one
    # that froze/crashed), reuse the stored markdown — costs ZERO Firecrawl
    # credits. This is what makes a re-run after a freeze cheap.
    if SCRAPE_CACHE_ENABLED:
        try:
            from . import db
            # Aggregator pages get the short (default 0) TTL; everything else 24h.
            ttl = AGG_SCRAPE_CACHE_HOURS if _aggregator_domain(c.url) else SCRAPE_CACHE_HOURS
            cached_md = db.get_cached_scrape(c.url, ttl) if ttl > 0 else None
        except Exception:
            cached_md = None
        if cached_md:
            # A cached variant-matrix page carries its {sku:price} marker on the
            # first line — split it off so the LLM never sees it, then re-apply the
            # MPN→variant price (keeps amtouch-style pages correct on cache hits).
            cached_matrix, cached_md = _split_vmatrix(cached_md)
            cached_sprice, cached_md = _split_sprice(cached_md)
            full_main = _strip_cross_sell(cached_md)          # full (uncapped) for table parsing
            # Login gate must run on CACHED pages too — otherwise a gated page
            # whose markdown holds a stray "$" (e.g. midwestdental floss "Sign in
            # for your pricing" + a substitute price) returns a phantom on every
            # cache hit. The hard gate is unconditional, so check it here as well.
            # EXCEPTION: aggregator marketplaces (net32 / supplyclinic) publish a
            # public seller table; their pages often also carry an "account pricing"
            # login prompt that would wrongly trip the hard gate and DROP a page the
            # deterministic table parser can price. Skip the login rejection for
            # those domains and let _apply_aggregator_lock be authoritative.
            if not _aggregator_domain(c.url) and _LOGIN_HARDGATE_RE.search(full_main):
                log.warning("%sscrape: login wall (price gated, cached) at %s", tag, c.url[:80])
                c.match_type = "rejected"
                c.rejected_reason = "price gated behind login (login-to-see-price)"
                return c
            c.scraped_markdown = full_main[:int(os.environ.get("SCRAPE_MD_CAP", "10000"))]
            if cached_matrix:
                _apply_variant_matrix(c, cached_matrix, tag)  # MPN→variant price (amtouch)
            if cached_sprice and not getattr(c, "mpn_confirmed", False):
                # JS-rendered page whose only price is structured data — re-apply it
                # so a stale search-snippet price can't win (alpinedentalshop $495.95).
                c.price = cached_sprice
                c.price_structured = True
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    f"price ${cached_sprice:.2f} via page structured data (cached)")
            _apply_aggregator_lock(c, full_main, tag)         # net32/supplyclinic → deterministic price
            log.info("%sscrape CACHE HIT (%d chars, 0 credits) — %s",
                     tag, len(c.scraped_markdown), c.url[:60])
            return c
    # FREE-FETCH TIER: before spending a Firecrawl scrape, try a plain HTTP GET and
    # read a single-offer structured price (0 credits). Most server-rendered
    # supplier stores expose it; only JS-only / bot-blocked / aggregator pages fall
    # through to Firecrawl below.
    if free_fetch_structured(c, tag):
        return c
    # CHANGE #1: only a PAID Firecrawl scrape is gated by the per-item cap. If the
    # caller has spent its paid budget, leave this page un-scraped (free fetch
    # already failed to find a structured price) rather than billing Firecrawl.
    if not allow_paid:
        c.notes = ((c.notes + " · ") if c.notes else "") + (
            "not scraped — per-item paid-scrape cap reached (no free structured price)")
        return c
    # Key check FIRST — a missing key means no request is ever made, so it must
    # not consume the scrape counter / per-run budget (which only the real scrapes
    # below should). Done before the lock block to keep the budget accounting honest.
    if not FIRECRAWL_KEY:
        log.warning("%sFirecrawl key missing; skipping scrape of %s", tag, c.url[:80])
        c.notes = "Firecrawl API key not configured"
        return c
    with _fc_lock:
        if _fc_state["exhausted"]:
            _fc_state["skipped"] += 1
            c.notes = "scrape skipped — Firecrawl credits exhausted (402) earlier in this run"
            return c
        if _fc_state["scrapes"] >= FIRECRAWL_MAX_SCRAPES:
            _mark_firecrawl_exhausted("budget")
            _fc_state["skipped"] += 1
            c.notes = f"scrape skipped — per-run budget of {FIRECRAWL_MAX_SCRAPES} scrapes reached"
            return c
        _fc_state["scrapes"] += 1
    c._paid_scrape = True            # a real (billed) Firecrawl scrape is being spent
    from .jobs import emit
    emit("firecrawl", action="scrape", sku=sku, site=c.source_site, url=c.url[:120],
         message=f"Firecrawl scraping {c.source_site}")
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
                    "formats": ["markdown", "rawHtml"],
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
        # Structured-data price: read JSON-LD / og-meta from the raw HTML so a
        # nav-bloated store page (BigCommerce/Shopify whose price sits below a
        # huge category menu, beyond the markdown cap) still yields its price.
        # Computed on FRESH scrapes only (the scrape cache stores markdown, not
        # HTML); harmless when absent. Skipped for aggregator pages, which have
        # their own deterministic table lock.
        if STRUCTURED_PRICE_ENABLED and not _aggregator_domain(c.url):
            try:
                sp, sname, sp_single = structured_price(
                    payload.get("metadata") or {},
                    payload.get("rawHtml") or payload.get("html") or "")
                if sp is not None:
                    c.structured_price = sp
                    if sname and not c.structured_name:
                        c.structured_name = sname.strip()[:200]
                    # A SINGLE-offer structured price is the page's own unambiguous
                    # declared price — accurate and 0 Groq tokens. Use it as the
                    # candidate price (overriding the unreliable search snippet) and
                    # mark it price_structured so the LLM VALIDATES the variant but
                    # never overwrites the number. Multi-variant / aggregate-range
                    # prices are NOT trusted here (usually the cheapest variant) and
                    # are left to the variant-aware LLM extraction.
                    if sp_single:
                        c.price = sp
                        c.price_structured = True
                        c.notes = ((c.notes + " · ") if c.notes else "") + (
                            "price $%.2f from page structured data (single offer)" % sp)
            except Exception:
                pass
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        log.error("%sscrape FAILED (%s) for %s", tag, status, c.url[:80])
        if status == 402:
            _mark_firecrawl_exhausted("402", url=c.url, tag=tag)
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

    # Login-wall heuristic (no AI call) so gated pages are dropped before the
    # Groq batch. KEY REFINEMENT: many dental B2B sites show a public LIST price
    # AND a "log in for your (contract) price" upsell — those are NOT gated, the
    # public price is right there (e.g. scottsdental shows $133.95 / $129.95
    # without login). So a login phrase only means "gated" when the page's MAIN
    # content has NO real price. We strip cross-sell first so a related item's
    # price (or its own "login to see price") can't sway the decision.
    if not markdown.strip():
        c.notes = "scrape returned empty content"
        return c
    main = _strip_cross_sell(markdown)
    main_prices = PRICE_RE.findall(main)        # real "$<number>" tokens in main content
    low_main = main.lower()
    login_phrase = bool(_LOGIN_PRICE_RE.search(main))   # "log in to see price" etc.
    hard_gate = bool(_LOGIN_HARDGATE_RE.search(main))   # "login to SEE price" (slot hidden)
    login_words = ("log in" in low_main or "login" in low_main or "sign in" in low_main
                   or "member price" in low_main or "call for price" in low_main
                   or "request a quote" in low_main)
    # Aggregator marketplaces (net32 / supplyclinic) publish a public seller table
    # and are priced deterministically by _apply_aggregator_lock below; their
    # "log in for account pricing" prompts must NOT trip the login-wall rejection
    # (doing so dropped genuinely-priceable net32 pages). Skip both rejections for
    # those domains — the seller-table parser is the source of truth there.
    is_aggregator = bool(_aggregator_domain(c.url))
    if hard_gate and not is_aggregator:
        # The price slot itself is gated ("login to see price"). Whatever "$" we
        # scraped is a promo / related-item / metadata artifact, never the buy
        # price (e.g. dhpsupply Clinpro shows $24.95 nowhere a customer can buy).
        log.warning("%sscrape: login wall (price gated — 'login to see price') at %s",
                    tag, c.url[:80])
        c.match_type = "rejected"
        c.rejected_reason = "price gated behind login (login-to-see-price)"
        return c
    if (login_phrase or login_words) and not main_prices and not is_aggregator:
        # gated AND no public price anywhere in the main content → reject
        log.warning("%sscrape: login wall (no public price shown) at %s", tag, c.url[:80])
        c.match_type = "rejected"
        c.rejected_reason = "login required for pricing (public pricing only)"
        return c
    if login_phrase and main_prices:
        # public price present alongside a "log in for your price" upsell — keep
        # it; the visible list price is usable. Just note the account-pricing hint.
        c.notes = ((c.notes + " · ") if c.notes else "") + \
            "public list price shown; site also offers account/contract pricing on login"
        log.info("%slogin-for-price upsell present but %d public price(s) shown — "
                 "keeping (NOT gated): %s", tag, len(main_prices), c.url[:70])

    # store the markdown for batched Groq extraction (done per-item in matcher).
    # Truncate to keep the Groq batch prompt within token limits. Raised from
    # 2000→6000: a real product page's price/pack/variant block can sit a few
    # thousand chars in (after title, breadcrumb, gallery, variant selector), so
    # a 2000-char slice often cut the price off — Groq then returned null and a
    # stale snippet price survived. Huge pages (380k-char category listings) are
    # still rejected outright by OVERSIZE below — that guard, not this cap, is
    # what keeps non-product pages out of the batch.
    # 10000 (was 6000): multi-product pages (safco's Fuji II LC lists the applier
    # + every shade on ONE url) push the ORDERED variant's row past 6000 — e.g. the
    # A3/48-box row + its $364.99 sit at ~6500-7700 while the $140.49 applier sits
    # at ~1955, so a 6000 cap fed the model ONLY the cheap applier. The LLM prompt
    # is unaffected (ai._slice still windows to EXTRACT_PER_PAGE_CHARS); this cap
    # only governs how much markdown the variant-anchor can reach.
    MD_CAP = int(os.environ.get("SCRAPE_MD_CAP", "10000"))
    OVERSIZE = int(os.environ.get("SCRAPE_OVERSIZE_REJECT", "150000"))
    if len(markdown) > OVERSIZE:
        log.info("%soversized page (%d chars) — almost certainly a category/search "
                 "listing, not a product page; skipping extraction", tag, len(markdown))
        c.notes = "skipped — page too large to be a product listing (category/search page)"
        return c
    c.scraped_markdown = main[:MD_CAP]          # reuse the already-stripped main content
    # Deterministic aggregator price (net32 / supplyclinic): parse the FULL main
    # content (before the MD_CAP truncation) so every seller row is seen, even
    # when the stored markdown is capped at 6000 chars.
    _apply_aggregator_lock(c, main, tag)
    # Persist immediately so a later freeze/crash never loses this fetched page
    # — a re-run reads it from cache for 0 credits. STORE THE FULL stripped main
    # (not the MD_CAP-truncated slice): the aggregator table parser on a CACHE
    # HIT must see every seller row, and net32 tables sit well past 6000 chars.
    if SCRAPE_CACHE_ENABLED:
        try:
            from . import db
            cache_cap = int(os.environ.get("SCRAPE_CACHE_MD_CAP", "150000"))
            db.save_scrape(c.url, main[:cache_cap])
        except Exception:
            pass
    log.info("%sscrape SUCCESS (markdown, stored %d of %d chars) — %s",
             tag, len(c.scraped_markdown), len(markdown), c.url[:60])
    return c