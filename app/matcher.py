"""Exact-match layer (Stage 1) and equivalency layer (Stage 2).

Matching = cheap deterministic pre-checks first (price sanity, volume
normalization, pack tolerance), then a single batched Groq (Llama 3.3 70B) validation per
item applying the four trust criteria. Only candidates with all four criteria
confirmed enter the primary price-match report (PRD 4.5).
"""
from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from . import ai, search
from .models import (EquivalencyEntry, EquivalencyFinding, ItemResult,
                     OrderLineItem, PriceCandidate)

log = logging.getLogger(__name__)

# ------------------------------------------------------------- heuristics ---

VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|cc|oz|fl\s?oz|l|liter|gallon|gal)\b", re.I)
_ML = {"ml": 1.0, "cc": 1.0, "oz": 29.5735, "fl oz": 29.5735, "floz": 29.5735,
       "l": 1000.0, "liter": 1000.0, "gallon": 3785.41, "gal": 3785.41}


def normalize_volume_ml(text: str) -> Optional[float]:
    m = VOLUME_RE.search(text or "")
    if not m:
        return None
    unit = re.sub(r"\s+", " ", m.group(2).lower())
    return float(m.group(1)) * _ML.get(unit, 1.0)


def _coerce_price(c: PriceCandidate) -> None:
    """Guarantee c.price is float or None. Snippet prices and some extraction
    paths can leave a string here; comparing str to number would 500."""
    if c.price is None or isinstance(c.price, (int, float)):
        return
    import re as _re
    s = str(c.price).replace(",", "").replace("$", "").strip()
    m = _re.search(r"\d+(?:\.\d+)?", s)
    c.price = float(m.group(0)) if m else None


def price_sane(item: OrderLineItem, c: PriceCandidate) -> bool:
    """Reject obviously-wrong extractions (dimension-like numbers, unit-price
    of a single piece vs a 100-pack, etc.)."""
    _coerce_price(c)
    if c.price is None or c.price <= 0:
        return False
    # a legitimate competitor price for the same pack rarely sits below 5% or
    # above 300% of the Schein unit price
    return 0.05 * item.unit_price <= c.price <= 3.0 * item.unit_price


def pack_compatible(item: OrderLineItem, c: PriceCandidate) -> Optional[bool]:
    """True/False when both pack quantities are known; None when unknown."""
    if item.pack_qty is None or c.pack_qty is None:
        return None
    return item.pack_qty == c.pack_qty       # exact only — no tolerance creep


# Deterministic pack-size reader for the weak free-tier extractor. The LLM often
# leaves c.pack_qty = None, so a wrong-pack page (a single ParaPost post at $42.31
# when 5/Pk was ordered, market floor ~$62) slips through as EXACT and wins on
# price. This reads an EXPLICIT pack count from the page's OWN text — only counts
# anchored to a pack keyword, never a bare number, so "Size 5" / "PF171-5" /
# "50mL" can never be misread as a pack quantity.
_PACK_COUNT_RES = (
    re.compile(r"\bcontains?:?\s+(\d{1,4})\b", re.I),                       # "Contains: 5 per Package"
    re.compile(r"(\d{1,4})\s*(?:per|/)\s*(?:pack|pkg|package|box|bx|case|ca|carton|ctn|refill)\b", re.I),
    re.compile(r"\b(?:pack|package|pkg|box|case|carton|bag)\s+of\s+(\d{1,4})\b", re.I),
    re.compile(r"\b(\d{1,4})\s*[-/]?\s*(?:pk|pack|pkg|count|ct|pcs|pieces|posts)\b", re.I),
)
_SINGLE_RE = re.compile(
    r"\b(?:sold\s+(?:individually|singly|separately)|single\s+(?:post|unit|item|piece|pack)|"
    r"1\s*(?:per|/)\s*(?:pack|pkg|package|box)|each\s+post|individual\s+post)\b", re.I)


def pack_from_page(c: PriceCandidate) -> Optional[int]:
    """Best-effort EXPLICIT pack count from the candidate's own page text, else
    None. Trusts the extractor's own pack_condition when it's a bare integer,
    then the product name/title, then only the buy-box slice of the markdown."""
    pc = (getattr(c, "pack_condition", None) or "").strip()
    if pc.isdigit() and 1 <= int(pc) <= 5000:      # extractor's own quantity field
        return int(pc)
    head = " ".join(p for p in (getattr(c, "scraped_product_name", None),
                                getattr(c, "title", None)) if p)
    body = (getattr(c, "scraped_markdown", None) or "")[:1200]   # buy-box region only
    for text in (head, body):
        if not text:
            continue
        if _SINGLE_RE.search(text):
            return 1
        for rx in _PACK_COUNT_RES:
            m = rx.search(text)
            if m and 1 <= int(m.group(1)) <= 5000:
                return int(m.group(1))
    return None


def volume_compatible(item: OrderLineItem, c: PriceCandidate) -> Optional[bool]:
    a = normalize_volume_ml(item.description)
    b = normalize_volume_ml((c.scraped_product_name or "") + " " + (c.title or ""))
    if a is None or b is None:
        return None
    return abs(a - b) / a <= 0.02


# Deterministic variant backstop for the weak free-tier extractor: catch
# flavor / shade / size-family mismatches the LLM let through (e.g. ordered
# Flavorless Clinpro but the page/URL is the Watermelon variant, or ordered
# ParaPost size #5 but the page is #4). Conservative by design: it only enforces
# a family the ORDERED item actually specifies, only fires when the candidate
# states a CONFLICTING value, and only DEMOTES (exact→approximate) — it never
# rejects, so a missing-data page is never punished.
_FLAVORS = (
    "flavorless", "unflavored", "no-flavor", "no flavor", "plain", "natural",
    "spearmint", "peppermint", "wintergreen", "bubblegum", "bubble gum",
    "watermelon", "strawberry", "cinnamon", "vanilla", "caramel", "tutti-frutti",
    "cherry", "grape", "orange", "melon", "berry", "lemon", "cola", "mint",
)
_NEUTRAL_FLAVORS = {"flavorless", "unflavored", "no-flavor", "no flavor", "plain", "natural"}
_SHADE_RE = re.compile(r"\b([ABCD][1-4](?:\.5)?)\b")
_SIZE_RE = re.compile(r"(?:#|\bsize\s+|\bsz\s+|pf171-)(\d{1,2})\b", re.I)
# A consumable/material order must not match the APPLIER / dispensing gun that
# shares the product-family name. The Fuji II LC A3 order (a 48/bx shade refill,
# ~$365) was matched to safco's "Fuji capsule applier" row ($140.49) — a wholly
# different SKU. These tokens name a tool, not a material; fire only when the
# candidate has one and the order does not.
_WRONG_FORM = ("applier", "dispensing gun", "capsule gun", "applicator gun")
# Henry Schein house brands — sold by no other store, so a different brand on the
# page is a generic substitute, not the same product.
_HOUSE_BRANDS = ("criterion", "acclean", "maxima")


# Out-of-stock / unavailable phrasing on a product page's buy box. A listing that
# can't be bought is not a real price and must never headline as the recommended
# best (e.g. orthazone's "OUT OF STOCK / NO LONGER AVAILABLE" Glide at $43.75
# undercutting the in-stock crazydental $44.99).
_OOS_TEXT_RE = re.compile(
    r"out\s*of\s*stock|out-of-stock|sold\s*out|no\s+longer\s+available|"
    r"currently\s+unavailable|temporarily\s+unavailable|discontinued|"
    r"notify\s+me\s+when|back\s*in\s*stock", re.I)


def _is_out_of_stock(c: PriceCandidate) -> bool:
    """True when the candidate's page is out of stock / no longer available.
    Trusts the LLM's product-specific in_stock=False first; for aggregator pages
    the seller-table lock already excluded unavailable sellers (in_stock=True), so
    we don't text-scan those. For normal pages, a strong unavailability phrase in
    the buy-box head is a backstop."""
    if getattr(c, "price_locked", False):
        return getattr(c, "in_stock", None) is False
    if getattr(c, "in_stock", None) is False:
        return True
    # Scan a wide window (not just the first 1200 chars): on nav-heavy pages the
    # buy-box 'Out of stock' text sits well past a big category sidebar (dds). The
    # markdown is already cross-sell-stripped, so an unambiguous OOS phrase here is
    # the product's own status, not a related item's.
    head = (getattr(c, "scraped_markdown", None) or "")[:6000]
    return bool(_OOS_TEXT_RE.search(head)) or bool(_OOS_TEXT_RE.search(getattr(c, "pack_condition", "") or ""))


def _flavor_in(text: str) -> Optional[str]:
    t = (text or "").lower()
    for f in _FLAVORS:           # specific flavors first; generic 'mint' last
        if f in t:
            return f
    return None


# Distinctive variant COLORS. Deliberately excludes white/clear/natural/ivory and
# metals (gold/silver/bronze) — those collide with product names (e.g. "Clinpro
# Clear", gold-coated burs) and would false-positive. Word-boundary matched.
_COLORS = ("teal", "turquoise", "aqua", "lavender", "violet", "magenta",
           "purple", "pink", "orange", "yellow", "green", "blue", "red",
           "black", "brown", "gray", "grey")
_COLOR_RE = re.compile(r"\b(" + "|".join(_COLORS) + r")\b", re.I)


def _colors_in(text: str) -> set:
    """Set of distinct palette colors named as standalone tokens (grey→gray)."""
    out = set()
    for m in _COLOR_RE.finditer(text or ""):
        col = m.group(1).lower()
        out.add("gray" if col == "grey" else col)
    return out


def variant_mismatch(item: OrderLineItem, c: PriceCandidate) -> Optional[str]:
    """Return a short reason if the candidate's variant conflicts with the order,
    else None. See module note above for the (deliberately conservative) rules."""
    ordered_txt = f"{getattr(item, 'variant', '') or ''} {item.description or ''}"
    cand_name = " ".join(p for p in (getattr(c, "scraped_variant", None),
                                     getattr(c, "scraped_product_name", None),
                                     getattr(c, "title", None)) if p)
    cand_all = cand_name + " " + (getattr(c, "url", None) or "")
    # ---- flavor: flag ONLY when exactly one side is a neutral (flavorless/
    # unflavored/no-flavor) and the other is a specific flavor. Neutral-vs-neutral
    # ("No Flavor" vs "Flavorless") are synonyms — NOT a mismatch. Specific-vs-
    # specific is left to the LLM (avoids spearmint-vs-mint false positives).
    of, cf = _flavor_in(ordered_txt), _flavor_in(cand_all)   # cand_all incl. URL: 'Unflavored' in the slug
    if of and cf and ((of in _NEUTRAL_FLAVORS) != (cf in _NEUTRAL_FLAVORS)):
        return f"flavor mismatch (ordered '{of}', page '{cf}')"
    # ---- shade (VITA A1..D4) — only when the order specifies one
    osd, csd = _SHADE_RE.search(ordered_txt), _SHADE_RE.search(cand_all)
    if osd and csd and osd.group(1).upper() != csd.group(1).upper():
        return f"shade mismatch (ordered {osd.group(1).upper()}, page {csd.group(1).upper()})"
    # ---- size / post number (#5, size 4, PF171-4) — only when order specifies one
    osz, csz = _SIZE_RE.search(ordered_txt), _SIZE_RE.search(cand_all)
    if osz and csz and osz.group(1) != csz.group(1):
        return f"size mismatch (ordered #{osz.group(1)}, page #{csz.group(1)})"
    # ---- color variant — fire ONLY when BOTH the order and the page name a
    # palette color AND the two color sets are DISJOINT (share none). Using
    # disjointness (not first-color) keeps multi-color items safe: ordered
    # "Blue/Orange" vs a page that also says blue+orange shares colors → no fire,
    # while ordered "Teal" vs a "Green" page is disjoint → demote. Page side uses
    # cand_all (name + URL slug) — a wrong color in the slug (".../pink-mixing-tips")
    # is a reliable variant signal that the extracted name often omits; word
    # boundaries keep embedded domain colors (e.g. "blueskydental") from matching.
    ocol, ccol = _colors_in(ordered_txt), _colors_in(cand_all)
    if ocol and ccol and ocol.isdisjoint(ccol):
        return (f"color mismatch (ordered {'/'.join(sorted(ocol))}, "
                f"page {'/'.join(sorted(ccol))})")
    # ---- product FORM: order is a material/refill, page is a dispensing tool
    cf_form = next((w for w in _WRONG_FORM if w in cand_name.lower()), None)
    if cf_form and not any(w in ordered_txt.lower() for w in _WRONG_FORM):
        return f"form mismatch (page is a '{cf_form}', order is the material)"
    # ---- HOUSE BRAND: a Henry Schein house brand (Criterion/Acclean/Maxima) is
    # sold by no one else, so a page that does NOT name that brand is a different
    # brand's generic substitute (net32 'Aurelia Sonic' gloves for a 'Criterion
    # N300' order), NOT the same product — keep it out of price_match.
    ohb = next((h for h in _HOUSE_BRANDS if h in ordered_txt.lower()), None)
    if ohb and ohb not in cand_all.lower():
        return f"house-brand mismatch (ordered {ohb.title()}, page is a different brand)"
    return None



def _brand_ok(item: OrderLineItem, c: PriceCandidate) -> bool:
    """Brand criterion passes when confirmed, or when the reference item has
    no identifiable brand (generic/house items like 'Barrier Film Blue').

    Also passes on a brand-LABEL mismatch when the product NAME, size/form AND
    pack are ALL confirmed: identical product-name + size + pack is the same
    product, so a differing brand label is almost always a brand-parse artifact
    (e.g. 'Dynamic Mixing Tips' mis-parsed as Kerr vs the genuine Coltene 6162
    listing) or a reseller relabel — not a different product. name_match already
    establishes product identity, so brand shouldn't veto EXACT here. The
    candidate is still tagged is_generic_equivalent for transparency. Disable
    via BRAND_LENIENT_ON_FULL_MATCH=0."""
    crit = c.criteria or {}
    if bool(crit.get("brand_match")) or not item.brand:
        return True
    if (os.environ.get("BRAND_LENIENT_ON_FULL_MATCH", "1") not in ("0", "false", "False")
            and crit.get("name_match") and crit.get("size_form_match")
            and crit.get("pack_match")):
        return True
    return False


def _likely_exact(item: OrderLineItem, c: PriceCandidate) -> bool:
    """Pre-validation heuristic for the early-stop counter (Groq criteria are
    not assigned until after the scraping loop): page verified, pack
    compatible, and strong token overlap between the item and the page."""
    if c.scraped_product_name is None:
        return False
    if pack_compatible(item, c) is False or volume_compatible(item, c) is False:
        return False
    ref = f"{item.brand or ''} {item.product_name or item.description}".lower()
    page = f"{c.scraped_product_name} {c.title}".lower()
    tokens = [w for w in re.findall(r"[a-z0-9.]+", ref) if len(w) > 2]
    if not tokens:
        return False
    hits = sum(1 for w in tokens if w in page)
    return hits / len(tokens) >= 0.6


def _size_form_ok(item: OrderLineItem, c: PriceCandidate) -> bool:
    """Size/form criterion passes when the LLM confirmed it, OR — for a
    deterministically LOCKED aggregator price (the product parsed off the page's
    OWN seller table) — when name + pack are confirmed and there is no hard volume
    conflict. The free-tier model often can't confirm a form detail that the
    aggregator listing simply doesn't repeat (e.g. 'w/LuerLock' on a 'Dynamic
    Mixing Tips Yellow 40/Pack' row), which wrongly blocked a genuine exact match.
    Disable via SIZE_LENIENT_ON_LOCKED=0."""
    crit = c.criteria or {}
    if crit.get("size_form_match"):
        return True
    if (os.environ.get("SIZE_LENIENT_ON_LOCKED", "1") not in ("0", "false", "False")
            and getattr(c, "price_locked", False)
            and crit.get("name_match") and crit.get("pack_match")
            and volume_compatible(item, c) is not False):
        return True
    return False


def _effective_exact(item: OrderLineItem, c: PriceCandidate) -> bool:
    # A confirmed MPN match is the identical product — exact, unless a hard variant
    # conflict or pack incompatibility was detected.
    if (getattr(c, "mpn_confirmed", False) and not getattr(c, "variant_conflict", False)
            and pack_compatible(item, c) is not False):
        return True
    crit = c.criteria or {}
    return (crit.get("name_match") and _size_form_ok(item, c)
            and crit.get("pack_match") and _brand_ok(item, c))

# --------------------------------------------------------------- pipeline ---

def process_item(item: OrderLineItem, max_verify: int = 8) -> ItemResult:
    sku = item.schein_sku
    result = ItemResult(item=item)
    cands, flagged = search.market_sweep(item)
    result.flagged_sites = flagged

    # ISSUE 1: order the scrape queue "likely-lowest-price first" so the per-item
    # scrape cap spends its slots on the real contenders (not buried past the cap).
    # An earlier version scraped trusted-suppliers first, which crowded the genuine
    # cheapest listings out of the cap — the cheapest source the buyer wants then
    # never got priced. Priority tiers (cap COUNT unchanged, so cost is unchanged):
    #   0  aggregator pages (net32/supplyclinic) — carry the marketplace lows and
    #      are deterministically priced (cheap to process); cheapest snippet first
    #   1  any candidate with a discovery price — cheapest first (the likely winner)
    #   2  trusted suppliers without a snippet price — price is revealed on-scrape
    #   3  everything else
    def _scrape_priority(c):
        if search._aggregator_domain(c.url):
            return (0, c.price if (c.price and c.price > 0) else 0.0)
        if c.price is not None and c.price > 0:
            return (1, c.price)
        if search.is_trusted_supplier(c.url):
            return (2, 0.0)
        return (3, 0.0)
    cands.sort(key=_scrape_priority)

    # verify the most promising candidates on-page (JS rendered). Batched Groq
    # extraction means prices aren't known mid-loop, so instead of a dynamic
    # early-stop we cap scrapes per item (cheapest-first shortlist preserves the
    # lowest-priced candidates). SCRAPE_CAP_PER_ITEM bounds Firecrawl cost.
    import os as _os
    # CHANGE 2: 12 (was 8). On items with many listings the genuinely cheapest
    # sellers sit at positions 9-12 (e.g. Tray Cover's clinicalsupplycompany /
    # ismiledental, Criterion's cheap glove sources) and were left "not scraped —
    # cap reached", so the buyer's lowest price never reached the report. Raising
    # to 12 reaches them. Cost is bounded: scrape priority puts aggregators +
    # cheapest first, already-scraped pages are cache-served (0 credits), fresh
    # scrapes are still capped by FIRECRAWL_MAX_SCRAPES_PER_RUN, and Groq stays
    # bounded by GROQ_EXTRACT_CAP (extra pages get the free regex fallback).
    # Tune via SCRAPE_CAP_PER_ITEM (lower it to spend fewer Firecrawl credits).
    # CHANGE #1: decouple FREE scraping from PAID Firecrawl. SCRAPE_CAP_PER_ITEM now
    # bounds only PAID Firecrawl scrapes; free scrapes (cache hit + free HTTP
    # structured fetch) cost $0 and are NOT capped — so every free-fetchable
    # supplier page gets priced, while Firecrawl is reserved for the cheapest
    # JS/aggregator pages that need it. FREE_FETCH_MAX bounds total candidates
    # attempted per item (the free-fetch HTTP GETs add latency), so a 40-URL item
    # doesn't free-fetch all 40.
    scrape_cap = int(_os.environ.get("SCRAPE_CAP_PER_ITEM", "12"))   # PAID Firecrawl cap
    free_max = int(_os.environ.get("FREE_FETCH_MAX", "24"))          # total candidates attempted
    verified: List[PriceCandidate] = []
    exacts_found = 0
    verified_ok = 0
    paid_n = 0
    for idx, c in enumerate(cands):
        if idx >= free_max:
            c.notes = c.notes or "not scraped — per-item coverage cap reached"
            result.candidates.append(c)
            continue
        c._ordered_variant = item.variant or ""
        c._order_mpn = item.mpn or ""        # variant-matrix pages price by MPN→SKU
        c._order_sku = item.schein_sku or "" # …or by Henry Schein SKU (afteractionmedical)
        c._order_sizeform = item.size_form or ""   # …or by size in the variant title
        c._order_desc = item.description or ""      # (33oz/Bt → 'Bottle 33oz' variant)
        c = search.firecrawl_verify(c, sku=sku, allow_paid=(paid_n < scrape_cap))
        if getattr(c, "_paid_scrape", False):     # a real Firecrawl scrape was spent
            paid_n += 1
        verified.append(c)

    # ONE Groq call per item that BOTH extracts pricing AND judges match criteria
    # (was two calls: extract_pages_batch + validate_candidates). Halves Groq
    # request volume for free-tier rate-limit relief; same inputs → same outputs.
    ai.extract_and_validate_batch(item, verified)

    # NO-AI SAFETY NET: if Groq was rate-limited/exhausted and left some scraped
    # pages unprocessed, pull their price straight from the markdown (regex,
    # flagged UNVERIFIED) so the report shows a price instead of nothing.
    ai.manual_extract_fallback(item, verified)

    # deterministic demotions run AFTER the combined call (price/pack populated).
    # These OVERRIDE the AI verdict: a price-insane or volume-mismatched candidate
    # is rejected even if the model called it exact.
    for c in verified:
        if c.match_type == "rejected":
            continue
        # DETERMINISTIC PACK GUARD: the free-tier LLM frequently leaves pack_qty
        # blank, so fill it from the page's own stated quantity before the
        # mismatch check below. Only when the order specifies a pack and the LLM
        # gave us nothing — we never override a pack the model did extract.
        if c.pack_qty is None and item.pack_qty:
            pg = pack_from_page(c)
            if pg is not None:
                c.pack_qty = pg
                if pg != item.pack_qty:
                    log.info("SKU %s — %s pack read from page: %s/pack (ordered %s)",
                             sku, c.source_site, pg, item.pack_qty)
        if not price_sane(item, c):
            c.match_type = "rejected"
            c.rejected_reason = (c.rejected_reason or
                                 f"price {c.price} failed sanity vs Schein unit {item.unit_price}")
        elif pack_compatible(item, c) is False:
            # Pack mismatch can NEVER be an exact match — demote it so it isn't
            # mislabeled EXACT in the options/evidence sheet (it was previously
            # only excluded from best_exact but kept its 'exact' label).
            if c.match_type == "exact":
                c.match_type = "approximate"
                log.info("SKU %s — demoted %s to approximate (pack %s vs ordered %s)",
                         sku, c.source_site, c.pack_qty, item.pack_qty)
            # NOTE: a GROSS pack mismatch (≥3×) is hard-excluded in the standalone
            # pack-backstop pass below (which also covers LLM-rejected candidates).
            # Here we only demote + annotate so a MILD difference (40 vs 50 tips)
            # still shows as a like-for-like approximate option.
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                f"PACK MISMATCH — page is {c.pack_qty}/pack but ordered "
                f"{item.pack_qty}/pack; treated as approximate, VERIFY")
            if c.pack_condition is None:
                c.pack_condition = f"{c.pack_qty}-pack price (ordered pack: {item.pack_qty})"
        elif volume_compatible(item, c) is False:
            c.match_type = "rejected"
            c.rejected_reason = "volume/size mismatch after normalization"
        if c.scraped_product_name is not None:
            verified_ok += 1
            if _likely_exact(item, c):
                exacts_found += 1

    # VARIANT BACKSTOP: deterministic flavor/shade/size/color/house-brand check, in
    # its OWN pass over EVERY candidate. It cannot live in the demotion loop above —
    # that loop `continue`s on already-rejected candidates, so a wrong product the
    # LLM rejected for some OTHER reason (net32 'Aurelia Sonic' rejected as
    # off-brand, but still the cheapest) would never get variant_conflict set and
    # would leak into price_match as a "POSSIBLE" option. Setting variant_conflict
    # here routes it to the alternate sheet. We never relax a verdict — only flag.
    for c in verified:
        if not getattr(c, "variant_conflict", False):
            vm = variant_mismatch(item, c)
            if vm:
                c.variant_unverified = True
                c.variant_conflict = True   # wrong color/shade/size/flavor/brand → keep out of price_match
                if c.match_type == "exact":
                    c.match_type = "approximate"
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    f"VARIANT MISMATCH — {vm}; wrong variant, routed to alternate, VERIFY")
                log.info("SKU %s — %s variant backstop: %s", sku, c.source_site, vm)
        # PACK BACKSTOP — same reason this is a SEPARATE pass: the demotion loop
        # skips already-rejected candidates, so a wrong-pack listing the LLM
        # rejected for pack reasons (mfidistribution's 10-count at $14 for a 300/Bx
        # order) never got pack_conflict and could still UNDERCUT the correct price.
        # HARD-exclude only on a GROSS pack ratio (≥3×) — a genuinely different unit
        # (10 vs 300). Mild differences (40 vs 50 tips, 1 vs 2 packs) are real
        # like-for-like alternatives and stay as APPROXIMATE rows with a pack note
        # (handled in the demotion loop above) so a correct match isn't nuked.
        if (not getattr(c, "pack_conflict", False) and pack_compatible(item, c) is False
                and item.pack_qty and c.pack_qty):
            ratio = max(item.pack_qty, c.pack_qty) / max(1, min(item.pack_qty, c.pack_qty))
            if ratio >= 3:
                c.pack_conflict = True
                if not (c.notes and "PACK MISMATCH" in c.notes):
                    c.notes = ((c.notes + " · ") if c.notes else "") + (
                        f"PACK MISMATCH — page is {c.pack_qty}/pack but ordered "
                        f"{item.pack_qty}/pack ({ratio:.0f}× off); different unit, "
                        f"routed to alternate")
                log.info("SKU %s — %s pack backstop: %s/pack vs ordered %s (%.0f× off)",
                         sku, c.source_site, c.pack_qty, item.pack_qty, ratio)

    # consistency guard: Groq may not call something exact its own criteria deny
    for c in verified:
        if c.match_type == "exact" and c.criteria and not _effective_exact(item, c):
            c.match_type = "approximate"
            log.info("SKU %s — demoted %s to approximate (criteria contradict exact)",
                     sku, c.source_site)
        # a candidate whose page was never successfully read cannot be EXACT —
        # the title alone is not page-level evidence (dead/expired listings)
        if c.match_type == "exact" and c.scraped_product_name is None:
            c.match_type = "approximate"
            c.notes = ((c.notes + " · ") if c.notes else "") + \
                "not page-verified — match based on listing title only, verify"
            log.info("SKU %s — demoted %s to approximate (page never verified)",
                     sku, c.source_site)
    # tag generic equivalents (brand differs but product/size/pack match) and
    # carry the variant-unverified flag forward (Q1 + Q2 decisions)
    for c in verified:
        crit = c.criteria or {}
        if (item.brand and crit.get("name_match") and crit.get("size_form_match")
                and crit.get("pack_match") and not crit.get("brand_match")):
            c.is_generic_equivalent = True

    # DETERMINISTIC MPN CONFIRMATION: the order's MPN appearing in a candidate's
    # URL or product name is definitive product identity — the SAME MPN is the
    # SAME product. This overrides the free model's unreliable name/size criteria
    # on aggregator-LOCKED pages, where the cheapest correct product was wrongly
    # excluded from price_match (supplyclinic Fuji A3.5 has '425004' in its URL at
    # the lowest price, yet a pricier normal-store match headlined). Sets name_match
    # so the strict gate passes; variant_conflict (a DETECTED wrong shade/color)
    # still blocks, so this can't promote a genuinely-different variant.
    if getattr(item, "mpn", None):
        _mnorm = re.sub(r"[^a-z0-9]", "", str(item.mpn).lower())
        if len(_mnorm) >= 5:
            for c in verified:
                hay = re.sub(r"[^a-z0-9]", "", (
                    (c.url or "") + " " + (getattr(c, "scraped_product_name", "") or "")
                    + " " + (c.title or "")).lower())
                if _mnorm in hay and not getattr(c, "variant_conflict", False):
                    c.mpn_confirmed = True
                    crit = dict(c.criteria or {})
                    crit["name_match"] = True
                    c.criteria = crit
                    log.info("SKU %s — %s MPN-confirmed (%s in listing) → price_match-eligible",
                             sku, c.source_site, item.mpn)

    # LOCKED-PRICE CONFIRMATION: aggregator-locked prices (the product parsed off
    # the page's own seller table) often get NO LLM criteria and score 0%, dropping
    # the cheapest correct product out of price_match (Genie Magic Mix supplyclinic,
    # which has no MPN to anchor on). When such a locked price is _likely_exact
    # (strong token overlap + pack-compatible) and has no hard variant conflict,
    # confirm its name so the strict gate passes — same principle as MPN-confirm.
    for c in verified:
        if (getattr(c, "price_locked", False) and not getattr(c, "variant_conflict", False)
                and not (c.criteria or {}).get("name_match") and _likely_exact(item, c)):
            crit = dict(c.criteria or {})
            crit["name_match"] = True
            c.criteria = crit
            log.info("SKU %s — %s locked-price token-confirmed → price_match-eligible",
                     sku, c.source_site)

    # MPN learning/validation from the real pages (confirms a seed MPN, learns new
    # ones, or overrides a wrong seed by consensus) — persisted for the next run.
    try:
        learn_mpn_from_pages(item, verified)
    except Exception:
        log.debug("SKU %s — MPN page-consensus skipped", sku)

    result.candidates.extend(verified)

    # INTRA-ITEM OUTLIER GUARD: a price far below this item's other page-verified
    # candidates is almost always a wrong extraction — a related/cross-sell item,
    # a single-unit price on a multi-pack page, or a stray number. (E.g. a $24.99
    # row when every other source for the same item is $250-300, or a $42.31 post
    # when the market floor is ~$63.) Flag such prices UNRELIABLE so they can't
    # headline as a trusted EXACT match; they still appear as options with a ⚠.
    import os as _os2
    out_ratio = float(_os2.environ.get("OUTLIER_LOW_RATIO", "0.45"))
    priced = sorted(c.price for c in verified
                    if c.price and c.price > 0 and c.scraped_product_name is not None)
    if len(priced) >= 3:
        mid = priced[len(priced) // 2]          # median of verified prices
        for c in verified:
            if (c.price and c.price > 0 and c.scraped_product_name is not None
                    and not getattr(c, "price_unreliable", False)
                    and not getattr(c, "price_locked", False)
                    and c.price < out_ratio * mid):
                c.price_unreliable = True
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    f"PRICE UNRELIABLE — ${c.price:.2f} is far below the other sources "
                    f"for this item (median ${mid:.2f}); likely a related-item or "
                    f"single-unit price, VERIFY MANUALLY")
                log.info("SKU %s — flagged %s $%.2f as low outlier vs median $%.2f",
                         sku, c.source_site, c.price, mid)

    # LOCKED-PRICE PEER SANITY: a deterministically-locked aggregator price is
    # normally the legit lowest, so it's exempt from the generic guard above.
    # But on a big multi-seller table (e.g. net32 with 20+ sellers and noisy
    # "Add $X more for free shipping" lines) the parser can still latch onto a
    # stray low number. Compare each locked price to the median of the OTHER
    # page-verified sources for this item (itself excluded so it can't dampen
    # the median); if it sits well below them, the seller-table parse is suspect
    # — flag UNRELIABLE (keeps it out of best_exact) rather than trusting it.
    # NOTE: net32 is a genuine discount marketplace — a CaviWipes canister really
    # is ~$10 there vs ~$16-18 at retail (ratio ~0.57). The old 0.6 floor wrongly
    # demoted those legit lows once the parser started returning accurate prices.
    # 0.5 still catches wrong-PRODUCT locks (Magic Mix $174 vs $581 median = 0.30,
    # mixing-tips $23 vs $80 = 0.29) which are the real failure mode.
    lock_ratio = float(_os2.environ.get("LOCK_OUTLIER_RATIO", "0.5"))
    for c in verified:
        if not (getattr(c, "price_locked", False) and c.price and c.price > 0
                and not getattr(c, "price_unreliable", False)):
            continue
        # ISSUE 2: do NOT suppress a locked price whose four criteria are ALL
        # confirmed. That is the RIGHT product on a discount marketplace, so being
        # half of retail peers is EXPECTED, not suspect (e.g. supplyclinic Teal
        # mixing tips $34.95 vs ~$76 retail — the genuine low the buyer wants). The
        # guard still fires on wrong-PRODUCT locks (Magic Mix $174 vs $581), because
        # those carry a name/size mismatch and so fail _effective_exact below.
        if _effective_exact(item, c):
            continue
        peers = sorted(o.price for o in verified
                       if o is not c and o.price and o.price > 0
                       and o.scraped_product_name is not None
                       and not getattr(o, "price_unreliable", False))
        if len(peers) < 2:
            continue                       # need ≥2 independent peers to judge
        peer_med = peers[len(peers) // 2]
        if c.price < lock_ratio * peer_med:
            c.price_unreliable = True
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                f"PRICE SUSPECT — locked ${c.price:.2f} is far below the other "
                f"sellers for this item (median ${peer_med:.2f}); the seller-table "
                f"parse may have caught a stray number, VERIFY")
            log.info("SKU %s — locked price %s $%.2f flagged vs peer median $%.2f",
                     sku, c.source_site, c.price, peer_med)

    # UNVERIFIED-SNIPPET-PRICE GUARD: when the scraped page body carries NO price
    # at all — a JS-rendered price we couldn't read (e.g. dentalcity's $294.99
    # renders client-side; with Firecrawl exhausted we only got the static shell)
    # — the only number left on the candidate is the stale SEARCH-SNIPPET price,
    # which can be a wrong variant or related item (dentalcity $24.95 vs the real
    # $294.99). Unless the price is a trusted structured single-offer or an
    # aggregator-locked value, a page that shows no price must never headline as
    # EXACT — flag it UNRELIABLE so it can't be the recommended best.
    _price_tok = re.compile(r"\$\s?\d")
    for c in verified:
        if (c.price and c.price > 0
                and not getattr(c, "price_structured", False)
                and not getattr(c, "price_locked", False)
                and not getattr(c, "price_unreliable", False)
                and isinstance(c.scraped_markdown, str)
                and len(c.scraped_markdown) > 200
                and not _price_tok.search(c.scraped_markdown)):
            c.price_unreliable = True
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                "PRICE UNVERIFIED — the scraped page shows no price (likely "
                "JS-rendered); this figure is the search-snippet price only and "
                "may be a wrong variant, VERIFY on the page")
            log.info("SKU %s — %s price $%.2f flagged UNVERIFIED (scraped page body "
                     "has no price)", sku, c.source_site, c.price)

    # OUT-OF-STOCK GUARD: a listing that is out of stock / no longer available is
    # not a buyable price, so it must never headline as the recommended best — it
    # would otherwise undercut a genuinely in-stock seller (orthazone's "NO LONGER
    # AVAILABLE" Glide $43.75 beating in-stock crazydental $44.99). Flag + demote
    # exact→approximate; it still appears as a clearly-labeled reference option.
    for c in verified:
        if c.match_type == "rejected":
            continue
        if _is_out_of_stock(c):
            c.out_of_stock = True
            if c.match_type == "exact":
                c.match_type = "approximate"
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                "OUT OF STOCK / no longer available on this page — not a buyable "
                "price; shown for reference only, VERIFY availability")
            log.info("SKU %s — %s flagged OUT OF STOCK (demoted; won't headline)",
                     sku, c.source_site)

    # Best exact: all four criteria true. Pack-mismatched candidates can NEVER
    # be primary, regardless of how cheap they are — they go to evidence with
    # their condition shown (PRD 4.5 / 4.6). Out-of-stock listings are excluded.
    exacts = [c for c in verified
              if c.match_type == "exact" and _effective_exact(item, c)
              and pack_compatible(item, c) is not False
              and not getattr(c, "price_unreliable", False)
              and not getattr(c, "out_of_stock", False)
              and c.price is not None]

    # BEST-EXACT PEER SANITY: the cheapest exact can still be a gross misextraction
    # the intra-item median guard above missed when smaller-pack candidates dragged
    # that median down (Clinpro $24.95 headlined EXACT while the real same-product
    # 100-pack was $241 — median was a 30-pack $36, so $24.95 didn't look low).
    # Compare the chosen best ONLY against its effective-exact peers (same product,
    # right pack — a clean baseline, no pack-mismatched dilution) with a TIGHTER
    # ratio than the generic guard: genuine discount-marketplace lows (Teal $34.95
    # vs $76 = .46) survive, only ~4-10x errors (Clinpro .10) are demoted. The
    # demoted price stays as a ⚠ UNVERIFIED option; the next-cheapest exact heads.
    best_ratio = float(_os2.environ.get("BEST_OUTLIER_RATIO", "0.25"))
    exacts = sorted(exacts, key=lambda c: c.price)
    while exacts:
        cand = exacts[0]
        peers = sorted(o.price for o in exacts[1:]
                       if o.price and o.price > 0 and price_sane(item, o))
        if peers and cand.price < best_ratio * peers[len(peers) // 2]:
            peer_med = peers[len(peers) // 2]
            cand.price_unreliable = True
            cand.notes = ((cand.notes + " · ") if cand.notes else "") + (
                f"PRICE UNRELIABLE — ${cand.price:.2f} is far below the verified "
                f"same-product sellers (median ${peer_med:.2f}); almost certainly a "
                f"related-item or wrong-pack price, VERIFY before using")
            log.info("SKU %s — best-exact %s $%.2f demoted vs exact-peer median "
                     "$%.2f (ratio %.2f)", sku, cand.source_site, cand.price,
                     peer_med, cand.price / peer_med)
            exacts.pop(0)
            continue
        break
    if exacts:
        result.best_exact = exacts[0]
    return result


# ------------------------------------------------------------------- MPN ----

def load_mpn_table(config_dir: Path) -> dict:
    """Client-maintained SKU→MPN map (config/mpn_table.txt), pipe-delimited:
       HENRY_SCHEIN_SKU | MPN | manufacturer (optional) | note (optional)
    Henry Schein descriptions rarely print the manufacturer part number, so this
    table supplies it per SKU — the single most precise match signal."""
    f = config_dir / "mpn_table.txt"
    table: dict = {}
    if not f.exists():
        return table
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip()          # trailing # comment
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        table[parts[0]] = (parts[1], parts[2] if len(parts) > 2 and parts[2] else None)
    return table


_MPN_NORM_RE = re.compile(r"[^A-Z0-9]")


def _norm_mpn(s) -> str:
    return _MPN_NORM_RE.sub("", str(s or "").upper())


# Patterns where a product page DECLARES its manufacturer part number.
_PAGE_MPN_RES = (
    re.compile(r"manu(?:facturer)?\s*(?:code|part(?:\s*(?:#|no\.?|number))?)\s*[:#]?\s*"
               r"([A-Za-z0-9][A-Za-z0-9./\-]{3,})", re.I),
    re.compile(r"mfg\.?\s*part\s*(?:#|no\.?|number)?\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9./\-]{3,})", re.I),
    re.compile(r"\bMPN\b\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9./\-]{3,})", re.I),
)


def _page_mpns(c) -> list:
    """Manufacturer part numbers a candidate page explicitly declares (original form)."""
    text = " ".join(p for p in (getattr(c, "scraped_product_name", None),
                                (getattr(c, "scraped_markdown", None) or "")[:4000]) if p)
    out = []
    for rx in _PAGE_MPN_RES:
        out.extend(m.group(1) for m in rx.finditer(text))
    return out


def _page_declares(c, mpn) -> bool:
    """True if the (normalized) MPN literally appears anywhere on the candidate."""
    n = _norm_mpn(mpn)
    if len(n) < 4:
        return False
    hay = _norm_mpn(" ".join(p for p in (
        getattr(c, "scraped_product_name", None), getattr(c, "title", None),
        getattr(c, "url", None), (getattr(c, "scraped_markdown", None) or "")[:4000]) if p))
    return n in hay


def seed_and_apply_mpn(conn, items, config_dir: Path) -> int:
    """Seed the MPN store from config/mpn_table.txt (manual), run the internal-
    consistency check, then set each item's mpn / mpn_query / mpn_verified from the
    store. An MPN is only mpn_verified (usable to REJECT conflicting listings) once
    page-consensus has confirmed it; an unconfirmed seed is a DISCOVERY HINT only,
    so a wrong manual MPN can't wrongly reject a correct page. Returns # enriched."""
    from . import db
    store = db.get_mpn_store(conn) if conn else {}
    # seed manual entries not already in the store
    if conn:
        for sku, (mpn, mfr) in load_mpn_table(config_dir).items():
            if sku not in store:
                db.upsert_mpn(conn, sku, mpn, mfr, source="manual", status="seed")
        store = db.get_mpn_store(conn)

    # VALIDATION 1 — internal consistency: a manual MPN shared across DIFFERENT
    # products in this order (e.g. Fuji A3 and A3.5 both '425003') is an obvious
    # sheet error → mark both 'conflict' so neither is trusted for rejection.
    by_mpn: dict = {}
    for it in items:
        rec = store.get(it.schein_sku)
        if rec and rec.get("mpn"):
            by_mpn.setdefault(_norm_mpn(rec["mpn"]), []).append(it)
    for _nm, its in by_mpn.items():
        if len(its) > 1 and len({i.description for i in its}) > 1:
            for i in its:
                rec = store.get(i.schein_sku)
                if rec and rec.get("status") == "seed" and conn:
                    db.upsert_mpn(conn, i.schein_sku, rec["mpn"], rec["manufacturer"],
                                  source=rec["source"], status="conflict")
                    rec["status"] = "conflict"
                    log.warning("MPN check — SKU %s MPN %s also used for a DIFFERENT "
                                "product in this order; marking SUSPECT (not used to reject)",
                                i.schein_sku, rec["mpn"])

    n = 0
    for it in items:
        rec = store.get(it.schein_sku)
        if not rec or not rec.get("mpn"):
            continue
        it.mpn = rec["mpn"]
        prod = (it.product_name or it.description or "").strip()
        it.mpn_query = " ".join(p for p in (rec.get("manufacturer") or it.brand or "",
                                            rec["mpn"], prod) if p).strip() or None
        # VALIDATION 2 gate: only a page-confirmed ('verified') MPN may REJECT a
        # listing. Seed/conflict → discovery hint only (matching ignores it).
        it.mpn_verified = rec.get("status") == "verified"
        n += 1
    return n


def learn_mpn_from_pages(item, candidates) -> None:
    """Page-consensus (VALIDATION 2 + dynamic fill): from the candidates the model
    confirmed as the SAME product, (a) if a page declares the stored MPN → mark it
    verified (the sheet was right), else (b) if ≥2 same-product pages agree on a
    part number → store THAT as the verified MPN (learns new SKUs and overrides a
    wrong seed). Thread-safe immediate write so the next run reuses it."""
    from . import db
    good = [c for c in candidates
            if (c.criteria or {}).get("name_match") and getattr(c, "scraped_product_name", None)]
    if not good:
        return
    stored = getattr(item, "mpn", None)
    if stored and any(_page_declares(c, stored) for c in good):
        db.upsert_mpn_now(item.schein_sku, stored, source="page-confirmed", status="verified")
        return
    from collections import Counter
    cnt: Counter = Counter()
    rep: dict = {}
    for c in good:
        for m in _page_mpns(c):
            nm = _norm_mpn(m)
            if len(nm) >= 4:
                cnt[nm] += 1
                rep.setdefault(nm, m)
    if cnt:
        nm, k = cnt.most_common(1)[0]
        if k >= 2:
            db.upsert_mpn_now(item.schein_sku, rep[nm], source="page-consensus", status="verified")
            log.info("SKU %s — learned MPN %s by page consensus (%d same-product pages agree)",
                     item.schein_sku, rep[nm], k)


# ------------------------------------------------------------ equivalency ---

def load_equivalency_table(config_dir: Path) -> List[EquivalencyEntry]:
    """Client-maintained plain-text file, pipe-delimited:
       SCHEIN_SKU | equivalent product name | brand (optional) | note (optional)
    """
    f = config_dir / "equivalency_table.txt"
    entries: List[EquivalencyEntry] = []
    if not f.exists():
        return entries
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        entries.append(EquivalencyEntry(
            schein_sku=parts[0], equivalent_name=parts[1],
            equivalent_brand=parts[2] if len(parts) > 2 and parts[2] else None,
            notes=parts[3] if len(parts) > 3 else None,
        ))
    return entries


def run_equivalency(result: ItemResult, entries: List[EquivalencyEntry]) -> Optional[EquivalencyFinding]:
    """Stage 2: only fires when the equivalency table has a mapping for this
    SKU (table-driven, not price-threshold-driven — Issue 6 fixed)."""
    item = result.item
    matches = [e for e in entries if e.schein_sku == item.schein_sku]
    if not matches:
        return None
    e = matches[0]

    # price the equivalent on the open market
    probe = OrderLineItem(
        qty=item.qty, schein_sku=item.schein_sku,
        description=e.equivalent_name, uom=item.uom,
        unit_price=item.unit_price, extended_price=item.extended_price,
        search_query=f"{e.equivalent_brand or ''} {e.equivalent_name}".strip(),
        pack_qty=item.pack_qty,
    )
    cands, _ = search.market_sweep(probe, max_candidates=4)
    best = None
    for c in cands[:3]:
        c = search.firecrawl_verify(c, sku=item.schein_sku)
        if c.price and (best is None or c.price < best.price):
            best = c

    market = (f"${best.price} at {best.source_site} ({best.url})" if best else "no public price found")
    verdict = ai.evaluate_equivalency(item, e.equivalent_name, e.notes or "", market)
    level = verdict.get("confidence_level", "possible_alternative")

    finding = EquivalencyFinding(
        item=item, equivalent_name=e.equivalent_name,
        confidence_level=level, basis=verdict.get("basis", ""),
        supplier=best.source_site if best else None,
        url=best.url if best else None,
        price=best.price if best else None,
        pack_condition=best.pack_condition if best else None,
        est_savings_total=(round((item.unit_price - best.price) * item.qty, 2)
                           if best and best.price else None),
    )
    # Stream separation (PRD 5.5): exact/close equivalents leave the
    # negotiation report entirely.
    if level in ("exact_equivalent", "close_equivalent"):
        result.routed_to_alternate = True
    return finding