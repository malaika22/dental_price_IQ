"""Exact-match layer (Stage 1) and equivalency layer (Stage 2).

Matching = cheap deterministic pre-checks first (price sanity, volume
normalization, pack tolerance), then a single batched Groq (Llama 3.3 70B) validation per
item applying the four trust criteria. Only candidates with all four criteria
confirmed enter the primary price-match report (PRD 4.5).
"""
from __future__ import annotations
import logging
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
    _coerce_price(c)
    """Reject obviously-wrong extractions (dimension-like numbers, unit-price
    of a single piece vs a 100-pack, etc.)."""
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


def _flavor_in(text: str) -> Optional[str]:
    t = (text or "").lower()
    for f in _FLAVORS:           # specific flavors first; generic 'mint' last
        if f in t:
            return f
    return None


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
    of, cf = _flavor_in(ordered_txt), _flavor_in(cand_name)
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
    return None



def _brand_ok(item: OrderLineItem, c: PriceCandidate) -> bool:
    """Brand criterion passes when confirmed, or when the reference item has
    no identifiable brand (generic/house items like 'Barrier Film Blue')."""
    return bool((c.criteria or {}).get("brand_match")) or not item.brand


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


def _effective_exact(item: OrderLineItem, c: PriceCandidate) -> bool:
    crit = c.criteria or {}
    return (crit.get("name_match") and crit.get("size_form_match")
            and crit.get("pack_match") and _brand_ok(item, c))

# --------------------------------------------------------------- pipeline ---

def process_item(item: OrderLineItem, max_verify: int = 8) -> ItemResult:
    sku = item.schein_sku
    result = ItemResult(item=item)
    cands, flagged = search.market_sweep(item)
    result.flagged_sites = flagged

    # verify the most promising candidates on-page (JS rendered). Batched Groq
    # extraction means prices aren't known mid-loop, so instead of a dynamic
    # early-stop we cap scrapes per item (cheapest-first shortlist preserves the
    # lowest-priced candidates). SCRAPE_CAP_PER_ITEM bounds Firecrawl cost.
    import os as _os
    scrape_cap = int(_os.environ.get("SCRAPE_CAP_PER_ITEM", "5"))
    verified: List[PriceCandidate] = []
    exacts_found = 0
    verified_ok = 0
    scraped_n = 0
    for c in cands:
        if scraped_n >= scrape_cap:
            c.notes = c.notes or "not scraped — per-item scrape cap reached"
            result.candidates.append(c)
            continue
        c._ordered_variant = item.variant or ""
        c = search.firecrawl_verify(c, sku=sku)   # markdown scrape (1 credit); no price yet
        scraped_n += 1
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
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                f"PACK MISMATCH — page is {c.pack_qty}/pack but ordered "
                f"{item.pack_qty}/pack; treated as approximate, VERIFY")
            if c.pack_condition is None:
                c.pack_condition = f"{c.pack_qty}-pack price (ordered pack: {item.pack_qty})"
        elif volume_compatible(item, c) is False:
            c.match_type = "rejected"
            c.rejected_reason = "volume/size mismatch after normalization"
        # VARIANT BACKSTOP: deterministic flavor/shade/size check — demote (never
        # reject) a candidate whose variant conflicts with the order, so a
        # wrong-flavor / wrong-shade / wrong-size page can't headline as EXACT.
        if c.match_type != "rejected":
            vm = variant_mismatch(item, c)
            if vm:
                c.variant_unverified = True
                if c.match_type == "exact":
                    c.match_type = "approximate"
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    f"VARIANT MISMATCH — {vm}; treated as approximate, VERIFY")
                log.info("SKU %s — %s variant backstop: %s", sku, c.source_site, vm)
        if c.scraped_product_name is not None:
            verified_ok += 1
            if _likely_exact(item, c):
                exacts_found += 1

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
    lock_ratio = float(_os2.environ.get("LOCK_OUTLIER_RATIO", "0.6"))
    for c in verified:
        if not (getattr(c, "price_locked", False) and c.price and c.price > 0
                and not getattr(c, "price_unreliable", False)):
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

    # Best exact: all four criteria true. Pack-mismatched candidates can NEVER
    # be primary, regardless of how cheap they are — they go to evidence with
    # their condition shown (PRD 4.5 / 4.6).
    exacts = [c for c in verified
              if c.match_type == "exact" and _effective_exact(item, c)
              and pack_compatible(item, c) is not False
              and not getattr(c, "price_unreliable", False)
              and c.price is not None]
    if exacts:
        result.best_exact = min(exacts, key=lambda c: c.price)
    return result


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