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


def price_sane(item: OrderLineItem, c: PriceCandidate) -> bool:
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


def volume_compatible(item: OrderLineItem, c: PriceCandidate) -> Optional[bool]:
    a = normalize_volume_ml(item.description)
    b = normalize_volume_ml((c.scraped_product_name or "") + " " + (c.title or ""))
    if a is None or b is None or a <= 0:
        return None
    return abs(a - b) / a <= 0.02



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

    # verify the most promising candidates on-page (JS rendered)
    verified: List[PriceCandidate] = []
    exacts_found = 0
    verified_ok = 0
    for c in cands:
        # early stop: once 2 verified exacts + 4 verified candidates exist,
        # skip remaining PRICED candidates — the shortlist is cheapest-first,
        # so they are provably more expensive than what's already verified.
        # Price-UNKNOWN candidates (reserve slots) are always scraped: one of
        # them could be the true lowest price, invisible until the page is read.
        if exacts_found >= 2 and verified_ok >= 4 and c.price is not None:
            c.notes = c.notes or "not scraped — enough verified candidates found"
            result.candidates.append(c)
            continue
        if len([v for v in verified if v.match_type != "rejected"]) >= max_verify:
            result.candidates.append(c)       # unverified leftovers -> evidence
            continue
        c._ordered_variant = item.variant or ""
        c = search.firecrawl_verify(c)
        # deterministic demotions before AI sees them
        if c.match_type != "rejected":
            if not price_sane(item, c):
                c.match_type = "rejected"
                c.rejected_reason = (c.rejected_reason or
                                     f"price {c.price} failed sanity vs Schein unit {item.unit_price}")
            elif pack_compatible(item, c) is False and c.pack_condition is None:
                c.pack_condition = f"{c.pack_qty}-pack price (ordered pack: {item.pack_qty})"
            elif volume_compatible(item, c) is False:
                c.match_type = "rejected"
                c.rejected_reason = "volume/size mismatch after normalization"
        if c.scraped_product_name is not None:
            verified_ok += 1
            if _likely_exact(item, c):
                exacts_found += 1
        verified.append(c)

    to_validate = [c for c in verified if c.match_type != "rejected"]
    ai.validate_candidates(item, to_validate)
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

    # Best exact: all four criteria true. Pack-mismatched candidates can NEVER
    # be primary, regardless of how cheap they are — they go to evidence with
    # their condition shown (PRD 4.5 / 4.6).
    exacts = [c for c in verified
              if c.match_type == "exact" and _effective_exact(item, c)
              and pack_compatible(item, c) is not False
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
        c = search.firecrawl_verify(c)
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