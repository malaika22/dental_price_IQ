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
    if a is None or b is None:
        return None
    return abs(a - b) / a <= 0.02


# --------------------------------------------------------------- pipeline ---

def process_item(item: OrderLineItem, max_verify: int = 5) -> ItemResult:
    sku = item.schein_sku
    log.info(
        "SKU %s — processing started: %r (Schein unit=$%.2f)",
        sku, item.description[:50], item.unit_price,
    )
    result = ItemResult(item=item)
    cands, flagged = search.market_sweep(item)
    result.flagged_sites = flagged

    # verify the most promising candidates on-page (JS rendered)
    verified: List[PriceCandidate] = []
    verify_num = 0
    for c in cands:
        if len([v for v in verified if v.match_type != "rejected"]) >= max_verify:
            log.debug("SKU %s — skipping unverified candidate %s (max_verify=%d reached)",
                      sku, c.url[:60], max_verify)
            result.candidates.append(c)       # unverified leftovers -> evidence
            continue
        verify_num += 1
        log.info("SKU %s — verifying candidate %d: %s", sku, verify_num, c.url[:80])
        c = search.firecrawl_verify(c, sku=sku)
        # deterministic demotions before AI sees them
        if c.match_type != "rejected":
            if not price_sane(item, c):
                c.match_type = "rejected"
                c.rejected_reason = (c.rejected_reason or
                                     f"price {c.price} failed sanity vs Schein unit {item.unit_price}")
                log.info("SKU %s — rejected (price sanity): $%s vs Schein $%.2f",
                         sku, c.price, item.unit_price)
            elif pack_compatible(item, c) is False and c.pack_condition is None:
                c.pack_condition = f"{c.pack_qty}-pack price (ordered pack: {item.pack_qty})"
                log.info("SKU %s — pack mismatch noted: ordered %s, page %s",
                         sku, item.pack_qty, c.pack_qty)
            elif volume_compatible(item, c) is False:
                c.match_type = "rejected"
                c.rejected_reason = "volume/size mismatch after normalization"
                log.info("SKU %s — rejected (volume mismatch)", sku)
        verified.append(c)

    to_validate = [c for c in verified if c.match_type != "rejected"]
    if to_validate:
        log.info("SKU %s — sending %d candidate(s) to Groq validation", sku, len(to_validate))
        ai.validate_candidates(item, to_validate)
    else:
        log.warning("SKU %s — no candidates survived pre-checks for Groq validation", sku)
    result.candidates.extend(verified)

    # Best exact: all four criteria true. Pack-mismatched candidates can NEVER
    # be primary, regardless of how cheap they are — they go to evidence with
    # their condition shown (PRD 4.5 / 4.6).
    exacts = [c for c in verified
              if c.match_type == "exact" and all(c.criteria.get(k) for k in
                 ("brand_match", "name_match", "size_form_match", "pack_match"))
              and c.price is not None]
    if exacts:
        result.best_exact = min(exacts, key=lambda c: c.price)
        savings = round(item.unit_price - result.best_exact.price, 2)
        log.info(
            "SKU %s — best exact match: $%s at %s (saves $%.2f/unit) %s",
            sku, result.best_exact.price, result.best_exact.source_site,
            savings, result.best_exact.url[:80],
        )
    else:
        log.info("SKU %s — no exact match found (%d candidate(s) in evidence)",
                 sku, len(result.candidates))
    return result


# ------------------------------------------------------------ equivalency ---

def load_equivalency_table(config_dir: Path) -> List[EquivalencyEntry]:
    """Client-maintained plain-text file, pipe-delimited:
       SCHEIN_SKU | equivalent product name | brand (optional) | note (optional)
    """
    f = config_dir / "equivalency_table.txt"
    entries: List[EquivalencyEntry] = []
    if not f.exists():
        log.warning("Equivalency table not found at %s", f)
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
    log.info("Loaded %d equivalency entries from %s", len(entries), f.name)
    return entries


def run_equivalency(result: ItemResult, entries: List[EquivalencyEntry]) -> Optional[EquivalencyFinding]:
    """Stage 2: only fires when the equivalency table has a mapping for this
    SKU (table-driven, not price-threshold-driven — Issue 6 fixed)."""
    item = result.item
    matches = [e for e in entries if e.schein_sku == item.schein_sku]
    if not matches:
        return None
    e = matches[0]
    log.info(
        "SKU %s — equivalency lookup: %r → %r",
        item.schein_sku, item.description[:40], e.equivalent_name[:40],
    )

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
    if best:
        log.info("SKU %s — equivalent best price: $%s at %s", item.schein_sku, best.price, best.source_site)
    else:
        log.warning("SKU %s — no public price found for equivalent %r", item.schein_sku, e.equivalent_name)

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
        log.info("SKU %s — routed to alternate purchases (%s)", item.schein_sku, level)
    else:
        log.info("SKU %s — equivalency %s (evidence only)", item.schein_sku, level)
    return finding
