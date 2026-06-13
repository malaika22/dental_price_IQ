"""Report generation — the three output files (PRD 4.7, 4.8, 5.4).

Layout (matches the approved reference styling):
- Row 1: merged title banner — report name · Ref · Date · Patient · Generated
- Row 2: merged legend line (italic gray)
- Row 3: navy header row, white bold, wrapped
- Data rows: banded light green, thin borders, wrapped text, savings
  highlighting (🟢 >10%, 🟡 5–10%) layered on top, clickable URLs.

Price-match report: row selection logic UNCHANGED — exact four-criteria
matches only, cheaper than Schein only, sorted by total savings descending.
Match Type + Score render as one column: "EXACT (97%)".

Alternate purchase list: every reviewable scraped candidate that matches a
product NOT in the price-match report (plus equivalency-table findings).
Columns follow the approved reference sheet:
  Original Schein Product and Schein Price | Recommended Equivalent Product |
  Recommended Supplier | Price of the Equivalent | Product URL | Match Score |
  Equivalency Basis and Confidence Level | Estimated Savings vs. Schein Price
Tiers: EXACT = same product · CLOSE = compatible specs · POSSIBLE = needs review.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .models import EquivalencyFinding, ItemResult, ParsedOrder, PriceCandidate

log = logging.getLogger(__name__)

# ------------------------------------------------------------- styling ------

NAVY_FILL = PatternFill("solid", fgColor="1F3650")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(bold=True, size=12)
LEGEND_FONT = Font(italic=True, size=9, color="595959")
LINK_FONT = Font(color="0563C1", underline="single", size=9)
BAND_FILL = PatternFill("solid", fgColor="EAF3EA")     # light green banding
GREEN_FILL = PatternFill("solid", fgColor="C6EFCE")    # >10% savings
YELLOW_FILL = PatternFill("solid", fgColor="FFEB9C")   # 5–10% savings
BOLD = Font(bold=True)
_thin = Side(style="thin", color="C9C9C9")
CELL_BORDER = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)
_green = Side(style="medium", color="538135")
TITLE_BORDER = Border(top=_green, bottom=_green)
WRAP = Alignment(wrap_text=True, vertical="center")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
MONEY = '"$"#,##0.00'

PM_LEGEND = ("🟢 >10% savings   🟡 5–10% savings   EXACT = all four trust criteria "
             "confirmed (brand · product name · size/form · pack qty)   "
             "Product URL links to the verified product page")
ALT_LEGEND = ("Includes equivalency-table substitutions and all matching scraped "
              "prices not in the Price Match report  ·  EXACT = same product · "
              "CLOSE = compatible specs · POSSIBLE = needs review")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _dash(v) -> object:
    """'—' placeholder for empty cells (matches reference styling)."""
    if v is None:
        return "—"
    s = str(v).strip()
    return "—" if s == "" or s.lower() in ("none", "null", "n/a", "na", "-") else v


def _clean(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("none", "null", "n/a", "na", "-") else s


def match_score(criteria: Optional[dict], confidence) -> int:
    """0–100: 60% weight on four-criteria coverage, 40% on AI confidence."""
    crit = criteria or {}
    met = sum(1 for k in ("brand_match", "name_match", "size_form_match", "pack_match")
              if crit.get(k))
    try:
        conf = int(confidence or 0)
    except (TypeError, ValueError):
        conf = 0
    return round(met / 4 * 60 + conf * 0.4)


def search_fallback_url(item) -> str:
    q = item.search_query or item.description
    return f"https://www.google.com/search?tbm=shop&q={quote_plus(q)}"


def _title_block(ws, title: str, legend: str, ncols: int,
                 headers: List[str], widths: List[int]):
    ws.append([title])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    c = ws.cell(row=1, column=1)
    c.font, c.alignment = TITLE_FONT, CENTER
    for col in range(1, ncols + 1):
        ws.cell(row=1, column=col).border = TITLE_BORDER
    ws.row_dimensions[1].height = 26

    ws.append([legend])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    c = ws.cell(row=2, column=1)
    c.font, c.alignment = LEGEND_FONT, CENTER
    ws.row_dimensions[2].height = 16

    ws.append(headers)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
        cell = ws.cell(row=3, column=i)
        cell.fill, cell.font, cell.alignment = NAVY_FILL, HEADER_FONT, CENTER
        cell.border = CELL_BORDER
    ws.row_dimensions[3].height = 32
    ws.freeze_panes = "A4"


def _style_row(ws, ridx: int, ncols: int, savings_pct: Optional[float],
               band: bool, wrap_cols: set[int], link_col: Optional[int] = None,
               url: str = "", bold_cols: set[int] = frozenset()):
    fill = None
    if savings_pct is not None:
        fill = GREEN_FILL if savings_pct > 10 else (YELLOW_FILL if 5 <= savings_pct <= 10 else None)
    if fill is None and band:
        fill = BAND_FILL
    for col in range(1, ncols + 1):
        cell = ws.cell(row=ridx, column=col)
        cell.border = CELL_BORDER
        cell.alignment = WRAP if col in wrap_cols else Alignment(vertical="center")
        if fill is not None:
            cell.fill = fill
        if col in bold_cols:
            cell.font = BOLD
    if link_col and url:
        cell = ws.cell(row=ridx, column=link_col)
        cell.value = url
        cell.hyperlink = url
        cell.font = LINK_FONT
        cell.alignment = WRAP


def _money(ws, ridx: int, cols: List[int]):
    for c in cols:
        cell = ws.cell(row=ridx, column=c)
        if isinstance(cell.value, (int, float)):
            cell.number_format = MONEY


def _savings_pct(per_unit: float, unit_price: float) -> Optional[float]:
    if not unit_price:
        return None
    return per_unit / unit_price * 100


def _price_match_options(r: ItemResult, options_per_item: int = 3) -> List[PriceCandidate]:
    """Same option selection as write_price_match_report — used for stream separation."""
    if r.routed_to_alternate:
        return []
    opts: List[PriceCandidate] = []
    exacts = [c for c in r.candidates
              if c.price is not None and _is_exact_cand(r.item, c)
              and not _pack_mismatch(r.item, c) and not _is_gated(c)]
    if exacts:
        opts.append(min(exacts, key=lambda c: c.price))
    for c in _option_pool(r):
        if len(opts) >= options_per_item:
            break
        if opts and c is opts[0]:
            continue
        opts.append(c)
    return opts


def _appears_in_price_match(r: ItemResult) -> bool:
    return bool(_price_match_options(r))


def _reviewable(c: PriceCandidate) -> bool:
    if c.price is None:
        return False
    reason = (c.rejected_reason or "").lower()
    return "login" not in reason and "sanity" not in reason


# tier mapping for the alternate sheet
_TIER = {
    "exact": ("EXACT", "EXACT — same product, different brand or channel; substitution is straightforward", 0),
    "approximate": ("APPROXIMATE", "CLOSE — same category, compatible specs, minor differences; substitution likely acceptable", 1),
    "rejected": ("POSSIBLE", "POSSIBLE — related product, requires client review before acting", 2),
    "unverified": ("POSSIBLE", "POSSIBLE — unverified listing, requires client review before acting", 3),
}
_CRIT_LABEL = {"brand_match": "brand", "name_match": "product name",
               "size_form_match": "form/spec", "pack_match": "pack size"}


def _basis_text(c: PriceCandidate) -> str:
    label, sentence, _ = _TIER.get(c.match_type, _TIER["unverified"])
    parts = [f"Web search (not in Price Match) · {label} ({match_score(c.criteria, c.confidence)}%)"]
    matched = [_CRIT_LABEL[k] for k, v in (c.criteria or {}).items() if v]
    missed = [_CRIT_LABEL[k] for k, v in (c.criteria or {}).items() if v is False]
    if matched:
        parts.append("Matching: " + ", ".join(matched))
    if missed:
        parts.append("Not matching: " + ", ".join(missed))
    note = _clean(c.notes) or _clean(c.rejected_reason)
    if note:
        parts.append(note)
    return sentence + "\n" + " · ".join(parts)


# ------------------------------------------------------ primary report ------

PM_HEADERS = ["Schein SKU", "Manufacturer Part\nNumber", "Description", "Qty\nOrder",
              "Schein Unit\nPrice", "Best Public Price\nFound", "Match Score",
              "Source Site", "Product URL Link",
              'Pack/Qty Condition\n(e.g. "6-pack price")',
              "Why Not Exact / Notes", "Savings Per\nUnit", "Total Savings"]
PM_WIDTHS = [11, 15, 38, 7, 12, 13, 17, 17, 42, 18, 40, 12, 12]
PM_WRAP = {3, 9, 10, 11}
OPTION_FONT = Font(italic=True, color="7F7F7F")

_OPT_RANK = {"exact": 0, "approximate": 1, "rejected": 2, "unverified": 3}


def _is_gated(c: PriceCandidate) -> bool:
    return "login" in (c.rejected_reason or "").lower()


def _pack_mismatch(item, c: PriceCandidate) -> bool:
    """Hard exclusion from the price-match report: a known different pack
    size is never negotiation-grade, no matter how cheap (PRD 4.5)."""
    if (c.criteria or {}).get("pack_match") is False:
        return True
    try:
        if c.pack_qty and item.pack_qty and int(c.pack_qty) != int(item.pack_qty):
            return True
    except (TypeError, ValueError):
        pass
    cond = (_clean(c.pack_condition) or "").lower()
    return any(k in cond for k in
               ("smaller pack", "larger pack", "case lot", "single unit",
                "instead of", "-pack price (ordered pack"))


def _brand_ok(item, c: PriceCandidate) -> bool:
    return bool((c.criteria or {}).get("brand_match")) or not getattr(item, "brand", None)


def _option_pool(r: ItemResult) -> list[PriceCandidate]:
    """Non-exact candidates eligible as options: approximate first, then
    verified-but-rejected (variant/brand mismatch), then unverified. Login-
    gated candidates with a search-listed price are allowed LAST, clearly
    labeled, only to fill otherwise-empty slots. Sanity rejects never appear.
    Within the leading rank, near-tied scores (15 pts) compete on price."""
    pool, seen = [], set()
    for c in r.candidates:
        if c.price is None or c.price <= 0 or c.url in seen:
            continue
        if "sanity" in (c.rejected_reason or "").lower():
            continue
        if _pack_mismatch(r.item, c):
            continue                      # stays in alternate + evidence only
        seen.add(c.url)
        pool.append(c)

    def key(c):
        rank = 5 if _is_gated(c) else _OPT_RANK.get(c.match_type, 3)
        return (rank, -match_score(c.criteria, c.confidence), c.price)

    pool.sort(key=key)
    if len(pool) > 1:
        lead = 5 if _is_gated(pool[0]) else _OPT_RANK.get(pool[0].match_type, 3)
        block = [c for c in pool if (5 if _is_gated(c) else _OPT_RANK.get(c.match_type, 3)) == lead]
        rest = [c for c in pool if c not in block]
        smax = max(match_score(c.criteria, c.confidence) for c in block)
        block.sort(key=lambda c: (match_score(c.criteria, c.confidence) < smax - 15, c.price))
        pool = block + rest
    return pool


def _why_not_exact(c: PriceCandidate) -> str:
    if _is_gated(c):
        return ("GATED — pricing requires login/membership on this site; price shown "
                "comes from the public search listing. Verify manually before negotiating.")
    missed = [_CRIT_LABEL[k] for k, v in (c.criteria or {}).items() if v is False]
    note = _clean(c.notes) or _clean(c.rejected_reason)
    parts = []
    if missed:
        parts.append("Not exact — mismatch on: " + ", ".join(missed))
    if note:
        parts.append(note)
    return " · ".join(parts) or "Not exact — could not confirm all four trust criteria"


def _score_label(item, c: PriceCandidate) -> str:
    """Label derives from the criteria themselves, never from a raw
    match_type that contradicts them."""
    if _is_gated(c):
        return f"GATED ({match_score(c.criteria, c.confidence)}%)"
    if _is_exact_cand(item, c):
        label = "EXACT"
    elif c.match_type in ("exact", "approximate"):
        label = "APPROXIMATE"
    else:
        label = "POSSIBLE"
    return f"{label} ({match_score(c.criteria, c.confidence)}%)"


def _is_exact_cand(item, c: PriceCandidate) -> bool:
    crit = c.criteria or {}
    return (crit.get("name_match") and crit.get("size_form_match")
            and crit.get("pack_match") and _brand_ok(item, c))


def write_price_match_report(order: ParsedOrder, results: List[ItemResult],
                             out: Path, options_per_item: int = 3) -> Path:
    """Primary negotiation report — up to 3 options per item.
    Row layout (per approved screenshot): the item's main row carries the full
    item details with Option 1; Options 2-3 render as indented "↳ Option N"
    sub-rows (SKU/MPN/Qty/Schein-price left blank) directly beneath it.
    Item groups are sorted by their best total savings descending."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Price Match"
    title = (f"Henry Schein Price Match Negotiation  ·  Ref: {order.reference or '—'}"
             f"  ·  Date: {order.order_date or '—'}"
             f"  ·  Patient: {getattr(order, 'ship_to_name', None) or '—'}"
             f"  ·  Generated: {_now()}")
    legend = ("🟢 >10% savings   🟡 5–10% savings   Main row = best option (EXACT when "
              "available) · ↳ Option 2-3 = next-closest matches with reasoning · "
              "GATED = login/membership pricing, verify manually")
    _title_block(ws, title, legend, len(PM_HEADERS), PM_HEADERS, PM_WIDTHS)

    groups = []
    for r in results:
        opts = _price_match_options(r, options_per_item)
        if not opts:
            continue
        best_total = max(round((r.item.unit_price - c.price) * r.item.qty, 2)
                         for c in opts)
        groups.append((best_total, r, opts))

    groups.sort(key=lambda g: g[0], reverse=True)
    band = 0
    for _, r, opts in groups:
        for n, c in enumerate(opts, start=1):
            per_unit = round(r.item.unit_price - c.price, 2)
            total = round(per_unit * r.item.qty, 2)
            pct = _savings_pct(per_unit, r.item.unit_price)
            if _is_exact_cand(r.item, c):
                reason = ("Exact — all four trust criteria confirmed"
                          if n == 1 else
                          "Exact — all four trust criteria confirmed (alternate source, higher price than Option 1)")
            else:
                reason = _why_not_exact(c)

            if n == 1:
                ws.append([r.item.schein_sku, _dash(r.item.mpn), r.item.description,
                           r.item.qty, r.item.unit_price, c.price, _score_label(r.item, c),
                           c.source_site, c.url, _dash(_clean(c.pack_condition)),
                           reason, per_unit, total])
                ridx = ws.max_row
                _style_row(ws, ridx, len(PM_HEADERS), pct, band % 2 == 0, PM_WRAP,
                           link_col=9, url=c.url, bold_cols={13})
                _money(ws, ridx, [5, 6, 12, 13])
            else:
                ws.append(["", "", f"   ↳ Option {n}", "", "", c.price,
                           _score_label(r.item, c), c.source_site, c.url,
                           _dash(_clean(c.pack_condition)), reason, per_unit, total])
                ridx = ws.max_row
                # sub-rows stay unfilled (white) per the reference screenshot
                _style_row(ws, ridx, len(PM_HEADERS), None, False, PM_WRAP,
                           link_col=9, url=c.url)
                ws.cell(row=ridx, column=3).font = OPTION_FONT
                _money(ws, ridx, [6, 12, 13])
            ws.row_dimensions[ridx].height = 56
        band += 1
    wb.save(out)
    n_rows = sum(len(o) for _, _, o in groups)
    log.info("Price match report: %d item(s), %d option row(s) → %s",
             len(groups), n_rows, out.name)
    return out


# ---------------------------------------------------- alternate report ------

ALT_HEADERS = ["Original Schein Product and\nSchein Price",
               "Recommended Equivalent Product", "Recommended Supplier",
               "Price of the\nEquivalent", "Product URL", "Match Score",
               "Equivalency Basis and Confidence Level",
               "Estimated Savings vs.\nSchein Price"]
ALT_WIDTHS = [34, 36, 19, 12, 44, 16, 52, 20]
ALT_WRAP = {1, 2, 7}


def _alt_row(ws, band, item, equiv_name, supplier, price, url, score_label,
             basis, savings_cell, pct):
    ws.append([
        f"{item.description}\nSKU: {item.schein_sku}  ·  Schein price: ${item.unit_price:,.2f}/unit",
        equiv_name, supplier or "—", price, url, score_label, basis, savings_cell,
    ])
    ridx = ws.max_row
    _style_row(ws, ridx, len(ALT_HEADERS), pct, band % 2 == 0, ALT_WRAP,
               link_col=5, url=url)
    _money(ws, ridx, [4, 8])
    ws.row_dimensions[ridx].height = 64
    return ridx


def write_alternate_purchase_list(order: ParsedOrder,
                                  findings: List[EquivalencyFinding], out: Path,
                                  results: Optional[List[ItemResult]] = None) -> Path:
    """Stage 2 output — equivalency-table findings PLUS every reviewable
    scraped candidate for items that did not make the Price Match report.
    Items here never appear in the Price Match report (stream separation)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Alternate Purchases"
    title = (f"Alternate Purchase Recommendations  ·  Ref: {order.reference or '—'}"
             f"  ·  Generated: {_now()}"
             f"  ·  Items here do NOT appear in the Price Match report")
    _title_block(ws, title, ALT_LEGEND, len(ALT_HEADERS), ALT_HEADERS, ALT_WIDTHS)

    entries = []   # (tier_rank, item_order_idx, price, row-args)
    order_idx = {i.schein_sku: n for n, i in enumerate(order.items)}

    # A — equivalency-table findings (confirmed substitutions)
    equivalency_skus = set()
    for f in findings:
        if f.confidence_level not in ("exact_equivalent", "close_equivalent"):
            continue
        equivalency_skus.add(f.item.schein_sku)
        label = "EXACT" if f.confidence_level == "exact_equivalent" else "CLOSE"
        sentence = _TIER["exact"][1] if label == "EXACT" else _TIER["approximate"][1]
        basis = sentence + "\nEquivalency table (client-maintained) · " + (f.basis or "")
        per_unit = round(f.item.unit_price - f.price, 2) if f.price else None
        pct = _savings_pct(per_unit, f.item.unit_price) if per_unit is not None else None
        savings = (per_unit if per_unit and per_unit > 0
                   else (f"Equiv ${f.price:,.2f} > Schein ${f.item.unit_price:,.2f}"
                         if f.price else "—"))
        entries.append((-1, order_idx.get(f.item.schein_sku, 999), f.price or 0,
                        (f.item, f.equivalent_name, f.supplier, f.price,
                         f.url or search_fallback_url(f.item), f"{label} (table)",
                         basis, savings, pct)))

    # B — all reviewable scraped candidates for items not in Price Match
    for r in (results or []):
        if _appears_in_price_match(r) or r.item.schein_sku in equivalency_skus:
            continue
        pool, seen = [], set()
        for c in r.candidates:
            if _reviewable(c) and c.url not in seen:
                seen.add(c.url)
                pool.append(c)
        if not pool:
            url = search_fallback_url(r.item)
            entries.append((9, order_idx.get(r.item.schein_sku, 999), 0,
                            (r.item, "— no public candidate found this run —", "—",
                             None, url, "POSSIBLE (0%)",
                             "POSSIBLE — no usable public listing found; link opens a supplier search",
                             "—", None)))
            continue
        for c in pool:
            label, _, rank = _TIER.get(c.match_type, _TIER["unverified"])
            per_unit = round(r.item.unit_price - c.price, 2)
            pct = _savings_pct(per_unit, r.item.unit_price)
            savings = (per_unit if per_unit > 0 else
                       f"Equiv ${c.price:,.2f} > Schein ${r.item.unit_price:,.2f}")
            entries.append((rank, order_idx.get(r.item.schein_sku, 999), c.price,
                            (r.item, c.scraped_product_name or c.title,
                             c.source_site, c.price, c.url or search_fallback_url(r.item),
                             f"{label} ({match_score(c.criteria, c.confidence)}%)",
                             _basis_text(c), savings, pct)))

    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    for band, (_, _, _, args) in enumerate(entries):
        _alt_row(ws, band, *args)
    wb.save(out)
    log.info("Alternate purchases report: %d row(s) → %s", len(entries), out.name)
    return out


# ----------------------------------------------------- evidence report ------

def write_evidence_file(order: ParsedOrder, results: List[ItemResult],
                        findings: List[EquivalencyFinding], out: Path) -> Path:
    """Everything, per PRD 4.8 / 5.6 — every URL, every condition, every
    rejection, every confidence note. Nothing silently discarded."""
    wb = Workbook()

    ws = wb.active
    ws.title = "All Findings"
    title = (f"Background Evidence  ·  Ref: {order.reference or '—'}"
             f"  ·  Generated: {_now()}")
    headers = ["Schein SKU", "Description", "Candidate Title", "Source Site",
               "Product URL", "Price", "Pack\nQty", "Pack/Qty Condition",
               "Match Type", "Match\nScore", "Confidence", "Brand✓", "Name✓",
               "Size✓", "Pack✓", "Notes / Rejection Reason"]
    widths = [11, 32, 36, 16, 44, 11, 7, 18, 12, 9, 10, 6, 6, 6, 6, 42]
    _title_block(ws, title, "Every price found, every condition, every rejection — nothing discarded",
                 len(headers), headers, widths)
    band = 0
    for r in results:
        if not r.candidates:
            ws.append([r.item.schein_sku, r.item.description,
                       "— no public candidates found —", "—", "—",
                       None, None, "—", "none", 0, 0, "", "", "", "", "—"])
            _style_row(ws, ws.max_row, len(headers), None, band % 2 == 0, {2, 3, 16})
            band += 1
        for c in r.candidates:
            ws.append([
                r.item.schein_sku, r.item.description, c.title, c.source_site,
                c.url, c.price, c.pack_qty, _dash(_clean(c.pack_condition)),
                c.match_type, match_score(c.criteria, c.confidence), c.confidence,
                *("Y" if c.criteria.get(k) else ("N" if k in c.criteria else "")
                  for k in ("brand_match", "name_match", "size_form_match", "pack_match")),
                c.rejected_reason or _clean(c.notes) or "—",
            ])
            _style_row(ws, ws.max_row, len(headers), None, band % 2 == 0,
                       {2, 3, 8, 16}, link_col=5, url=c.url)
            _money(ws, ws.max_row, [6])
            band += 1

    ws2 = wb.create_sheet("Equivalency Findings")
    h2 = ["Schein SKU", "Original Product", "Equivalent", "Confidence",
          "Basis", "Supplier", "Product URL", "Price", "Routed To"]
    _title_block(ws2, "Equivalency Findings (all levels)", "Table-driven — PRD 5.6",
                 len(h2), h2, [11, 36, 32, 18, 42, 18, 46, 11, 22])
    for band, f in enumerate(findings):
        routed = ("Alternate Purchase List"
                  if f.confidence_level in ("exact_equivalent", "close_equivalent")
                  else "Evidence only — manual review")
        ws2.append([f.item.schein_sku, f.item.description, f.equivalent_name,
                    f.confidence_level.replace("_", " "), f.basis,
                    f.supplier or "—", f.url or "—", f.price, routed])
        _style_row(ws2, ws2.max_row, len(h2), None, band % 2 == 0, {2, 3, 5},
                   link_col=7 if f.url else None, url=f.url or "")
        _money(ws2, ws2.max_row, [8])

    ws3 = wb.create_sheet("Flagged Sites")
    h3 = ["Schein SKU", "Site / URL", "Reason"]
    _title_block(ws3, "Flagged Sites", "Login-required / no public price — skipped, never accessed",
                 len(h3), h3, [11, 50, 38])
    band = 0
    for r in results:
        for s in r.flagged_sites:
            ws3.append([r.item.schein_sku, s, "login required / no public price"])
            _style_row(ws3, ws3.max_row, len(h3), None, band % 2 == 0, {2, 3})
            band += 1
        for c in r.candidates:
            if c.rejected_reason and "login" in c.rejected_reason:
                ws3.append([r.item.schein_sku, c.url, c.rejected_reason])
                _style_row(ws3, ws3.max_row, len(h3), None, band % 2 == 0, {2, 3},
                           link_col=2, url=c.url)
                band += 1

    ws4 = wb.create_sheet("Parsed Order Audit")
    h4 = ["Qty", "Schein SKU", "Description", "UOM", "Unit Price",
          "Extended Price", "Pack Qty", "MPN", "Brand", "Variant"]
    _title_block(ws4, f"Parsed Order Audit  ·  {order.source_file}",
                 "Every extracted line item — totals must reconcile with the PDF",
                 len(h4), h4, [6, 11, 44, 6, 12, 13, 8, 16, 14, 14])
    for band, r in enumerate(results):
        i = r.item
        ws4.append([i.qty, i.schein_sku, i.description, i.uom, i.unit_price,
                    i.extended_price, _dash(i.pack_qty), _dash(i.mpn),
                    _dash(i.brand), _dash(i.variant)])
        _style_row(ws4, ws4.max_row, len(h4), None, band % 2 == 0, {3})
        _money(ws4, ws4.max_row, [5, 6])
    ws4.append([])
    ws4.append(["", "", "Computed total", "", "", order.computed_total])
    ws4.append(["", "", "Printed PDF total", "", "", order.total_price])
    for rr in (ws4.max_row - 1, ws4.max_row):
        ws4.cell(row=rr, column=3).font = BOLD
        ws4.cell(row=rr, column=6).number_format = MONEY
        ws4.cell(row=rr, column=6).font = BOLD

    wb.save(out)
    log.info("Evidence file written → %s", out.name)
    return out