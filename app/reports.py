"""Report generation — the three output files (PRD 4.7, 4.8, 5.4).

Product URLs are included in ALL three files and rendered as clickable
hyperlinks; when no verified product page exists, the link falls back to a
supplier search page (Google Shopping) for that item.

v2 formatting:
- Legend row in the primary + alternate reports:
  🟢 >10% savings  🟡 5–10% savings  EXACT/CLOSE = confirmed match ·
  APPROXIMATE = closest match for review
- Match Type and Match Score columns (score = 60% four-criteria coverage
  + 40% validation confidence, 0–100).
- Alternate purchase list now has TWO sections in one table:
  (a) equivalency-table findings (exact/close — PRD 5.4), and
  (b) every order item that did NOT earn a primary price-match row, shown
      with its closest reviewable candidate so nothing leaves the run
      without an actionable lead.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .models import EquivalencyFinding, ItemResult, ParsedOrder, PriceCandidate

log = logging.getLogger(__name__)

HEADER_FILL = PatternFill("solid", fgColor="2F4858")
HEADER_FONT = Font(color="FFFFFF", bold=True)
GREEN_FILL = PatternFill("solid", fgColor="C6EFCE")    # >10% savings
YELLOW_FILL = PatternFill("solid", fgColor="FFEB9C")   # 5–10% savings
LINK_FONT = Font(color="0563C1", underline="single")
LEGEND_FONT = Font(italic=True, size=9, color="555555")
MONEY = '"$"#,##0.00'

LEGEND = ("🟢 >10% savings   🟡 5–10% savings   "
          "EXACT/CLOSE = confirmed match · APPROXIMATE = closest match for review   "
          "Product URL links to verified product or supplier search page")


# ------------------------------------------------------------- helpers ------

def _clean(val) -> str:
    """Sanitize scraped condition/notes — 'None'/'null' strings become ''."""
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
    """Supplier search page when no verified product URL exists."""
    q = item.search_query or item.description
    return f"https://www.google.com/search?tbm=shop&q={quote_plus(q)}"


def _sheet(ws, headers: List[str], widths: List[int], legend: bool = False):
    """Header block. With legend=True: row 1 = legend, row 2 = headers."""
    if legend:
        ws.append([LEGEND])
        ws.merge_cells(start_row=1, start_column=1, end_row=1,
                       end_column=len(headers))
        ws.cell(row=1, column=1).font = LEGEND_FONT
        ws.cell(row=1, column=1).alignment = Alignment(vertical="center")
    ws.append(headers)
    hrow = ws.max_row
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
        cell = ws.cell(row=hrow, column=i)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    ws.freeze_panes = f"A{hrow + 1}"


def _money_cols(ws, cols: List[int], min_row: int = 2):
    for row in ws.iter_rows(min_row=min_row):
        for c in cols:
            row[c - 1].number_format = MONEY


def _link_cell(ws, row: int, col: int, url: str):
    if not url:
        return
    cell = ws.cell(row=row, column=col)
    cell.value = url
    cell.hyperlink = url
    cell.font = LINK_FONT


def _highlight(ws, row: int, ncols: int, savings_pct: Optional[float]):
    if savings_pct is None:
        return
    fill = GREEN_FILL if savings_pct > 10 else (YELLOW_FILL if savings_pct >= 5 else None)
    if fill:
        for c in range(1, ncols + 1):
            ws.cell(row=row, column=c).fill = fill


def _has_primary_row(r: ItemResult) -> bool:
    return (not r.routed_to_alternate and r.best_exact is not None
            and r.best_exact.price is not None
            and r.best_exact.price < r.item.unit_price)


def _reviewable(c: PriceCandidate) -> bool:
    if c.price is None:
        return False
    reason = (c.rejected_reason or "").lower()
    return "login" not in reason and "sanity" not in reason


_RANK = {"exact": 0, "approximate": 1, "rejected": 2, "unverified": 3}


def pick_closest_candidate(r: ItemResult) -> Optional[PriceCandidate]:
    """Best reviewable lead for an item that has no primary price-match row:
    exact > approximate > verified-but-rejected (e.g. brand/variant mismatch)
    > unverified; then highest match score, then lowest price."""
    pool = [c for c in r.candidates if _reviewable(c)]
    if not pool:
        return None
    best_rank = min(_RANK.get(c.match_type, 9) for c in pool)
    pool = [c for c in pool if _RANK.get(c.match_type, 9) == best_rank]
    # among the best rank, near-tied scores (within 15 pts of max) compete on
    # price — a 2-point score edge shouldn't hide a 2x cheaper lead
    smax = max(match_score(c.criteria, c.confidence) for c in pool)
    pool = [c for c in pool if match_score(c.criteria, c.confidence) >= smax - 15]
    pool.sort(key=lambda c: (c.price, -match_score(c.criteria, c.confidence)))
    return pool[0]


def _match_type_label(c: PriceCandidate) -> str:
    return {"exact": "EXACT", "approximate": "APPROXIMATE",
            "rejected": "REVIEW", "unverified": "UNVERIFIED"}.get(c.match_type, "REVIEW")


# ------------------------------------------------------ primary report ------

PM_HEADERS = ["Schein SKU", "Manufacturer Part Number", "Description", "Qty Ordered",
              "Schein Unit Price", "Best Public Price", "Source Site", "Product URL",
              "Qty/Pack Condition", "Match Type", "Match Score",
              "Savings Per Unit", "Total Savings", "Savings %"]
PM_WIDTHS = [12, 22, 44, 11, 16, 16, 20, 50, 22, 13, 12, 14, 13, 11]


def write_price_match_report(order: ParsedOrder, results: List[ItemResult], out: Path) -> Path:
    """Primary negotiation report — exact matches only, sorted by total savings
    descending. PRD 4.7 columns + URL, Match Type, Match Score, Savings %."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Price Match"
    _sheet(ws, PM_HEADERS, PM_WIDTHS, legend=True)

    rows = []
    for r in results:
        if not _has_primary_row(r):
            continue
        b = r.best_exact
        per_unit = round(r.item.unit_price - b.price, 2)
        pct = per_unit / r.item.unit_price * 100
        rows.append(([
            r.item.schein_sku, r.item.mpn or "", r.item.description, r.item.qty,
            r.item.unit_price, b.price, b.source_site, b.url,
            _clean(b.pack_condition), "EXACT", match_score(b.criteria, b.confidence),
            per_unit, round(per_unit * r.item.qty, 2), round(pct, 1),
        ], pct, b.url))
    rows.sort(key=lambda x: x[0][-2], reverse=True)

    for vals, pct, url in rows:
        ws.append(vals)
        ridx = ws.max_row
        _highlight(ws, ridx, len(PM_HEADERS), pct)
        _link_cell(ws, ridx, 8, url)
    _money_cols(ws, [5, 6, 12, 13], min_row=3)
    wb.save(out)
    log.info("Price match report: %d row(s) with savings → %s", len(rows), out.name)
    return out


# ---------------------------------------------------- alternate report ------

ALT_HEADERS = ["Schein SKU", "Original Product (Schein)", "Schein Unit Price",
               "Recommended Alternative", "Match Type", "Match Score",
               "Basis / Notes", "Supplier", "Product URL", "Alt Price",
               "Pack/Qty Condition", "Est. Savings Per Unit", "Est. Total Savings",
               "Savings %"]
ALT_WIDTHS = [12, 40, 14, 38, 18, 12, 44, 20, 50, 13, 22, 14, 14, 11]


def write_alternate_purchase_list(order: ParsedOrder,
                                  findings: List[EquivalencyFinding], out: Path,
                                  results: Optional[List[ItemResult]] = None) -> Path:
    """Stage 2 output (PRD 5.4) — equivalency exact/close rows — PLUS every
    order item that didn't make the primary price-match list, paired with its
    closest reviewable candidate (or a supplier search link if none)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Alternate Purchases"
    _sheet(ws, ALT_HEADERS, ALT_WIDTHS, legend=True)
    nrows = 0

    # Section A — equivalency-table findings (confirmed equivalents)
    equivalency_skus = set()
    for f in findings:
        if f.confidence_level not in ("exact_equivalent", "close_equivalent"):
            continue
        equivalency_skus.add(f.item.schein_sku)
        per_unit = (round(f.item.unit_price - f.price, 2) if f.price else None)
        pct = (per_unit / f.item.unit_price * 100) if per_unit is not None else None
        url = f.url or search_fallback_url(f.item)
        label = "EXACT (equivalent)" if f.confidence_level == "exact_equivalent" else "CLOSE (equivalent)"
        ws.append([f.item.schein_sku, f.item.description, f.item.unit_price,
                   f.equivalent_name, label, "", f.basis, f.supplier or "",
                   url, f.price, _clean(f.pack_condition),
                   per_unit, f.est_savings_total,
                   round(pct, 1) if pct is not None else ""])
        ridx = ws.max_row
        _highlight(ws, ridx, len(ALT_HEADERS), pct)
        _link_cell(ws, ridx, 9, url)
        nrows += 1

    # Section B — items with no primary price-match row → closest lead
    for r in (results or []):
        if _has_primary_row(r) or r.item.schein_sku in equivalency_skus:
            continue
        c = pick_closest_candidate(r)
        if c is None:
            url = search_fallback_url(r.item)
            ws.append([r.item.schein_sku, r.item.description, r.item.unit_price,
                       "— no public candidate found —", "REVIEW", 0,
                       "No usable public listing this run; link opens a supplier search",
                       "", url, None, "", None, None, ""])
            ridx = ws.max_row
            _link_cell(ws, ridx, 9, url)
            nrows += 1
            continue
        per_unit = round(r.item.unit_price - c.price, 2)
        pct = per_unit / r.item.unit_price * 100
        note = _clean(c.notes) or _clean(c.rejected_reason)
        if per_unit <= 0:
            note = (note + " · " if note else "") + "No cheaper public price found"
        url = c.url or search_fallback_url(r.item)
        ws.append([r.item.schein_sku, r.item.description, r.item.unit_price,
                   c.scraped_product_name or c.title, _match_type_label(c),
                   match_score(c.criteria, c.confidence), note,
                   c.source_site, url, c.price, _clean(c.pack_condition),
                   per_unit, round(per_unit * r.item.qty, 2), round(pct, 1)])
        ridx = ws.max_row
        _highlight(ws, ridx, len(ALT_HEADERS), pct)
        _link_cell(ws, ridx, 9, url)
        nrows += 1

    _money_cols(ws, [3, 10, 12, 13], min_row=3)
    wb.save(out)
    log.info("Alternate purchases report: %d row(s) → %s", nrows, out.name)
    return out


# ----------------------------------------------------- evidence report ------

def write_evidence_file(order: ParsedOrder, results: List[ItemResult],
                        findings: List[EquivalencyFinding], out: Path) -> Path:
    """Everything, per PRD 4.8 / 5.6 — every URL, every condition, every
    rejection, every confidence note. Nothing silently discarded."""
    wb = Workbook()

    ws = wb.active
    ws.title = "All Findings"
    _sheet(ws, ["Schein SKU", "Description", "Candidate Title", "Source Site",
                "Product URL", "Price", "Pack Qty", "Pack/Qty Condition",
                "Match Type", "Match Score", "Confidence", "Brand✓", "Name✓",
                "Size✓", "Pack✓", "Notes / Rejection Reason"],
           [12, 38, 42, 18, 52, 12, 9, 22, 13, 12, 11, 7, 7, 7, 7, 48])
    for r in results:
        if not r.candidates:
            ws.append([r.item.schein_sku, r.item.description, "— no public candidates found —",
                       "", "", None, None, "", "none", 0, 0, "", "", "", "", ""])
        for c in r.candidates:
            ws.append([
                r.item.schein_sku, r.item.description, c.title, c.source_site,
                c.url, c.price, c.pack_qty, _clean(c.pack_condition),
                c.match_type, match_score(c.criteria, c.confidence), c.confidence,
                *("Y" if c.criteria.get(k) else ("N" if k in c.criteria else "")
                  for k in ("brand_match", "name_match", "size_form_match", "pack_match")),
                c.rejected_reason or _clean(c.notes) or "",
            ])
            _link_cell(ws, ws.max_row, 5, c.url)
    _money_cols(ws, [6])

    ws2 = wb.create_sheet("Equivalency Findings")
    _sheet(ws2, ["Schein SKU", "Original Product", "Equivalent", "Confidence",
                 "Basis", "Supplier", "Product URL", "Price", "Routed To"],
           [12, 40, 36, 20, 46, 20, 52, 12, 24])
    for f in findings:
        routed = ("Alternate Purchase List"
                  if f.confidence_level in ("exact_equivalent", "close_equivalent")
                  else "Evidence only — manual review")
        ws2.append([f.item.schein_sku, f.item.description, f.equivalent_name,
                    f.confidence_level.replace("_", " "), f.basis,
                    f.supplier or "", f.url or "", f.price, routed])
        if f.url:
            _link_cell(ws2, ws2.max_row, 7, f.url)
    _money_cols(ws2, [8])

    ws3 = wb.create_sheet("Flagged Sites")
    _sheet(ws3, ["Schein SKU", "Site / URL", "Reason"], [12, 52, 40])
    for r in results:
        for s in r.flagged_sites:
            ws3.append([r.item.schein_sku, s, "login required / no public price"])
        for c in r.candidates:
            if c.rejected_reason and "login" in c.rejected_reason:
                ws3.append([r.item.schein_sku, c.url, c.rejected_reason])

    ws4 = wb.create_sheet("Parsed Order Audit")
    _sheet(ws4, ["Qty", "Schein SKU", "Description", "UOM", "Unit Price",
                 "Extended Price", "Pack Qty", "MPN", "Brand", "Variant"],
           [7, 12, 48, 7, 13, 14, 9, 18, 16, 16])
    for r in results:
        i = r.item
        ws4.append([i.qty, i.schein_sku, i.description, i.uom, i.unit_price,
                    i.extended_price, i.pack_qty, i.mpn or "", i.brand or "", i.variant or ""])
    ws4.append([])
    ws4.append(["", "", "Computed total", "", "", order.computed_total])
    ws4.append(["", "", "Printed PDF total", "", "", order.total_price])
    _money_cols(ws4, [5, 6])

    wb.save(out)
    candidate_rows = sum(len(r.candidates) for r in results) or len(results)
    log.info(
        "Evidence file: %d candidate row(s), %d equivalency finding(s) → %s",
        candidate_rows, len(findings), out.name,
    )
    return out