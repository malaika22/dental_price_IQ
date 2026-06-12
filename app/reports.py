"""Report generation — the three output files (PRD 4.7, 4.8, 5.4).

Product URLs are included in ALL three files as requested.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .models import EquivalencyFinding, ItemResult, ParsedOrder

log = logging.getLogger(__name__)

HEADER_FILL = PatternFill("solid", fgColor="2F4858")
HEADER_FONT = Font(color="FFFFFF", bold=True)
MONEY = '"$"#,##0.00'


def _sheet(ws, headers: List[str], widths: List[int]):
    ws.append(headers)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
        cell = ws.cell(row=1, column=i)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"


def _money_cols(ws, cols: List[int]):
    for row in ws.iter_rows(min_row=2):
        for c in cols:
            row[c - 1].number_format = MONEY


def write_price_match_report(order: ParsedOrder, results: List[ItemResult], out: Path) -> Path:
    """Primary negotiation report — exact matches only, sorted by total savings
    descending. Exactly the client-defined columns (PRD 4.7) + Product URL."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Price Match"
    _sheet(ws, ["Schein SKU", "Manufacturer Part Number", "Description", "Qty Ordered",
                "Schein Unit Price", "Best Public Price", "Source Site", "Product URL",
                "Qty/Pack Condition", "Savings Per Unit", "Total Savings"],
           [12, 22, 46, 11, 16, 16, 18, 50, 24, 15, 15])

    rows = []
    for r in results:
        if r.routed_to_alternate or not r.best_exact:
            continue
        b = r.best_exact
        if b.price is None or b.price >= r.item.unit_price:
            continue                       # only genuine savings go to the rep
        per_unit = round(r.item.unit_price - b.price, 2)
        rows.append([
            r.item.schein_sku, r.item.mpn or "", r.item.description, r.item.qty,
            r.item.unit_price, b.price, b.source_site, b.url,
            b.pack_condition or "", per_unit, round(per_unit * r.item.qty, 2),
        ])
    rows.sort(key=lambda x: x[-1], reverse=True)
    for row in rows:
        ws.append(row)
    _money_cols(ws, [5, 6, 10, 11])
    wb.save(out)
    log.info("Price match report: %d row(s) with savings → %s", len(rows), out.name)
    return out


def write_alternate_purchase_list(order: ParsedOrder,
                                  findings: List[EquivalencyFinding], out: Path) -> Path:
    """Stage 2 output (PRD 5.4) — exact/close equivalents only."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Alternate Purchases"
    _sheet(ws, ["Schein SKU", "Original Product (Schein)", "Schein Unit Price",
                "Recommended Equivalent", "Confidence", "Equivalency Basis",
                "Recommended Supplier", "Product URL", "Equivalent Price",
                "Pack/Qty Condition", "Est. Total Savings"],
           [12, 42, 15, 36, 18, 44, 20, 50, 15, 22, 16])
    for f in findings:
        if f.confidence_level not in ("exact_equivalent", "close_equivalent"):
            continue
        ws.append([
            f.item.schein_sku, f.item.description, f.item.unit_price,
            f.equivalent_name, f.confidence_level.replace("_", " "), f.basis,
            f.supplier or "", f.url or "", f.price,
            f.pack_condition or "", f.est_savings_total,
        ])
    _money_cols(ws, [3, 9, 11])
    wb.save(out)
    log.info("Alternate purchases report: %d row(s) → %s", ws.max_row - 1, out.name)
    return out


def write_evidence_file(order: ParsedOrder, results: List[ItemResult],
                        findings: List[EquivalencyFinding], out: Path) -> Path:
    """Everything, per PRD 4.8 / 5.6 — every URL, every condition, every
    rejection, every confidence note. Nothing silently discarded."""
    wb = Workbook()

    ws = wb.active
    ws.title = "All Findings"
    _sheet(ws, ["Schein SKU", "Description", "Candidate Title", "Source Site",
                "Product URL", "Price", "Pack Qty", "Pack/Qty Condition",
                "Match Type", "Confidence", "Brand✓", "Name✓", "Size✓", "Pack✓",
                "Notes / Rejection Reason"],
           [12, 38, 42, 18, 52, 12, 9, 22, 13, 11, 7, 7, 7, 7, 48])
    for r in results:
        if not r.candidates:
            ws.append([r.item.schein_sku, r.item.description, "— no public candidates found —",
                       "", "", None, None, "", "none", 0, "", "", "", "", ""])
        for c in r.candidates:
            ws.append([
                r.item.schein_sku, r.item.description, c.title, c.source_site,
                c.url, c.price, c.pack_qty, c.pack_condition or "",
                c.match_type, c.confidence,
                *("Y" if c.criteria.get(k) else ("N" if k in c.criteria else "")
                  for k in ("brand_match", "name_match", "size_form_match", "pack_match")),
                c.rejected_reason or c.notes or "",
            ])
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
