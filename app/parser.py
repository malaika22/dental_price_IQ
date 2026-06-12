"""Henry Schein order PDF intake layer.

Fixes Section 9 Issues 1, 2 and 7 by design:
- Only lines matching the strict line-item grammar
  (qty, 6-8 digit SKU, description, UOM, $unit, $extended) are accepted.
  Addresses, phone numbers, dates, ZIP codes, totals and footers can never
  satisfy this pattern, so they cannot become products or prices.
- Column alignment is enforced by the regex itself: unit price is always the
  first $-amount and is cross-validated against qty * unit ~= extended.

Format 1 (structured PDF) is parsed directly with PyMuPDF.
Format 2 (scanned) falls back to Tesseract OCR if no line items are found
and the page contains images.
"""
from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from .models import OrderLineItem, ParsedOrder

log = logging.getLogger(__name__)

# qty | 6-8 digit Schein SKU | description | UOM (2-3 caps) | $unit | $extended
LINE_ITEM_RE = re.compile(
    r"^\s*(\d{1,5})\s+"            # qty
    r"(\d{6,8})\s+"                # Schein product number
    r"(.+?)\s+"                    # description (non-greedy)
    r"([A-Z]{2,3})\s+"             # UOM — must be ALL CAPS token right before price
    r"\$\s?([\d,]+\.\d{2})\s+"     # unit price
    r"\$\s?([\d,]+\.\d{2})\s*$"    # extended price
)
REFERENCE_RE = re.compile(r"\b(OR\d{16,22})\b")
TOTAL_RE = re.compile(r"Total(?:\s*Price)?\s*:?\s*\$\s?([\d,]+\.\d{2})", re.I)
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")

# Pack size from description: "100/Bx", "59ml/Bt", "4/Pk", "12 PK", "2x445 Gr"
PACK_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*/\s*(Bx|Pk|Bg|Cn|Ca|Bt|Jr|Ea|Kt|Rl|Sl)\b", re.I
)
PACK_ALT_RE = re.compile(r"\b(\d{1,4})\s*(?:PK|PACK|CT)\b", re.I)
PACK_MULT_RE = re.compile(r"\b(\d{1,2})\s*[xX]\s*(\d+)\b")


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_pack(description: str) -> tuple[Optional[int], Optional[str]]:
    """Heuristic pack quantity from the printed description."""
    m = PACK_RE.search(description)
    if m:
        val = float(m.group(1))
        # "59ml/Bt" -> volume, not a count; treat count only if integer-like and unit is countable
        unit = m.group(2).capitalize()
        if unit in ("Bt", "Jr") and re.search(r"(ml|oz|cc|l)\s*/", description, re.I):
            return None, unit          # volume container, count = 1 effectively
        return int(val), unit
    m = PACK_ALT_RE.search(description)
    if m:
        return int(m.group(1)), "Pk"
    m = PACK_MULT_RE.search(description)
    if m:
        return int(m.group(1)) * 1, None   # e.g. "2x445 Gr" -> 2 units of 445g
    return None, None


def _extract_text(doc: fitz.Document) -> str:
    return "\n".join(page.get_text("text") for page in doc)


def _ocr_fallback(doc: fitz.Document) -> str:
    """Format 2: rasterize and OCR. Requires tesseract + pytesseract installed."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        log.warning("OCR fallback requested but pytesseract/Pillow not installed.")
        return ""
    chunks = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        chunks.append(pytesseract.image_to_string(img))
    return "\n".join(chunks)


def parse_order_pdf(pdf_path: str | Path) -> ParsedOrder:
    pdf_path = Path(pdf_path)
    log.info("Opening PDF: %s", pdf_path.name)
    doc = fitz.open(pdf_path)
    text = _extract_text(doc)

    items = _parse_lines(text)
    if not items:  # possible scanned PDF (Format 2)
        log.info("No structured line items found in %s — trying OCR fallback", pdf_path.name)
        text = _ocr_fallback(doc)
        if not text:
            log.error("OCR fallback returned no text for %s", pdf_path.name)
        items = _parse_lines(text)

    ref = REFERENCE_RE.search(text)
    total = TOTAL_RE.search(text)
    date = DATE_RE.search(text)

    order = ParsedOrder(
        source_file=pdf_path.name,
        reference=ref.group(1) if ref else None,
        order_date=date.group(1) if date else None,
        total_price=_to_float(total.group(1)) if total else None,
        items=items,
    )
    log.info(
        "Parsed %d line item(s) — ref=%s date=%s",
        len(items), order.reference, order.order_date,
    )
    _validate(order)
    return order


_INT_RE = re.compile(r"^\d{1,5}$")
_SKU_RE = re.compile(r"^\d{6,8}$")
_UOM_RE = re.compile(r"^[A-Z]{2,3}$")
_PRICE_TOKEN_RE = re.compile(r"^\$\s?([\d,]+\.\d{2})$")


def _emit(items, seen, qty, sku, desc, uom, unit, ext) -> bool:
    """Validate + append one reconstructed row. Returns True if accepted."""
    # Column-alignment guard (Issue 7): unit * qty must reconcile with extended.
    # This also kills phantom rows built from stray numbers.
    if abs(qty * unit - ext) > 0.05:
        log.warning("Row failed price reconciliation, skipped: %s %s %s", sku, unit, ext)
        return False
    key = (sku, qty, round(unit, 2), round(ext, 2))
    if key in seen:              # duplicate text layer / page repeat guard
        return True
    seen.add(key)
    pack_qty, pack_unit = parse_pack(desc)
    items.append(OrderLineItem(
        qty=qty, schein_sku=sku, description=desc, uom=uom,
        unit_price=unit, extended_price=ext,
        pack_qty=pack_qty, pack_unit=pack_unit,
    ))
    return True


def _parse_lines(text: str) -> List[OrderLineItem]:
    """Reconstruct line items from either PDF text layout.

    Henry Schein order PDFs carry the table as ONE CELL PER LINE
    (Qty / SKU / Description / UOM / $unit / $extended on consecutive lines);
    some files additionally embed a single-line-per-row text layer. Both are
    handled; rows are deduplicated on (sku, qty, unit, extended).

    Only token sequences matching the full row grammar are accepted, so
    header blocks, addresses, phone numbers, ZIP codes, dates, totals and
    footers can never become products (Issues 1 + 2).
    """
    items: List[OrderLineItem] = []
    seen: set[tuple] = set()
    tokens = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Pass 1 — token state machine (cell-per-line layout)
    i, n = 0, len(tokens)
    while i < n - 5:
        if _INT_RE.match(tokens[i]) and _SKU_RE.match(tokens[i + 1]):
            j = i + 2
            desc_parts: List[str] = []
            # description may wrap across a few lines; stop at the UOM token
            while j < n and len(desc_parts) <= 4 and not _UOM_RE.match(tokens[j]):
                desc_parts.append(tokens[j])
                j += 1
            if (desc_parts and j + 2 < n and _UOM_RE.match(tokens[j])
                    and _PRICE_TOKEN_RE.match(tokens[j + 1])
                    and _PRICE_TOKEN_RE.match(tokens[j + 2])):
                unit = _to_float(_PRICE_TOKEN_RE.match(tokens[j + 1]).group(1))
                ext = _to_float(_PRICE_TOKEN_RE.match(tokens[j + 2]).group(1))
                if _emit(items, seen, int(tokens[i]), tokens[i + 1],
                         " ".join(desc_parts), tokens[j], unit, ext):
                    i = j + 3
                    continue
        i += 1

    # Pass 2 — single-line rows (alternate text layer / future format drift)
    for raw in text.splitlines():
        m = LINE_ITEM_RE.match(raw)
        if m:
            _emit(items, seen, int(m.group(1)), m.group(2), m.group(3).strip(),
                  m.group(4), _to_float(m.group(5)), _to_float(m.group(6)))
    return items


def _validate(order: ParsedOrder) -> None:
    if order.total_price is not None:
        diff = abs(order.computed_total - order.total_price)
        if diff > 0.01:
            log.warning(
                "%s: sum of line items (%.2f) != printed total (%.2f)",
                order.source_file, order.computed_total, order.total_price,
            )
        else:
            log.info("%s: %d items, total reconciles at $%.2f",
                     order.source_file, len(order.items), order.total_price)
