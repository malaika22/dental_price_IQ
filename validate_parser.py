"""Cross-check: parse every sample order PDF and verify against known truth.

Sample PDFs are located from (first match wins, all globs combined):
  1. any path(s) passed on the command line
  2. $SAMPLE_PDF_GLOB
  3. ./samples/*.pdf
  4. /mnt/user-data/uploads/*.pdf   (original sandbox location)
If none are found the script exits non-zero instead of trivially "passing".
"""
import glob, os, sys
sys.path.insert(0, ".")
from app.parser import parse_order_pdf

EXPECTED = {  # reference -> (item_count, printed_total)
    "OR202602190851373110": (11, 2849.58),
    "OR202603121104075460": (12, 3951.58),
    "OR202602251732249700": (16, 5528.06),
    "OR202605280844126040": (25, 10610.55),
}
# Issue 7 spot checks: SKU -> correct unit price (parser must NOT return pack qty)
SPOT = {"1077811": 22.65, "1071379": 29.60, "1126862": 34.56,
        "2222233": 133.53, "3120199": 518.89, "3333468": 372.66}

cli_globs = [a for a in sys.argv[1:] if not a.startswith("-")]
_globs = cli_globs or [g for g in (
    os.environ.get("SAMPLE_PDF_GLOB", ""),
    "samples/*.pdf",
    "/mnt/user-data/uploads/*.pdf",
) if g]
pdfs, _seen = [], set()
for _g in _globs:
    for p in sorted(glob.glob(_g)):
        if p not in _seen:
            _seen.add(p)
            pdfs.append(p)
if not pdfs:
    print("No sample PDFs found. Pass PDF path(s) as arguments, set "
          "SAMPLE_PDF_GLOB, or place files in ./samples/*.pdf")
    sys.exit(2)

ok = True
checked_expected = 0   # how many known-truth orders were actually validated
for pdf in pdfs:
    if "Briefing" in pdf:
        continue
    o = parse_order_pdf(pdf)
    exp = EXPECTED.get(o.reference)
    status = []
    if exp:
        checked_expected += 1
        c_ok = len(o.items) == exp[0]
        t_ok = abs(o.computed_total - exp[1]) < 0.01 and abs((o.total_price or 0) - exp[1]) < 0.01
        ok &= c_ok and t_ok
        status.append(f"items {len(o.items)}/{exp[0]} {'PASS' if c_ok else 'FAIL'}")
        status.append(f"total ${o.computed_total:,.2f} vs printed ${o.total_price:,.2f} {'PASS' if t_ok else 'FAIL'}")
    # no phantom rows: every SKU must be numeric 6-8 digits, no address words
    bad = [i for i in o.items if any(w in i.description for w in
           ("Duryea", "Melville", "Cameron", "PO Box", "Total", "henryschein"))]
    ok &= not bad
    status.append(f"phantom rows: {len(bad)} {'PASS' if not bad else 'FAIL'}")
    for it in o.items:
        if it.schein_sku in SPOT:
            good = abs(it.unit_price - SPOT[it.schein_sku]) < 0.001
            ok &= good
            status.append(f"  SKU {it.schein_sku} unit ${it.unit_price} (expect ${SPOT[it.schein_sku]}) {'PASS' if good else 'FAIL'}")
    print(f"\n=== {o.source_file}  ref={o.reference}  date={o.order_date}")
    print("\n".join("  " + s for s in status))
    for it in o.items:
        print(f"   {it.qty:>3} | {it.schein_sku} | {it.description[:46]:<46} | {it.uom} | ${it.unit_price:>8.2f} | ${it.extended_price:>9.2f} | pack={it.pack_qty}")
if checked_expected == 0:
    print("\nNO KNOWN-TRUTH ORDERS MATCHED — none of the parsed PDFs are in the "
          "EXPECTED table, so nothing was actually validated.")
    sys.exit(2)
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
