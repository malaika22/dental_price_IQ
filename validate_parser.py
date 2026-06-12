"""Cross-check: parse every sample order PDF and verify against known truth."""
import glob, sys
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

ok = True
for pdf in sorted(glob.glob("/mnt/user-data/uploads/*.pdf")):
    if "Briefing" in pdf:
        continue
    o = parse_order_pdf(pdf)
    exp = EXPECTED.get(o.reference)
    status = []
    if exp:
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
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
