import { formatMoney } from "../api";
import type { ParsePreview } from "../types";

interface ParsePreviewPanelProps {
  preview: ParsePreview;
  onConfirm: () => void;
  onCancel: () => void;
  confirming: boolean;
}

export function ParsePreviewPanel({
  preview,
  onConfirm,
  onCancel,
  confirming,
}: ParsePreviewPanelProps) {
  return (
    <section className="parse-preview">
      <div className="parse-preview__head">
        <div>
          <span className="hero-card__eyebrow">Parsed preview</span>
          <h2>Review line items before price search</h2>
          <p>
            Confirm the parser looks correct — then start the full search. This avoids
            burning API credits on a bad parse.
          </p>
        </div>
        <div className="parse-preview__meta">
          <div>
            <span>Reference</span>
            <strong>{preview.reference || "—"}</strong>
          </div>
          <div>
            <span>Items</span>
            <strong>{preview.items.length}</strong>
          </div>
          <div>
            <span>Order total</span>
            <strong>{formatMoney(preview.total ?? preview.computed_total)}</strong>
          </div>
        </div>
      </div>

      <div className="parse-preview__table-wrap">
        <table className="parse-preview__table">
          <thead>
            <tr>
              <th>SKU</th>
              <th>Description</th>
              <th>Qty</th>
              <th>UOM</th>
              <th>Schein unit</th>
              <th>Extended</th>
            </tr>
          </thead>
          <tbody>
            {preview.items.map((row) => (
              <tr key={`${row.sku}-${row.description}`}>
                <td>
                  <code>{row.sku}</code>
                </td>
                <td>{row.description}</td>
                <td>{row.qty}</td>
                <td>{row.uom || "—"}</td>
                <td>{formatMoney(row.unit_price)}</td>
                <td>{formatMoney(row.extended_price)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="toolbar">
        <button
          type="button"
          className="btn btn--primary btn--lg"
          disabled={confirming}
          onClick={onConfirm}
        >
          {confirming ? (
            <>
              <span className="spinner" aria-hidden /> Starting full analysis…
            </>
          ) : (
            "Confirm & run price search"
          )}
        </button>
        <button type="button" className="btn btn--soft" disabled={confirming} onClick={onCancel}>
          Cancel
        </button>
      </div>
    </section>
  );
}
