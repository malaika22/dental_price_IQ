import { basename, formatMoney, reportDownloadUrl } from "../api";
import { REPORT_META, type OrderRunResult } from "../types";

interface ResultsPanelProps {
  result: OrderRunResult;
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const total = result.total ?? result.computed_total;
  const stats = [
    { label: "Exact matches", value: result.exact_matches ?? "—" },
    { label: "Near matches", value: result.near_matches ?? "—" },
    { label: "Alternate candidates", value: result.alternate_candidates ?? "—" },
    { label: "No public price", value: result.no_public_price ?? "—" },
    { label: "Est. total savings", value: formatMoney(result.estimated_savings) },
    { label: "Items processed", value: result.items_processed ?? result.items },
  ];

  return (
    <section className="results">
      <div className="results__header">
        <div className="results__check" aria-hidden>
          <svg viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
            <path d="M8 12l3 3 5-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div>
          <h2>Analysis complete</h2>
          <p>Summary below — then download the Excel reports.</p>
        </div>
      </div>

      <div className="summary-grid">
        <div className="summary-card">
          <span className="summary-card__label">Reference</span>
          <span className="summary-card__value">{result.reference || "—"}</span>
        </div>
        <div className="summary-card">
          <span className="summary-card__label">Line items</span>
          <span className="summary-card__value">{result.items}</span>
        </div>
        <div className="summary-card">
          <span className="summary-card__label">Order total</span>
          <span className="summary-card__value">{formatMoney(total)}</span>
        </div>
      </div>

      <div className="results-stats">
        <h3 className="downloads__title">Match summary</h3>
        <div className="results-stats__grid">
          {stats.map((s) => (
            <div key={s.label} className="results-stats__card">
              <span className="results-stats__value">{s.value}</span>
              <span className="results-stats__label">{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="downloads">
        <h3 className="downloads__title">Download reports</h3>
        {REPORT_META.map(({ key, title, desc, icon }) => {
          const path = result[key];
          const name = basename(path);
          if (!name) return null;
          return (
            <a
              key={key}
              className="download-card"
              href={reportDownloadUrl(name)}
              download={name}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="download-card__icon" aria-hidden>
                {icon}
              </span>
              <span className="download-card__info">
                <strong>{title}</strong>
                <small>{desc}</small>
              </span>
              <span className="download-card__action">Download</span>
            </a>
          );
        })}
      </div>
    </section>
  );
}
