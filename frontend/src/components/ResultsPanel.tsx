import { basename, formatMoney, reportDownloadUrl } from "../api";
import { REPORT_META, type OrderRunResult } from "../types";

interface ResultsPanelProps {
  result: OrderRunResult;
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const total = result.total ?? result.computed_total;

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
          <p>Your reports are ready to download.</p>
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
