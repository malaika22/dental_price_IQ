import { basename, formatDateTime, formatDuration, formatMoney, reportDownloadUrl } from "../api";
import { REPORT_META, type OrderHistoryEntry } from "../types";

interface HistoryPanelProps {
  history: OrderHistoryEntry[];
}

const STATUS_CONFIG = {
  completed: { label: "Completed", icon: "✓" },
  processing: { label: "Processing", icon: "◌" },
  failed: { label: "Failed", icon: "!" },
} as const;

function StatusBadge({ status }: { status: OrderHistoryEntry["status"] }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span className={`hist-badge hist-badge--${status}`}>
      <span className="hist-badge__icon" aria-hidden>{cfg.icon}</span>
      {cfg.label}
    </span>
  );
}

function HistoryCard({ entry }: { entry: OrderHistoryEntry }) {
  return (
    <li className={`hist-card hist-card--${entry.status}`}>
      <div className="hist-card__accent" aria-hidden />

      <div className="hist-card__body">
        <header className="hist-card__header">
          <div className="hist-card__icon-wrap" aria-hidden>
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M14 2H8a2 2 0 00-2 2v16a2 2 0 002 2h8a2 2 0 002-2V8l-6-6z"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinejoin="round"
              />
              <path d="M14 2v6h6M12 18v-4M10 16h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </div>

          <div className="hist-card__title-block">
            <h3 className="hist-card__ref" title={entry.reference || entry.fileName}>
              {entry.reference || entry.fileName}
            </h3>
            <p className="hist-card__meta">
              <span className="hist-card__file">{entry.fileName}</span>
              <span className="hist-card__dot">·</span>
              <time dateTime={entry.createdAt}>{formatDateTime(entry.createdAt)}</time>
              {entry.durationMs != null && (
                <>
                  <span className="hist-card__dot">·</span>
                  <span>{formatDuration(entry.durationMs)}</span>
                </>
              )}
            </p>
          </div>

          <StatusBadge status={entry.status} />
        </header>

        <div className="hist-card__metrics">
          <div className="hist-metric">
            <span className="hist-metric__value">{entry.items}</span>
            <span className="hist-metric__label">Items</span>
          </div>
          <div className="hist-metric hist-metric--total">
            <span className="hist-metric__value">{formatMoney(entry.total)}</span>
            <span className="hist-metric__label">Order total</span>
          </div>
        </div>

        {entry.error && (
          <p className="hist-card__error" role="alert">
            {entry.error}
          </p>
        )}

        {entry.status === "processing" && (
          <div className="hist-card__processing">
            <span className="spinner spinner--sm" aria-hidden />
            <span>Analysis in progress…</span>
          </div>
        )}

        {entry.status === "completed" && entry.reports && (
          <div className="hist-card__reports">
            <span className="hist-card__reports-label">Reports</span>
            <div className="hist-reports">
              {REPORT_META.map(({ key, title, desc, icon }) => {
                const path = entry.reports![key];
                const name = basename(path);
                if (!name) return null;
                return (
                  <a
                    key={key}
                    className="hist-report"
                    href={reportDownloadUrl(name)}
                    download={name}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span className="hist-report__icon" aria-hidden>{icon}</span>
                    <span className="hist-report__text">
                      <strong>{title}</strong>
                      <small>{desc}</small>
                    </span>
                    <span className="hist-report__arrow" aria-hidden>↓</span>
                  </a>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </li>
  );
}

export function HistoryPanel({ history }: HistoryPanelProps) {
  if (history.length === 0) {
    return (
      <section className="history history--empty">
        <div className="empty-state">
          <div className="empty-state__icon" aria-hidden>
            <svg viewBox="0 0 48 48" fill="none">
              <rect x="8" y="10" width="32" height="28" rx="3" stroke="currentColor" strokeWidth="2" />
              <path d="M16 18h16M16 26h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>
          <h2>No past orders yet</h2>
          <p>Upload and analyze an order PDF — every run is saved on the server permanently.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="history">
      <header className="history__toolbar">
        <div>
          <h2>Past Orders</h2>
          <p>
            {history.length} order{history.length !== 1 ? "s" : ""} on server — safe even if browser data is cleared
          </p>
        </div>
      </header>

      <ul className="history-list">
        {history.map((entry) => (
          <HistoryCard key={entry.id} entry={entry} />
        ))}
      </ul>
    </section>
  );
}
