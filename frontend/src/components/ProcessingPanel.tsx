import { formatDuration, formatTime } from "../api";
import { PIPELINE_STEPS, type ActivityEntry, type ServiceState, type StepStatus } from "../types";

interface ProcessingPanelProps {
  processing: boolean;
  failed: boolean;
  done: boolean;
  stepStatus: Record<string, StepStatus>;
  progressPercent: number;
  elapsedMs: number;
  activities: ActivityEntry[];
  services: ServiceState;
  itemProgress: { current: number; total: number; sku: string };
  reference: string | null;
}

const SERVICE_LABELS: Record<keyof ServiceState, string> = {
  groq: "Groq / AI",
  firecrawl: "Firecrawl",
  serpapi: "SerpAPI",
};

function StepIcon({ status }: { status: StepStatus }) {
  if (status === "complete") return <span className="step-dot step-dot--done">✓</span>;
  if (status === "active") return <span className="step-dot step-dot--active" />;
  if (status === "error") return <span className="step-dot step-dot--error">!</span>;
  if (status === "warning") return <span className="step-dot step-dot--warn">!</span>;
  return <span className="step-dot step-dot--pending" />;
}

function ServiceChip({ name, state }: { name: keyof ServiceState; state: ServiceState[keyof ServiceState] }) {
  return (
    <div className={`service-chip service-chip--${state}`}>
      <span className="service-chip__dot" />
      <span className="service-chip__name">{SERVICE_LABELS[name]}</span>
      <span className="service-chip__state">{state}</span>
    </div>
  );
}

function serviceTag(s: ActivityEntry["service"]) {
  const map: Record<string, string> = {
    groq: "AI",
    firecrawl: "Firecrawl",
    serpapi: "SerpAPI",
    parse: "Parse",
    mpn: "MPN",
    system: "System",
  };
  return map[s] ?? s;
}

export function ProcessingPanel({
  processing,
  failed,
  done,
  stepStatus,
  progressPercent,
  elapsedMs,
  activities,
  services,
  itemProgress,
  reference,
}: ProcessingPanelProps) {
  return (
    <section className="processing-panel">
      <div className="processing-panel__header">
        <div>
          <h2>
            {failed ? "Analysis failed" : done ? "Analysis complete" : "Live processing"}
          </h2>
          <p className="processing-panel__sub">
            {reference ? `Order ${reference}` : "Real-time updates from the backend pipeline"}
          </p>
        </div>
        <div className="timer-box">
          <span className="timer-box__label">Elapsed</span>
          <span className="timer-box__value">{formatDuration(elapsedMs)}</span>
        </div>
      </div>

      <div className="service-bar">
        <ServiceChip name="groq" state={services.groq} />
        <ServiceChip name="serpapi" state={services.serpapi} />
        <ServiceChip name="firecrawl" state={services.firecrawl} />
      </div>

      <div className="progress-track" role="progressbar" aria-valuenow={progressPercent}>
        <div className="progress-track__fill" style={{ width: `${progressPercent}%` }} />
      </div>

      {itemProgress.total > 0 && processing && (
        <div className="item-progress">
          <span>
            Item {itemProgress.current} of {itemProgress.total}
          </span>
          {itemProgress.sku && <code>{itemProgress.sku}</code>}
        </div>
      )}

      <div className="processing-grid">
        <ol className="step-list">
          {PIPELINE_STEPS.map((step) => {
            const status = stepStatus[step.id] ?? "pending";
            return (
              <li key={step.id} className={`step-list__item step-list__item--${status}`}>
                <StepIcon status={status} />
                <div>
                  <div className="step-list__title">
                    <span className="step-list__icon">{step.icon}</span>
                    {step.label}
                  </div>
                  <div className="step-list__desc">{step.description}</div>
                </div>
              </li>
            );
          })}
        </ol>

        <div className="activity-feed">
          <div className="activity-feed__header">
            <h3>Activity log</h3>
            <span>{activities.length} events</span>
          </div>
          <ul className="activity-feed__list">
            {activities.length === 0 ? (
              <li className="activity-feed__empty">Waiting for backend events…</li>
            ) : (
              activities.map((a) => (
                <li key={a.id} className={`activity-row activity-row--${a.service}`}>
                  <span className="activity-row__time">{formatTime(a.ts)}</span>
                  <span className={`activity-row__tag activity-row__tag--${a.service}`}>
                    {serviceTag(a.service)}
                  </span>
                  <div className="activity-row__body">
                    <span className="activity-row__msg">{a.message}</span>
                    {a.detail && <span className="activity-row__detail">{a.detail}</span>}
                  </div>
                </li>
              ))
            )}
          </ul>
        </div>
      </div>
    </section>
  );
}
