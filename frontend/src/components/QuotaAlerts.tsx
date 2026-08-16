import type { QuotaAlert } from "../types";

const SERVICE_LABELS: Record<string, string> = {
  groq: "Groq",
  gemini: "Gemini",
  firecrawl: "Firecrawl",
  serpapi: "SerpAPI",
  openai: "OpenAI",
  openrouter: "OpenRouter",
};

interface QuotaAlertsProps {
  alerts: QuotaAlert[];
}

export function QuotaAlerts({ alerts }: QuotaAlertsProps) {
  if (alerts.length === 0) return null;

  const hasGemini = alerts.some((a) => a.service === "gemini");

  return (
    <div className={`quota-alerts ${hasGemini ? "quota-alerts--gemini" : ""}`} role="alert">
      <div className="quota-alerts__header">
        <span className="quota-alerts__icon" aria-hidden>
          ⚠
        </span>
        <div>
          <strong>
            {hasGemini ? "Gemini tokens exhausted" : "API credit limits reached"}
          </strong>
          <p>
            {hasGemini
              ? "Please add more tokens to your Gemini account, then re-run analysis."
              : "Some services ran out of quota during this run. Reports may be less complete."}
          </p>
        </div>
      </div>
      <ul className="quota-alerts__list">
        {alerts.map((alert) => (
          <li
            key={alert.service}
            className={`quota-alerts__item ${alert.service === "gemini" ? "quota-alerts__item--gemini" : ""}`}
          >
            <span className="quota-alerts__service">
              {SERVICE_LABELS[alert.service] ?? alert.service}
            </span>
            <span className="quota-alerts__message">{alert.message}</span>
            {alert.detail && <span className="quota-alerts__detail">{alert.detail}</span>}
            {alert.service === "gemini" && (
              <a
                className="quota-alerts__action"
                href="https://aistudio.google.com/apikey"
                target="_blank"
                rel="noreferrer"
              >
                Open Google AI Studio →
              </a>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
