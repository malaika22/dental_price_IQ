import type { QuotaAlert } from "./types";

const SERVICE_LABELS: Record<string, string> = {
  groq: "Groq",
  gemini: "Gemini",
  firecrawl: "Firecrawl",
  serpapi: "SerpAPI",
  openai: "OpenAI",
  openrouter: "OpenRouter",
};

export function quotaAlertFromEvent(data: Record<string, unknown>): QuotaAlert {
  const service = String(data.service ?? "unknown");
  const kind = String(data.kind ?? "credits");
  let message = String(data.message ?? "API credit limit reached.");
  let detail = data.detail ? String(data.detail) : undefined;

  if (service === "gemini") {
    message =
      "Gemini tokens exhausted — please add more tokens to your Gemini account.";
    detail =
      detail ||
      "Open Google AI Studio / Cloud Console, top up Gemini API quota, then re-run the analysis.";
  }

  return { service, kind, message, detail };
}

export function quotaAlertFromFirecrawlSummary(data: Record<string, unknown>): QuotaAlert | null {
  if (!data.exhausted) return null;
  const reason = String(data.exhausted_reason ?? "");
  if (reason === "402") {
    return {
      service: "firecrawl",
      kind: "credits",
      message: "Your Firecrawl credit limit has been reached.",
      detail: "Web page scraping was paused during this run.",
    };
  }
  if (reason === "budget") {
    return {
      service: "firecrawl",
      kind: "budget",
      message: "The Firecrawl scrape limit for this run has been reached.",
      detail: "Remaining pages used cached or free-fetch data only.",
    };
  }
  return null;
}

export function quotaAlertsFromError(error: string): QuotaAlert[] {
  const lower = error.toLowerCase();
  const alerts: QuotaAlert[] = [];

  const checks: Array<{ match: RegExp; service: string; message: string; detail?: string }> = [
    { match: /firecrawl|402/, service: "firecrawl", message: "Your Firecrawl credit limit has been reached." },
    { match: /serpapi/, service: "serpapi", message: "Your SerpAPI credit limit has been reached." },
    {
      match: /gemini|generativelanguage|resource.?exhausted/,
      service: "gemini",
      message: "Gemini tokens exhausted — please add more tokens to your Gemini account.",
      detail: "Open Google AI Studio / Cloud Console, top up Gemini API quota, then re-run the analysis.",
    },
    { match: /groq|rate.?limit|quota/, service: "groq", message: "Your Groq API credit or rate limit has been reached." },
  ];

  for (const { match, service, message, detail } of checks) {
    if (match.test(lower) && !alerts.some((a) => a.service === service)) {
      alerts.push({
        service,
        kind: "credits",
        message,
        detail:
          detail ??
          `Check your ${SERVICE_LABELS[service] ?? service} plan or wait for the limit to reset.`,
      });
    }
  }

  return alerts;
}

export function mergeQuotaAlerts(prev: QuotaAlert[], next: QuotaAlert): QuotaAlert[] {
  if (prev.some((a) => a.service === next.service)) return prev;
  return [...prev, next];
}
