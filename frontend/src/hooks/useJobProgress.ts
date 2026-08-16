import { useCallback, useState } from "react";
import type {
  ActivityEntry,
  PipelineStepId,
  ProgressEvent,
  QuotaAlert,
  ServiceState,
  StepStatus,
} from "../types";
import { PIPELINE_STEPS } from "../types";
import {
  mergeQuotaAlerts,
  quotaAlertFromEvent,
  quotaAlertFromFirecrawlSummary,
  quotaAlertsFromError,
} from "../quotaAlerts";

const INITIAL_SERVICES: ServiceState = {
  groq: "idle",
  firecrawl: "idle",
  serpapi: "idle",
};

function eventToActivity(evt: ProgressEvent): ActivityEntry | null {
  const d = evt.data;
  const id = `${evt.ts}-${evt.event}-${Math.random().toString(36).slice(2, 7)}`;

  switch (evt.event) {
    case "upload_complete":
      return { id, ts: evt.ts, service: "system", message: "PDF uploaded", detail: String(d.filename ?? "") };
    case "step_start":
      return {
        id,
        ts: evt.ts,
        service: d.step === "parse" ? "parse" : d.step === "mpn" ? "mpn" : "system",
        message: String(d.label ?? d.message ?? `Starting ${d.step}`),
      };
    case "step_complete":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: String(d.message ?? `Completed ${d.step}`),
        detail: d.reference ? `Ref ${d.reference}` : undefined,
      };
    case "step_warning":
      return { id, ts: evt.ts, service: "system", message: String(d.message ?? "Warning") };
    case "groq": {
      const provider = String(d.provider ?? "AI").toUpperCase();
      const action = String(d.action ?? "");
      if (action === "start") {
        return {
          id,
          ts: evt.ts,
          service: "groq",
          message: String(d.message ?? `${provider} request started`),
          detail: d.sku ? `SKU ${d.sku}` : d.phase ? String(d.phase) : undefined,
        };
      }
      if (action === "chunk") {
        return {
          id,
          ts: evt.ts,
          service: "groq",
          message: `${provider} parsing chunk ${d.chunk}`,
          detail: `${d.items} items`,
        };
      }
      if (action === "complete") {
        return {
          id,
          ts: evt.ts,
          service: "groq",
          message: String(d.message ?? `${provider} finished`),
        };
      }
      return {
        id,
        ts: evt.ts,
        service: "groq",
        message: `${provider} ${action}`,
        detail: d.sku ? `SKU ${d.sku}` : undefined,
      };
    }
    case "serpapi":
      return {
        id,
        ts: evt.ts,
        service: "serpapi",
        message: "SerpAPI search",
        detail: String(d.query ?? "").slice(0, 80),
      };
    case "firecrawl":
      if (d.action === "scrape") {
        return {
          id,
          ts: evt.ts,
          service: "firecrawl",
          message: `Scraping ${d.site}`,
          detail: String(d.url ?? "").slice(0, 70),
        };
      }
      return {
        id,
        ts: evt.ts,
        service: "firecrawl",
        message: `Firecrawl search (${d.stage})`,
        detail: `${d.results ?? 0} results · ${String(d.query ?? "").slice(0, 50)}`,
      };
    case "firecrawl_summary":
      return {
        id,
        ts: evt.ts,
        service: "firecrawl",
        message: d.exhausted ? "Firecrawl credits exhausted" : "Firecrawl run complete",
        detail: `${d.scrapes} scrapes · ${d.credits} credits`,
      };
    case "quota_limit":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: String(d.message ?? "API credit limit reached"),
        detail: d.detail ? String(d.detail) : undefined,
      };
    case "discovery_start":
      return {
        id,
        ts: evt.ts,
        service: "serpapi",
        message: `Discovery for SKU ${d.sku}`,
        detail: String(d.message ?? ""),
      };
    case "discovery_complete":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: `SKU ${d.sku}: ${d.shortlisted} candidates shortlisted`,
      };
    case "item_start":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: `Item ${d.index}/${d.total} — SKU ${d.sku}`,
        detail: String(d.description ?? "").slice(0, 60),
      };
    case "item_complete":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: `SKU ${d.sku} done`,
        detail: d.best_exact ? String(d.best_exact) : `${d.candidates} candidates`,
      };
    case "item_timeout":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: `SKU ${d.sku} timed out (${d.seconds}s)`,
      };
    case "equivalency_found":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: `Equivalency: SKU ${d.sku}`,
        detail: String(d.level),
      };
    case "pipeline_complete":
      return {
        id,
        ts: evt.ts,
        service: "system",
        message: "Pipeline complete",
        detail: String(d.reference ?? ""),
      };
    case "pipeline_error":
      return { id, ts: evt.ts, service: "system", message: String(d.message ?? "Error") };
    default:
      return null;
  }
}

function updateServices(evt: ProgressEvent, prev: ServiceState): ServiceState {
  const next = { ...prev };
  if (evt.event === "groq") {
    const action = String(evt.data.action ?? "");
    if (action === "start" || action === "chunk") next.groq = "active";
    if (action === "complete" || action === "cache_only") next.groq = "done";
    if (action === "chunk_failed") next.groq = "error";
  }
  if (evt.event === "firecrawl") {
    next.firecrawl = "active";
    if (evt.data.action === "search" && evt.data.results === 0) next.firecrawl = "active";
  }
  if (evt.event === "firecrawl_summary") next.firecrawl = evt.data.exhausted ? "error" : "done";
  if (evt.event === "quota_limit") {
    const svc = String(evt.data.service ?? "");
    if (svc === "firecrawl") next.firecrawl = "error";
    if (svc === "serpapi") next.serpapi = "error";
    if (svc === "groq" || svc === "gemini" || svc === "openai" || svc === "openrouter") {
      next.groq = "error";
    }
  }
  if (evt.event === "serpapi" || evt.event === "discovery_start") next.serpapi = "active";
  if (evt.event === "pipeline_complete") {
    next.groq = next.groq === "active" ? "done" : next.groq;
    next.firecrawl = next.firecrawl === "active" ? "done" : next.firecrawl;
    next.serpapi = next.serpapi === "active" ? "done" : next.serpapi;
  }
  return next;
}

function applyStepEvent(
  evt: ProgressEvent,
  steps: Record<PipelineStepId, StepStatus>,
): Record<PipelineStepId, StepStatus> {
  const next = { ...steps };
  const step = evt.data.step as PipelineStepId | undefined;

  if (evt.event === "upload_complete") {
    next.upload = "complete";
    next.parse = "active";
  }
  if (evt.event === "step_start" && step) {
    const idx = PIPELINE_STEPS.findIndex((s) => s.id === step);
    for (let i = 0; i < idx; i++) {
      if (next[PIPELINE_STEPS[i].id] !== "complete") {
        next[PIPELINE_STEPS[i].id] = "complete";
      }
    }
    next[step] = "active";
  }
  if (evt.event === "step_complete" && step) {
    next[step] = "complete";
    const idx = PIPELINE_STEPS.findIndex((s) => s.id === step);
    if (idx >= 0 && idx < PIPELINE_STEPS.length - 1) {
      const nxt = PIPELINE_STEPS[idx + 1].id;
      if (next[nxt] === "pending") next[nxt] = "active";
    }
  }
  if (evt.event === "step_warning" && step) {
    next[step] = "warning";
  }
  if (evt.event === "pipeline_complete") {
    for (const s of PIPELINE_STEPS) next[s.id] = "complete";
  }
  if (evt.event === "pipeline_error") {
    const active = PIPELINE_STEPS.find((s) => next[s.id] === "active");
    if (active) next[active.id] = "error";
  }
  return next;
}

function initialSteps(): Record<PipelineStepId, StepStatus> {
  return {
    upload: "pending",
    parse: "pending",
    ai: "pending",
    mpn: "pending",
    stage1: "pending",
    stage2: "pending",
    reports: "pending",
  };
}

export function useJobProgress() {
  const [stepStatus, setStepStatus] = useState(initialSteps);
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [services, setServices] = useState<ServiceState>(INITIAL_SERVICES);
  const [itemProgress, setItemProgress] = useState({ current: 0, total: 0, sku: "" });
  const [reference, setReference] = useState<string | null>(null);
  const [quotaAlerts, setQuotaAlerts] = useState<QuotaAlert[]>([]);

  const reset = useCallback(() => {
    setStepStatus(initialSteps());
    setActivities([]);
    setServices(INITIAL_SERVICES);
    setItemProgress({ current: 0, total: 0, sku: "" });
    setReference(null);
    setQuotaAlerts([]);
  }, []);

  const handleEvent = useCallback((evt: ProgressEvent) => {
    setStepStatus((prev) => applyStepEvent(evt, prev));
    setServices((prev) => updateServices(evt, prev));

    if (evt.event === "step_complete" && evt.data.step === "parse" && evt.data.reference) {
      setReference(String(evt.data.reference));
    }
    if (evt.event === "item_start") {
      setItemProgress({
        current: Number(evt.data.index ?? 0),
        total: Number(evt.data.total ?? 0),
        sku: String(evt.data.sku ?? ""),
      });
    }
    if (evt.event === "pipeline_complete" && evt.data.reference) {
      setReference(String(evt.data.reference));
    }

    if (evt.event === "quota_limit") {
      setQuotaAlerts((prev) => mergeQuotaAlerts(prev, quotaAlertFromEvent(evt.data)));
    }
    if (evt.event === "firecrawl_summary") {
      const summaryAlert = quotaAlertFromFirecrawlSummary(evt.data);
      if (summaryAlert) {
        setQuotaAlerts((prev) => mergeQuotaAlerts(prev, summaryAlert));
      }
    }

    const activity = eventToActivity(evt);
    if (activity) {
      setActivities((prev) => [...prev, activity].slice(-200));
    }
  }, []);

  const ingestErrorMessage = useCallback((message: string) => {
    const fromError = quotaAlertsFromError(message);
    if (fromError.length === 0) return;
    setQuotaAlerts((prev) => {
      let next = prev;
      for (const alert of fromError) next = mergeQuotaAlerts(next, alert);
      return next;
    });
  }, []);

  const progressPercent = (() => {
    const weights = PIPELINE_STEPS.length;
    let done = 0;
    for (const s of PIPELINE_STEPS) {
      const st = stepStatus[s.id];
      if (st === "complete") done += 1;
      else if (st === "active") done += 0.45;
      else if (st === "warning") done += 0.8;
    }
    if (itemProgress.total > 0 && stepStatus.stage1 === "active") {
      done += (itemProgress.current / itemProgress.total) * 0.3;
    }
    return Math.min(100, Math.round((done / weights) * 100));
  })();

  return {
    stepStatus,
    activities,
    services,
    itemProgress,
    reference,
    progressPercent,
    quotaAlerts,
    handleEvent,
    ingestErrorMessage,
    reset,
  };
}
