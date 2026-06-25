import type { OrderRunResult, ProgressEvent } from "./types";

const STORAGE_KEY = "dental_api_base";

export function getApiBase(): string {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) return stored.replace(/\/+$/, "");
  return "/api";
}

export function setApiBase(url: string): void {
  localStorage.setItem(STORAGE_KEY, url.replace(/\/+$/, ""));
}

export function basename(path: string): string {
  if (!path) return "";
  return path.split(/[/\\]/).pop() ?? "";
}

export function reportDownloadUrl(filename: string): string {
  return `${getApiBase()}/reports/${encodeURIComponent(filename)}`;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${getApiBase()}/healthz`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return false;
    const data = await res.json();
    return data?.ok === true;
  } catch {
    return false;
  }
}

export async function uploadOrder(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${getApiBase()}/orders/run`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Upload failed (${res.status})`);
  }
  return data.job_id as string;
}

export function subscribeToJob(
  jobId: string,
  onEvent: (evt: ProgressEvent) => void,
): () => void {
  const es = new EventSource(`${getApiBase()}/orders/${jobId}/events`);

  es.onmessage = (msg) => {
    try {
      const evt = JSON.parse(msg.data) as ProgressEvent;
      onEvent(evt);
      if (evt.event === "pipeline_complete" || evt.event === "pipeline_error") {
        es.close();
      }
    } catch {
      /* ignore malformed */
    }
  };

  es.onerror = () => es.close();
  return () => es.close();
}

import type { OrderHistoryEntry } from "./types";
import { mapHistoryRow } from "./history";

export async function fetchOrderHistory(): Promise<OrderHistoryEntry[]> {
  const res = await fetch(`${getApiBase()}/orders/history`);
  if (!res.ok) throw new Error("Could not load order history");
  const data = await res.json();
  return (Array.isArray(data) ? data : []).map((row) =>
    mapHistoryRow(row as Record<string, unknown>),
  );
}

export async function runOrderStreaming(
  file: File,
  onEvent: (evt: ProgressEvent) => void,
  onJobCreated?: () => void,
): Promise<OrderRunResult> {
  const jobId = await uploadOrder(file);
  onJobCreated?.();
  onEvent({ event: "upload_complete", data: { job_id: jobId, filename: file.name }, ts: Date.now() / 1000 });

  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) {
        settled = true;
        window.clearInterval(pollId);
        unsubscribe();
        fn();
      }
    };

    const pollId = window.setInterval(async () => {
      if (settled) return;
      try {
        const res = await fetch(`${getApiBase()}/orders/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();
        if (job.status === "complete" && job.result) {
          settle(() => resolve(job.result as OrderRunResult));
        } else if (job.status === "failed") {
          settle(() => reject(new Error(job.error || "Pipeline failed")));
        }
      } catch {
        /* poll fallback only */
      }
    }, 4000);

    const unsubscribe = subscribeToJob(jobId, (evt) => {
      onEvent(evt);
      if (evt.event === "pipeline_complete") {
        settle(() => resolve(evt.data as unknown as OrderRunResult));
      }
      if (evt.event === "pipeline_error") {
        settle(() => reject(new Error(String(evt.data.message || "Pipeline failed"))));
      }
    });
  });
}

export function formatMoney(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n);
}

export function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min === 0) return `${sec}s`;
  return `${min}m ${sec.toString().padStart(2, "0")}s`;
}

export function formatTime(ts: number): string {
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(ts * 1000));
}

export function formatDateTime(iso: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(iso));
}
