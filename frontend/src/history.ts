import type { OrderHistoryEntry } from "./types";

export function mapHistoryRow(row: Record<string, unknown>): OrderHistoryEntry {
  const reports = row.reports as OrderHistoryEntry["reports"] | undefined;
  return {
    id: String(row.id),
    fileName: String(row.fileName ?? ""),
    reference: String(row.reference ?? ""),
    items: Number(row.items ?? 0),
    total: row.total != null ? Number(row.total) : null,
    status: row.status as OrderHistoryEntry["status"],
    createdAt: String(row.createdAt ?? ""),
    completedAt: row.completedAt ? String(row.completedAt) : undefined,
    durationMs: row.durationMs != null ? Number(row.durationMs) : undefined,
    reports,
    error: row.error ? String(row.error) : undefined,
  };
}
