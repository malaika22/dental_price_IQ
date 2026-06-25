export type StepStatus = "pending" | "active" | "complete" | "error" | "warning";

export type PipelineStepId =
  | "upload"
  | "parse"
  | "ai"
  | "mpn"
  | "stage1"
  | "stage2"
  | "reports";

export interface PipelineStepDef {
  id: PipelineStepId;
  label: string;
  description: string;
  icon: string;
}

export interface OrderRunResult {
  reference: string;
  items: number;
  total?: number;
  computed_total?: number;
  price_match_report: string;
  alternate_purchase_list: string;
  evidence_file: string;
}

export interface OrderHistoryEntry {
  id: string;
  fileName: string;
  reference: string;
  items: number;
  total: number | null;
  status: "processing" | "completed" | "failed";
  createdAt: string;
  completedAt?: string;
  durationMs?: number;
  reports?: {
    price_match_report: string;
    alternate_purchase_list: string;
    evidence_file: string;
  };
  error?: string;
}

export interface ProgressEvent {
  event: string;
  data: Record<string, unknown>;
  ts: number;
}

export interface ActivityEntry {
  id: string;
  ts: number;
  service: "system" | "groq" | "firecrawl" | "serpapi" | "parse" | "mpn";
  message: string;
  detail?: string;
}

export interface ServiceState {
  groq: "idle" | "active" | "done" | "error";
  firecrawl: "idle" | "active" | "done" | "error";
  serpapi: "idle" | "active" | "done" | "error";
}

export const PIPELINE_STEPS: PipelineStepDef[] = [
  {
    id: "upload",
    label: "Upload",
    description: "Receiving your PDF on the server",
    icon: "↑",
  },
  {
    id: "parse",
    label: "Parse PDF",
    description: "Extracting line items, SKUs, and prices",
    icon: "📄",
  },
  {
    id: "ai",
    label: "AI Enrichment",
    description: "Groq/Gemini parsing brands, variants, and search queries",
    icon: "🧠",
  },
  {
    id: "mpn",
    label: "MPN Lookup",
    description: "Manufacturer part number matching",
    icon: "🔍",
  },
  {
    id: "stage1",
    label: "Price Search",
    description: "SerpAPI discovery, Firecrawl scraping, match validation",
    icon: "💰",
  },
  {
    id: "stage2",
    label: "Equivalency",
    description: "Alternate product evaluation",
    icon: "🔄",
  },
  {
    id: "reports",
    label: "Reports",
    description: "Generating Excel workbooks",
    icon: "📊",
  },
];

export const REPORT_META = [
  {
    key: "price_match_report" as const,
    title: "Price Match Report",
    desc: "Exact matches with savings",
    icon: "📊",
  },
  {
    key: "alternate_purchase_list" as const,
    title: "Alternate Purchases",
    desc: "Equivalent product options",
    icon: "🔄",
  },
  {
    key: "evidence_file" as const,
    title: "Evidence File",
    desc: "Full audit of all candidates",
    icon: "📋",
  },
];
