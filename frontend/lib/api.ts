export type ModelOption = {
  model_id: string;
  label: string;
  provider_mode: string;
  base_url?: string;
  model_name?: string;
};

export type AskResponse = {
  model_id: string;
  provider_mode: string;
  raw_output: string;
  reasoning: string;
  action_type: string | null;
  sql: string;
  total_elapsed_ms: number;
  model_elapsed_ms: number;
  execution?: {
    status: string;
    row_count: number;
    columns: string[];
    rows: Record<string, unknown>[];
    error_message: string;
    elapsed_ms: number;
  };
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function fetchModels(): Promise<ModelOption[]> {
  const res = await fetch(`${API_BASE_URL}/models`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Failed to fetch models");
  }
  return res.json();
}

export async function askQuestion(payload: {
  question: string;
  model_id: string;
  mode: "native" | "intelligent";
  execute_sql: boolean;
  knowledge?: string;
  table_list?: string[];
}): Promise<AskResponse> {
  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Request failed");
  }
  return res.json();
}
