export type NLPCapability =
  | "sentiment"
  | "ner"
  | "classification"
  | "keywords"
  | "summarization"
  | "extraction";

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface SentimentResponse {
  label: string;
  confidence: number;
  scores: Record<string, number>;
  model_id: string;
  latency_ms: number;
}

export interface EntityItem {
  entity: string;
  text: string;
  confidence: number;
  start_offset: number;
  end_offset: number;
}

export interface NERResponse {
  entities: EntityItem[];
  entity_count: number;
  model_id: string;
  latency_ms: number;
}

export interface LabelScore {
  label: string;
  score: number;
}

export interface ZeroShotResponse {
  top_label: string | null;
  top_score: number | null;
  predictions: LabelScore[];
  multi_label: boolean;
  model_id: string;
  latency_ms: number;
}

export interface KeywordItem {
  keyword: string;
  score: number;
}

export interface KeywordResponse {
  keywords: KeywordItem[];
  total_candidates: number;
  model_id: string;
  latency_ms: number;
}

export interface SummarizationResponse {
  summary: string;
  original_character_count: number;
  summary_character_count: number;
  chunks_processed: number;
  model_id: string;
  latency_ms: number;
}

export interface ExtractionFieldInput {
  name: string;
  type: "string" | "number" | "integer" | "boolean" | "array" | "object";
  description?: string;
  required: boolean;
}

export interface ExtractionResponse {
  extracted_data: Record<string, unknown>;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_units: number;
  latency_ms: number;
}

export type CapabilityResultMap = {
  sentiment: SentimentResponse;
  ner: NERResponse;
  classification: ZeroShotResponse;
  keywords: KeywordResponse;
  summarization: SummarizationResponse;
  extraction: ExtractionResponse;
};

export interface ModelStatusResponse {
  model_id: string;
  name: string;
  task: string;
  version: string;
  state: string;
  device: string | null;
  loaded_at: string | null;
  last_used_at: string | null;
  active_inferences: number;
  total_inferences: number;
  max_concurrency: number;
  vram_estimate_mb: number;
  ram_estimate_mb: number;
  error_message: string | null;
}

export interface HardwareInfoResponse {
  device_type: string;
  cuda_available: boolean;
  device_name: string;
  device_count: number;
  total_vram_mb: number | null;
  allocated_vram_mb: number | null;
  reserved_vram_mb: number | null;
  cpu_count: number;
  system_ram_mb: number | null;
}

export interface UsageSummaryResponse {
  developer_id: string;
  current_period: string;
  requests_used: number;
  monthly_quota: number;
  quota_remaining: number;
  quota_used_percentage: number;
  is_quota_exceeded: boolean;
}
