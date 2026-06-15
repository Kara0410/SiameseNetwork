/** Mirrors `backend/app/schemas/*` - keep in sync with the FastAPI response models. */

export type Verdict = "verified" | "review" | "blocked";

export interface VerifyResponse {
  trace_id: string;
  model: string;
  similarity: number;
  distance: number;
  verdict: Verdict;
  spoof_risk: number;
  anomalies: string[];
  reasoning: string;
  latency_ms: number;
  heatmap_a: string | null;
  heatmap_b: string | null;
  created_at: string;
}

export interface SearchMatch {
  id: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface EmbeddingPoint {
  id: string;
  x: number;
  y: number;
  [key: string]: unknown;
}

export interface SearchResponse {
  model: string;
  matches: SearchMatch[];
  embedding_map: EmbeddingPoint[];
}

export interface HistoryItem {
  trace_id: string;
  model: string;
  similarity: number;
  distance: number;
  verdict: Verdict;
  spoof_risk: number;
  anomalies: string[];
  reasoning: string;
  latency_ms: number;
  created_at: string;
}

export interface HistoryResponse {
  items: HistoryItem[];
}

export interface ModelInfo {
  name: string;
  display_name: string;
  dimension: number;
  description: string;
  explainability: "gradcam" | "attention" | "heuristic" | "none";
}

export interface ModelsResponse {
  models: ModelInfo[];
  default: string;
}

export interface PipelineStage {
  id: string;
  layer: string;
  title: string;
  metric: string;
  detail: string;
  stack: string[];
}

export interface LatencyStage {
  stage: string;
  ms: number;
}

export interface DeploymentGroup {
  category: string;
  items: string[];
}

export interface ArchitectureResponse {
  pipeline: PipelineStage[];
  latency: LatencyStage[];
  deployment: DeploymentGroup[];
}

export interface LiveFrameResult {
  model: string;
  similarity: number;
  spoof_risk: number;
  flags: string[];
  latency_ms: number;
}

export interface LiveFrameError {
  error: string;
}
