import type {
  ArchitectureResponse,
  HistoryResponse,
  ModelsResponse,
  SearchResponse,
  VerifyResponse,
} from "./types";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API error ${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function verifyImages(
  imageA: File | Blob,
  imageB: File | Blob,
  model?: string
): Promise<VerifyResponse> {
  const form = new FormData();
  form.append("image_a", imageA);
  form.append("image_b", imageB);

  const url = new URL("/api/v1/verify", API_BASE_URL);
  if (model) url.searchParams.set("model", model);

  const response = await fetch(url, { method: "POST", body: form });
  return parseResponse<VerifyResponse>(response);
}

export async function searchImage(image: File | Blob, model?: string, k = 5): Promise<SearchResponse> {
  const form = new FormData();
  form.append("image", image);

  const url = new URL("/api/v1/search", API_BASE_URL);
  if (model) url.searchParams.set("model", model);
  url.searchParams.set("k", String(k));

  const response = await fetch(url, { method: "POST", body: form });
  return parseResponse<SearchResponse>(response);
}

export async function getHistory(limit = 20): Promise<HistoryResponse> {
  const url = new URL("/api/v1/history", API_BASE_URL);
  url.searchParams.set("limit", String(limit));

  const response = await fetch(url, { cache: "no-store" });
  return parseResponse<HistoryResponse>(response);
}

export async function getModels(): Promise<ModelsResponse> {
  const response = await fetch(new URL("/api/v1/models", API_BASE_URL), { cache: "no-store" });
  return parseResponse<ModelsResponse>(response);
}

export async function getArchitecture(): Promise<ArchitectureResponse> {
  const response = await fetch(new URL("/api/v1/architecture", API_BASE_URL), { cache: "no-store" });
  return parseResponse<ArchitectureResponse>(response);
}

export function liveSocketUrl(): string {
  const wsBase = API_BASE_URL.replace(/^http/, "ws");
  return `${wsBase}/api/v1/live`;
}
