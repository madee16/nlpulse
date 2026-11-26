import type {
  CapabilityResultMap,
  HardwareInfoResponse,
  HealthResponse,
  ModelStatusResponse,
  NLPCapability,
  UsageSummaryResponse,
} from "../types/api";

export class ApiRequestError extends Error {
  readonly status: number;
  readonly retryAfter: string | null;
  readonly rawBody: unknown;

  constructor(message: string, status: number, retryAfter: string | null = null, rawBody: unknown = null) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.retryAfter = retryAfter;
    this.rawBody = rawBody;
  }
}

export function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function formatErrorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const loc = Array.isArray((item as { loc?: unknown }).loc)
              ? (item as { loc: unknown[] }).loc.join(".")
              : "";
            return loc ? `${loc}: ${String((item as { msg: unknown }).msg)}` : String((item as { msg: unknown }).msg);
          }
          return JSON.stringify(item);
        })
        .join("; ");
    }
  }
  return `Request failed with HTTP ${status}`;
}

async function requestJson<T>(
  baseUrl: string,
  path: string,
  options: { method?: "GET" | "POST"; apiKey?: string; body?: unknown } = {}
): Promise<{ data: T; status: number; endpoint: string }> {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`;
  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (options.apiKey?.trim()) {
    headers["X-API-Key"] = options.apiKey.trim();
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? "GET",
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    });
  } catch (err) {
    const reason = err instanceof Error ? err.message : "Network connection failed";
    throw new ApiRequestError(`Unable to reach backend (${reason})`, 0);
  }

  const rawText = await response.text();
  let parsed: unknown = null;
  if (rawText) {
    try {
      parsed = JSON.parse(rawText);
    } catch {
      parsed = rawText;
    }
  }

  if (!response.ok) {
    const retryAfter = response.headers.get("Retry-After");
    throw new ApiRequestError(
      formatErrorMessage(response.status, parsed),
      response.status,
      retryAfter,
      parsed
    );
  }

  return { data: parsed as T, status: response.status, endpoint: path };
}

export async function fetchHealth(baseUrl: string): Promise<HealthResponse> {
  const res = await requestJson<HealthResponse>(baseUrl, "/api/v1/health");
  return res.data;
}

export async function fetchModels(baseUrl: string, apiKey: string): Promise<ModelStatusResponse[]> {
  const res = await requestJson<ModelStatusResponse[]>(baseUrl, "/api/v1/models", { apiKey });
  return res.data;
}

export async function fetchHardware(baseUrl: string, apiKey: string): Promise<HardwareInfoResponse> {
  const res = await requestJson<HardwareInfoResponse>(baseUrl, "/api/v1/hardware", { apiKey });
  return res.data;
}

export async function fetchUsageSummary(baseUrl: string, apiKey: string): Promise<UsageSummaryResponse> {
  const res = await requestJson<UsageSummaryResponse>(baseUrl, "/api/v1/usage/summary", { apiKey });
  return res.data;
}

export async function invokeCapability<K extends NLPCapability>(
  baseUrl: string,
  apiKey: string,
  capability: K,
  payload: Record<string, unknown>
): Promise<{ data: CapabilityResultMap[K]; status: number; endpoint: string }> {
  return requestJson<CapabilityResultMap[K]>(baseUrl, `/api/v1/nlp/${capability}`, {
    method: "POST",
    apiKey,
    body: payload,
  });
}
