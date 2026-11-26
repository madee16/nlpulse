import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiRequestError, fetchHealth, invokeCapability, normalizeBaseUrl } from "./client";

describe("client API transport", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("normalizes trailing slashes from baseUrl", () => {
    expect(normalizeBaseUrl("http://localhost:8000///")).toBe("http://localhost:8000");
  });

  it("sends X-API-Key header cleanly and returns parsed JSON", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          label: "POSITIVE",
          confidence: 0.99,
          scores: { POSITIVE: 0.99, NEGATIVE: 0.01 },
          model_id: "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
          latency_ms: 12.5,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const res = await invokeCapability("http://localhost:8000/", "nlp_live_testkey", "sentiment", {
      text: "Great developer tools",
    });

    expect(res.status).toBe(200);
    expect(res.data.label).toBe("POSITIVE");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/nlp/sentiment",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          "X-API-Key": "nlp_live_testkey",
        }),
      })
    );
  });

  it("preserves HTTP status, Retry-After, and backend error detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Local Ollama service unavailable at http://localhost:11434." }), {
          status: 503,
          headers: { "Content-Type": "application/json", "Retry-After": "15" },
        })
      )
    );

    await expect(
      invokeCapability("http://localhost:8000", "nlp_live_testkey", "extraction", { text: "hi" })
    ).rejects.toMatchObject({
      name: "ApiRequestError",
      status: 503,
      retryAfter: "15",
      message: "Local Ollama service unavailable at http://localhost:11434.",
    } satisfies Partial<ApiRequestError>);
  });

  it("surfaces network failures cleanly with status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(fetchHealth("http://127.0.0.1:9999")).rejects.toMatchObject({
      status: 0,
      message: "Unable to reach backend (Failed to fetch)",
    });
  });
});
