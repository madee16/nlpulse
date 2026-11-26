import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

describe("NLPulse Playground App", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("stores API key only in sessionStorage and clears it on Clear Key", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            status: "healthy",
            service: "NLPulse",
            version: "0.1.0",
            environment: "development",
            timestamp: "2026-11-26T15:00:00Z",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
    );

    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId("health-status")).toHaveTextContent("healthy (0.1.0)");
    });

    const keyInput = screen.getByLabelText("X-API-Key");
    fireEvent.change(keyInput, { target: { value: "nlp_live_session_only" } });

    expect(sessionStorage.getItem("nlpulse.session.apiKey")).toBe("nlp_live_session_only");
    expect(localStorage.length).toBe(0);

    fireEvent.click(screen.getByRole("button", { name: "Clear Key" }));
    expect(sessionStorage.getItem("nlpulse.session.apiKey")).toBeNull();
    expect((keyInput as HTMLInputElement).value).toBe("");
  });

  it("switches capability tabs and surfaces real backend API errors cleanly", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              status: "healthy",
              service: "NLPulse",
              version: "0.1.0",
              environment: "development",
              timestamp: "2026-11-26T15:00:00Z",
            }),
            { status: 200 }
          )
        );
      }
      return Promise.resolve(
        new Response(JSON.stringify({ detail: "API key required." }), { status: 401 })
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Extraction" }));
    fireEvent.click(screen.getByRole("button", { name: /Send POST \/api\/v1\/nlp\/extraction/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Backend Error (HTTP 401)");
      expect(screen.getByRole("alert")).toHaveTextContent("API key required.");
    });
  });
});
