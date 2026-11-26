import { useState } from "react";
import type {
  CapabilityResultMap,
  ExtractionResponse,
  KeywordResponse,
  NERResponse,
  NLPCapability,
  SentimentResponse,
  SummarizationResponse,
  ZeroShotResponse,
} from "../types/api";
import type { ApiRequestError } from "../api/client";

interface ResultPanelProps {
  capability: NLPCapability;
  loading: boolean;
  responseMeta: {
    capability: NLPCapability;
    endpoint: string;
    status: number;
    data: CapabilityResultMap[NLPCapability];
  } | null;
  error: ApiRequestError | null;
}

function renderStructuredView(capability: NLPCapability, payload: CapabilityResultMap[NLPCapability]) {
  switch (capability) {
    case "sentiment": {
      const res = payload as SentimentResponse;
      const entries = Object.entries(res.scores);
      return (
        <div className="form-stack">
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "baseline" }}>
            <span className="pill ok mono">{res.label}</span>
            <span className="mono" style={{ fontSize: "0.85rem" }}>
              confidence {(res.confidence * 100).toFixed(2)}%
            </span>
          </div>
          {entries.map(([lbl, score]) => (
            <div key={lbl}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                <span className="mono">{lbl}</span>
                <span className="mono">{(score * 100).toFixed(2)}%</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }} />
              </div>
            </div>
          ))}
        </div>
      );
    }

    case "ner": {
      const res = payload as NERResponse;
      return (
        <div className="form-stack">
          <div className="mono" style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            entities detected: {res.entity_count}
          </div>
          {res.entities.length === 0 ? (
            <p style={{ margin: 0, fontSize: "0.84rem", color: "var(--text-secondary)" }}>
              No named entities found.
            </p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Entity</th>
                  <th>Text</th>
                  <th>Offsets</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {res.entities.map((ent, idx) => (
                  <tr key={idx}>
                    <td className="mono">{ent.entity}</td>
                    <td>{ent.text}</td>
                    <td className="mono">
                      [{ent.start_offset}, {ent.end_offset})
                    </td>
                    <td className="mono">{(ent.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      );
    }

    case "classification": {
      const res = payload as ZeroShotResponse;
      return (
        <div className="form-stack">
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <span className="pill ok mono">top: {res.top_label ?? "none"}</span>
            <span className="mono" style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              mode: {res.multi_label ? "multi_label" : "single_label (softmax)"}
            </span>
          </div>
          {res.predictions.map((pred) => (
            <div key={pred.label}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                <span>{pred.label}</span>
                <span className="mono">{(pred.score * 100).toFixed(2)}%</span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${Math.min(100, Math.max(0, pred.score * 100))}%` }} />
              </div>
            </div>
          ))}
        </div>
      );
    }

    case "keywords": {
      const res = payload as KeywordResponse;
      return (
        <div className="form-stack">
          <div className="mono" style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            total candidates evaluated: {res.total_candidates}
          </div>
          {res.keywords.length === 0 ? (
            <p style={{ margin: 0, fontSize: "0.84rem", color: "var(--text-secondary)" }}>
              No candidate keywords ranked.
            </p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Keyword / Keyphrase</th>
                  <th>Cosine Score</th>
                </tr>
              </thead>
              <tbody>
                {res.keywords.map((kw) => (
                  <tr key={kw.keyword}>
                    <td>{kw.keyword}</td>
                    <td className="mono">{kw.score.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      );
    }

    case "summarization": {
      const res = payload as SummarizationResponse;
      return (
        <div className="form-stack">
          <div className="panel-box" style={{ background: "var(--bg-inset)" }}>
            <p style={{ margin: 0, fontSize: "0.9rem" }}>{res.summary}</p>
          </div>
          <div className="mono" style={{ display: "flex", flexWrap: "wrap", gap: "1rem", fontSize: "0.78rem", color: "var(--text-secondary)" }}>
            <span>chars: {res.original_character_count} → {res.summary_character_count}</span>
            <span>chunks_processed: {res.chunks_processed}</span>
          </div>
        </div>
      );
    }

    case "extraction": {
      const res = payload as ExtractionResponse;
      return (
        <div className="form-stack">
          <div className="mono" style={{ display: "flex", flexWrap: "wrap", gap: "1rem", fontSize: "0.78rem", color: "var(--text-secondary)" }}>
            <span>prompt_tokens: {res.prompt_tokens}</span>
            <span>completion_tokens: {res.completion_tokens}</span>
            <span>total_tokens: {res.total_tokens}</span>
            <span>cost_units: {res.cost_units}</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Field</th>
                <th>Extracted Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(res.extracted_data).map(([key, val]) => (
                <tr key={key}>
                  <td className="mono">{key}</td>
                  <td className="mono">{val === null ? "null" : typeof val === "object" ? JSON.stringify(val) : String(val)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
  }
}

export function ResultPanel({ capability, loading, responseMeta, error }: ResultPanelProps) {
  const [viewMode, setViewMode] = useState<"structured" | "raw">("structured");

  const activeData = responseMeta?.data;
  const activeCap = responseMeta?.capability ?? capability;
  const modelLabel = activeData
    ? "model_id" in activeData
      ? activeData.model_id
      : activeData.model
    : null;

  return (
    <section className="pane" aria-label="Response output">
      <div className="telemetry-strip">
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <span className="mono" style={{ color: "var(--text-secondary)" }}>
            {responseMeta ? `POST ${responseMeta.endpoint}` : `Target: /api/v1/nlp/${capability}`}
          </span>
          {responseMeta && <span className="pill ok mono">HTTP {responseMeta.status}</span>}
          {error && <span className="pill err mono">HTTP {error.status || "NETWORK"}</span>}
        </div>

        <div style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
          {activeData && (
            <span className="mono" style={{ color: "var(--text-secondary)" }}>
              {activeData.latency_ms} ms {modelLabel ? `· ${modelLabel}` : ""}
            </span>
          )}
          <button
            type="button"
            className={viewMode === "structured" ? "cap-tab active" : "cap-tab"}
            onClick={() => setViewMode("structured")}
          >
            Structured
          </button>
          <button
            type="button"
            className={viewMode === "raw" ? "cap-tab active" : "cap-tab"}
            onClick={() => setViewMode("raw")}
          >
            Raw JSON
          </button>
        </div>
      </div>

      {loading && (
        <div className="panel-box mono" style={{ color: "var(--accent-amber)", fontSize: "0.84rem" }}>
          Waiting for backend response...
        </div>
      )}

      {error && (
        <div className="error-banner" role="alert">
          <div className="mono" style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
            Backend Error {error.status ? `(HTTP ${error.status})` : "(Network)"}
            {error.retryAfter ? ` · Retry-After: ${error.retryAfter}s` : ""}
          </div>
          <div>{error.message}</div>
          {error.rawBody != null && viewMode === "raw" && (
            <pre className="raw-json mono" style={{ marginTop: "0.6rem" }}>
              {JSON.stringify(error.rawBody, null, 2)}
            </pre>
          )}
        </div>
      )}

      {!loading && !error && !responseMeta && (
        <div className="panel-box" style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
          Enter an <code className="mono">X-API-Key</code> in the header bar and send a request to inspect live backend results.
        </div>
      )}

      {!loading && responseMeta && (
        viewMode === "structured" ? (
          <div className="panel-box">{renderStructuredView(activeCap, responseMeta.data)}</div>
        ) : (
          <pre className="raw-json mono">{JSON.stringify(responseMeta.data, null, 2)}</pre>
        )
      )}
    </section>
  );
}
