import { useState } from "react";
import type { HealthResponse } from "../types/api";

interface ConnectionBarProps {
  baseUrl: string;
  apiKey: string;
  health: HealthResponse | null;
  healthError: string | null;
  checkingHealth: boolean;
  onBaseUrlChange: (value: string) => void;
  onApiKeyChange: (value: string) => void;
  onClearKey: () => void;
  onCheckHealth: () => void;
}

export function ConnectionBar({
  baseUrl,
  apiKey,
  health,
  healthError,
  checkingHealth,
  onBaseUrlChange,
  onApiKeyChange,
  onClearKey,
  onCheckHealth,
}: ConnectionBarProps) {
  const [revealKey, setRevealKey] = useState(false);

  const statusTone = checkingHealth
    ? "warn"
    : health
      ? "ok"
      : healthError
        ? "err"
        : "warn";
  const statusLabel = checkingHealth
    ? "Checking..."
    : health
      ? `${health.status} (${health.version})`
      : healthError
        ? "offline"
        : "untested";

  return (
    <header className="header-bar">
      <div className="brand">
        <h1>NLPulse Playground</h1>
        <span className="mono">local API console</span>
      </div>

      <div className="conn-controls">
        <div className="field-group">
          <label htmlFor="backend-url">Backend URL</label>
          <input
            id="backend-url"
            type="url"
            value={baseUrl}
            onChange={(e) => onBaseUrlChange(e.target.value)}
            placeholder="http://localhost:8000"
            style={{ width: "210px" }}
          />
        </div>

        <div className="field-group">
          <label htmlFor="api-key-input">X-API-Key</label>
          <input
            id="api-key-input"
            type={revealKey ? "text" : "password"}
            value={apiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            placeholder="nlp_live_..."
            autoComplete="off"
            style={{ width: "220px" }}
          />
          <button type="button" onClick={() => setRevealKey((v) => !v)}>
            {revealKey ? "Hide" : "Show"}
          </button>
          <button type="button" onClick={onClearKey} disabled={!apiKey}>
            Clear Key
          </button>
        </div>

        <button type="button" onClick={onCheckHealth} disabled={checkingHealth}>
          {checkingHealth ? "Checking..." : "Check Health"}
        </button>

        <span
          className={`pill mono ${statusTone}`}
          data-testid="health-status"
          title={healthError ?? health?.environment ?? ""}
        >
          {statusLabel}
        </span>
      </div>
    </header>
  );
}
