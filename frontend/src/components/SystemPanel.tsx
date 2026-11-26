import type { HardwareInfoResponse, ModelStatusResponse, UsageSummaryResponse } from "../types/api";

interface SystemPanelProps {
  hasKey: boolean;
  models: ModelStatusResponse[];
  modelsError: string | null;
  hardware: HardwareInfoResponse | null;
  hardwareError: string | null;
  usage: UsageSummaryResponse | null;
  usageError: string | null;
  loadingSystem: boolean;
  onRefresh: () => void;
}

function modelPillClass(state: string): string {
  if (state === "ready") return "pill ok mono";
  if (state === "loading" || state === "unloading") return "pill warn mono";
  if (state === "failed") return "pill err mono";
  return "pill mono";
}

export function SystemPanel({
  hasKey,
  models,
  modelsError,
  hardware,
  hardwareError,
  usage,
  usageError,
  loadingSystem,
  onRefresh,
}: SystemPanelProps) {
  return (
    <section className="system-drawer" aria-label="Models and system telemetry">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.75rem" }}>
        <strong style={{ fontSize: "0.86rem" }}>Models, Usage & Hardware Telemetry</strong>
        <button type="button" onClick={onRefresh} disabled={!hasKey || loadingSystem}>
          {loadingSystem ? "Refreshing..." : "Refresh System Info"}
        </button>
      </div>

      {!hasKey && (
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
          Provide an <code className="mono">X-API-Key</code> to query <code className="mono">/api/v1/models</code>, <code className="mono">/api/v1/usage/summary</code>, and admin hardware telemetry.
        </p>
      )}

      {hasKey && (
        <div className="system-grid">
          <div className="panel-box">
            <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "0.45rem" }}>
              Model Registry (<code className="mono">/api/v1/models</code>)
            </div>
            {modelsError ? (
              <div style={{ fontSize: "0.8rem", color: "var(--accent-rose)" }}>{modelsError}</div>
            ) : models.length === 0 ? (
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Click Refresh System Info to inspect models.</div>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model ID</th>
                    <th>State</th>
                    <th>Device</th>
                    <th>Inferences</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.model_id}>
                      <td className="mono">{m.model_id}</td>
                      <td>
                        <span className={modelPillClass(m.state)}>{m.state}</span>
                      </td>
                      <td className="mono">{m.device ?? "-"}</td>
                      <td className="mono">{m.total_inferences}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="panel-box form-stack">
            <div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "0.3rem" }}>
                Quota Summary (<code className="mono">/api/v1/usage/summary</code>)
              </div>
              {usageError ? (
                <div style={{ fontSize: "0.8rem", color: "var(--accent-rose)" }}>{usageError}</div>
              ) : usage ? (
                <div className="mono" style={{ fontSize: "0.8rem" }}>
                  {usage.requests_used} / {usage.monthly_quota} used ({usage.quota_used_percentage}%) · {usage.quota_remaining} left
                </div>
              ) : (
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Not loaded.</div>
              )}
            </div>

            <div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "0.3rem" }}>
                Compute Hardware (<code className="mono">/api/v1/hardware</code>, admin scope)
              </div>
              {hardwareError ? (
                <div style={{ fontSize: "0.8rem", color: "var(--accent-amber)" }}>{hardwareError}</div>
              ) : hardware ? (
                <div className="mono" style={{ fontSize: "0.8rem" }}>
                  device={hardware.device_type} · cuda={String(hardware.cuda_available)} · {hardware.device_name} · cpus={hardware.cpu_count}
                </div>
              ) : (
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Not loaded.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
