import { useCallback, useEffect, useState } from "react";
import {
  ApiRequestError,
  fetchHardware,
  fetchHealth,
  fetchModels,
  fetchUsageSummary,
  invokeCapability,
} from "./api/client";
import { CapabilityForms } from "./components/CapabilityForms";
import { ConnectionBar } from "./components/ConnectionBar";
import { ResultPanel } from "./components/ResultPanel";
import { SystemPanel } from "./components/SystemPanel";
import {
  clearSessionApiKey,
  loadSessionApiKey,
  loadSessionBaseUrl,
  saveSessionApiKey,
  saveSessionBaseUrl,
} from "./session";
import type {
  CapabilityResultMap,
  HardwareInfoResponse,
  HealthResponse,
  ModelStatusResponse,
  NLPCapability,
  UsageSummaryResponse,
} from "./types/api";

export default function App() {
  const [baseUrl, setBaseUrl] = useState(loadSessionBaseUrl);
  const [apiKey, setApiKey] = useState(loadSessionApiKey);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);

  const [capability, setCapability] = useState<NLPCapability>("sentiment");
  const [loadingInference, setLoadingInference] = useState(false);
  const [responseMeta, setResponseMeta] = useState<{
    capability: NLPCapability;
    endpoint: string;
    status: number;
    data: CapabilityResultMap[NLPCapability];
  } | null>(null);
  const [inferenceError, setInferenceError] = useState<ApiRequestError | null>(null);

  const [models, setModels] = useState<ModelStatusResponse[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [hardware, setHardware] = useState<HardwareInfoResponse | null>(null);
  const [hardwareError, setHardwareError] = useState<string | null>(null);
  const [usage, setUsage] = useState<UsageSummaryResponse | null>(null);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [loadingSystem, setLoadingSystem] = useState(false);

  const handleCheckHealth = useCallback(async () => {
    setCheckingHealth(true);
    setHealthError(null);
    try {
      const h = await fetchHealth(baseUrl);
      setHealth(h);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : "Health check failed");
    } finally {
      setCheckingHealth(false);
    }
  }, [baseUrl]);

  useEffect(() => {
    void handleCheckHealth();
  }, [handleCheckHealth]);

  const handleRefreshSystem = useCallback(async () => {
    if (!apiKey.trim()) return;
    setLoadingSystem(true);
    setModelsError(null);
    setHardwareError(null);
    setUsageError(null);

    const [modelsRes, hardwareRes, usageRes] = await Promise.allSettled([
      fetchModels(baseUrl, apiKey),
      fetchHardware(baseUrl, apiKey),
      fetchUsageSummary(baseUrl, apiKey),
    ]);

    if (modelsRes.status === "fulfilled") {
      setModels(modelsRes.value);
    } else {
      setModels([]);
      setModelsError(modelsRes.reason instanceof Error ? modelsRes.reason.message : "Failed to load models");
    }

    if (hardwareRes.status === "fulfilled") {
      setHardware(hardwareRes.value);
    } else {
      setHardware(null);
      setHardwareError(hardwareRes.reason instanceof Error ? hardwareRes.reason.message : "Failed to load hardware");
    }

    if (usageRes.status === "fulfilled") {
      setUsage(usageRes.value);
    } else {
      setUsage(null);
      setUsageError(usageRes.reason instanceof Error ? usageRes.reason.message : "Failed to load usage");
    }

    setLoadingSystem(false);
  }, [apiKey, baseUrl]);

  async function handleRunCapability(cap: NLPCapability, payload: Record<string, unknown>) {
    setLoadingInference(true);
    setInferenceError(null);
    try {
      const res = await invokeCapability(baseUrl, apiKey, cap, payload);
      setResponseMeta({
        capability: cap,
        endpoint: res.endpoint,
        status: res.status,
        data: res.data,
      });
      if (apiKey.trim()) {
        void handleRefreshSystem();
      }
    } catch (err) {
      setResponseMeta(null);
      setInferenceError(
        err instanceof ApiRequestError
          ? err
          : new ApiRequestError(err instanceof Error ? err.message : "Unexpected error", 0)
      );
    } finally {
      setLoadingInference(false);
    }
  }

  function handleBaseUrlChange(val: string) {
    setBaseUrl(val);
    saveSessionBaseUrl(val);
  }

  function handleApiKeyChange(val: string) {
    setApiKey(val);
    saveSessionApiKey(val);
  }

  function handleClearKey() {
    clearSessionApiKey();
    setApiKey("");
    setModels([]);
    setHardware(null);
    setUsage(null);
    setModelsError(null);
    setHardwareError(null);
    setUsageError(null);
  }

  return (
    <div className="shell">
      <ConnectionBar
        baseUrl={baseUrl}
        apiKey={apiKey}
        health={health}
        healthError={healthError}
        checkingHealth={checkingHealth}
        onBaseUrlChange={handleBaseUrlChange}
        onApiKeyChange={handleApiKeyChange}
        onClearKey={handleClearKey}
        onCheckHealth={() => void handleCheckHealth()}
      />

      <main className="workspace">
        <CapabilityForms
          capability={capability}
          loading={loadingInference}
          onSelectCapability={(cap) => {
            setCapability(cap);
            setInferenceError(null);
          }}
          onSubmit={(cap, payload) => void handleRunCapability(cap, payload)}
        />

        <ResultPanel
          capability={capability}
          loading={loadingInference}
          responseMeta={responseMeta}
          error={inferenceError}
        />
      </main>

      <SystemPanel
        hasKey={Boolean(apiKey.trim())}
        models={models}
        modelsError={modelsError}
        hardware={hardware}
        hardwareError={hardwareError}
        usage={usage}
        usageError={usageError}
        loadingSystem={loadingSystem}
        onRefresh={() => void handleRefreshSystem()}
      />
    </div>
  );
}
