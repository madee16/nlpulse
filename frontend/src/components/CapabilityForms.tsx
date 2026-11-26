import { useState, type FormEvent } from "react";
import type { ExtractionFieldInput, NLPCapability } from "../types/api";

interface CapabilityFormsProps {
  capability: NLPCapability;
  loading: boolean;
  onSelectCapability: (cap: NLPCapability) => void;
  onSubmit: (capability: NLPCapability, payload: Record<string, unknown>) => void;
}

const CAPABILITY_LABELS: { id: NLPCapability; label: string }[] = [
  { id: "sentiment", label: "Sentiment" },
  { id: "ner", label: "NER" },
  { id: "classification", label: "Zero-Shot" },
  { id: "keywords", label: "Keywords" },
  { id: "summarization", label: "Summarization" },
  { id: "extraction", label: "Extraction" },
];

const DEFAULT_JSON_SCHEMA = JSON.stringify(
  {
    type: "object",
    additionalProperties: false,
    required: ["company"],
    properties: {
      company: { type: "string", minLength: 1 },
      revenue: { type: ["string", "null"] },
      location: { type: ["string", "null"] },
    },
  },
  null,
  2
);

export function CapabilityForms({
  capability,
  loading,
  onSelectCapability,
  onSubmit,
}: CapabilityFormsProps) {
  const [text, setText] = useState(
    "Apple Inc. announced quarterly revenue of $89.5 billion in Cupertino, California."
  );

  // Zero-shot
  const [labelsInput, setLabelsInput] = useState("finance, technology, healthcare, sports");
  const [multiLabel, setMultiLabel] = useState(false);

  // Keywords
  const [maxKeywords, setMaxKeywords] = useState(5);
  const [ngramMin, setNgramMin] = useState(1);
  const [ngramMax, setNgramMax] = useState(2);
  const [candidatesInput, setCandidatesInput] = useState("");

  // Summarization
  const [minLength, setMinLength] = useState(15);
  const [maxLength, setMaxLength] = useState(80);
  const [doSample, setDoSample] = useState(false);
  const [sumTemperature, setSumTemperature] = useState(0.7);

  // Extraction
  const [extractionMode, setExtractionMode] = useState<"fields" | "schema">("fields");
  const [fields, setFields] = useState<ExtractionFieldInput[]>([
    { name: "company", type: "string", description: "Company name", required: true },
    { name: "revenue", type: "string", description: "Reported revenue", required: false },
  ]);
  const [schemaText, setSchemaText] = useState(DEFAULT_JSON_SCHEMA);
  const [schemaParseError, setSchemaParseError] = useState<string | null>(null);
  const [instructions, setInstructions] = useState("");
  const [ollamaModel, setOllamaModel] = useState("");
  const [extractTemp, setExtractTemp] = useState(0.0);

  function handleFormSubmit(e: FormEvent) {
    e.preventDefault();
    setSchemaParseError(null);

    if (capability === "sentiment" || capability === "ner") {
      onSubmit(capability, { text });
      return;
    }

    if (capability === "classification") {
      const candidate_labels = labelsInput
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      onSubmit("classification", {
        text,
        candidate_labels,
        multi_label: multiLabel,
      });
      return;
    }

    if (capability === "keywords") {
      const candidates = candidatesInput
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const payload: Record<string, unknown> = {
        text,
        max_keywords: maxKeywords,
        ngram_range: [ngramMin, ngramMax],
      };
      if (candidates.length > 0) {
        payload.candidates = candidates;
      }
      onSubmit("keywords", payload);
      return;
    }

    if (capability === "summarization") {
      const payload: Record<string, unknown> = {
        text,
        min_length: minLength,
        max_length: maxLength,
        do_sample: doSample,
      };
      if (doSample) {
        payload.temperature = sumTemperature;
      }
      onSubmit("summarization", payload);
      return;
    }

    // Extraction
    const payload: Record<string, unknown> = {
      text,
      temperature: extractTemp,
    };
    if (instructions.trim()) {
      payload.instructions = instructions.trim();
    }
    if (ollamaModel.trim()) {
      payload.model = ollamaModel.trim();
    }
    if (extractionMode === "fields") {
      payload.fields = fields.map((f) => ({
        name: f.name,
        type: f.type,
        description: f.description?.trim() || undefined,
        required: f.required,
      }));
    } else {
      try {
        payload.schema_definition = JSON.parse(schemaText);
      } catch (err) {
        setSchemaParseError(
          err instanceof Error ? `Invalid schema JSON: ${err.message}` : "Invalid schema JSON"
        );
        return;
      }
    }
    onSubmit("extraction", payload);
  }

  return (
    <section className="pane" aria-label="Request composer">
      <div className="cap-tabs" role="tablist" aria-label="NLP capabilities">
        {CAPABILITY_LABELS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={capability === item.id}
            className={`cap-tab ${capability === item.id ? "active" : ""}`}
            onClick={() => onSelectCapability(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <form className="form-stack" onSubmit={handleFormSubmit}>
        <div className="form-row">
          <label htmlFor="nlp-input-text">Input Text</label>
          <textarea
            id="nlp-input-text"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter document or passage to analyze..."
          />
        </div>

        {capability === "classification" && (
          <>
            <div className="form-row">
              <label htmlFor="zs-labels">Candidate Labels (comma-separated)</label>
              <input
                id="zs-labels"
                type="text"
                value={labelsInput}
                onChange={(e) => setLabelsInput(e.target.value)}
              />
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.82rem" }}>
              <input
                type="checkbox"
                checked={multiLabel}
                onChange={(e) => setMultiLabel(e.target.checked)}
              />
              Multi-label scoring
            </label>
          </>
        )}

        {capability === "keywords" && (
          <>
            <div className="inline-grid">
              <div className="form-row">
                <label htmlFor="kw-max">Max Keywords</label>
                <input
                  id="kw-max"
                  type="number"
                  min={1}
                  max={50}
                  value={maxKeywords}
                  onChange={(e) => setMaxKeywords(Number(e.target.value))}
                />
              </div>
              <div className="form-row">
                <label htmlFor="kw-min-n">Min N-Gram</label>
                <input
                  id="kw-min-n"
                  type="number"
                  min={1}
                  max={3}
                  value={ngramMin}
                  onChange={(e) => setNgramMin(Number(e.target.value))}
                />
              </div>
              <div className="form-row">
                <label htmlFor="kw-max-n">Max N-Gram</label>
                <input
                  id="kw-max-n"
                  type="number"
                  min={1}
                  max={3}
                  value={ngramMax}
                  onChange={(e) => setNgramMax(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="form-row">
              <label htmlFor="kw-candidates">Optional Candidates (comma-separated)</label>
              <input
                id="kw-candidates"
                type="text"
                value={candidatesInput}
                onChange={(e) => setCandidatesInput(e.target.value)}
                placeholder="Leave blank to extract n-grams automatically"
              />
            </div>
          </>
        )}

        {capability === "summarization" && (
          <>
            <div className="inline-grid">
              <div className="form-row">
                <label htmlFor="sum-min">Min Length (tokens)</label>
                <input
                  id="sum-min"
                  type="number"
                  min={5}
                  max={500}
                  value={minLength}
                  onChange={(e) => setMinLength(Number(e.target.value))}
                />
              </div>
              <div className="form-row">
                <label htmlFor="sum-max">Max Length (tokens)</label>
                <input
                  id="sum-max"
                  type="number"
                  min={10}
                  max={1000}
                  value={maxLength}
                  onChange={(e) => setMaxLength(Number(e.target.value))}
                />
              </div>
              {doSample && (
                <div className="form-row">
                  <label htmlFor="sum-temp">Temperature</label>
                  <input
                    id="sum-temp"
                    type="number"
                    step="0.1"
                    min={0.1}
                    max={2.0}
                    value={sumTemperature}
                    onChange={(e) => setSumTemperature(Number(e.target.value))}
                  />
                </div>
              )}
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.82rem" }}>
              <input
                type="checkbox"
                checked={doSample}
                onChange={(e) => setDoSample(e.target.checked)}
              />
              Enable sampling (`do_sample`)
            </label>
          </>
        )}

        {capability === "extraction" && (
          <>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>Schema Mode:</span>
              <button
                type="button"
                className={extractionMode === "fields" ? "cap-tab active" : "cap-tab"}
                onClick={() => setExtractionMode("fields")}
              >
                Fields Builder
              </button>
              <button
                type="button"
                className={extractionMode === "schema" ? "cap-tab active" : "cap-tab"}
                onClick={() => setExtractionMode("schema")}
              >
                JSON Schema
              </button>
            </div>

            {extractionMode === "fields" ? (
              <div className="panel-box form-stack">
                {fields.map((field, idx) => (
                  <div
                    key={idx}
                    style={{ display: "grid", gridTemplateColumns: "1fr 110px 75px auto", gap: "0.4rem", alignItems: "center" }}
                  >
                    <input
                      type="text"
                      value={field.name}
                      aria-label={`Field ${idx + 1} name`}
                      onChange={(e) => {
                        const next = [...fields];
                        next[idx] = { ...field, name: e.target.value };
                        setFields(next);
                      }}
                      placeholder="field_name"
                    />
                    <select
                      aria-label={`Field ${idx + 1} type`}
                      value={field.type}
                      onChange={(e) => {
                        const next = [...fields];
                        next[idx] = {
                          ...field,
                          type: e.target.value as ExtractionFieldInput["type"],
                        };
                        setFields(next);
                      }}
                    >
                      <option value="string">string</option>
                      <option value="number">number</option>
                      <option value="integer">integer</option>
                      <option value="boolean">boolean</option>
                      <option value="array">array</option>
                      <option value="object">object</option>
                    </select>
                    <label style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontSize: "0.75rem" }}>
                      <input
                        type="checkbox"
                        checked={field.required}
                        onChange={(e) => {
                          const next = [...fields];
                          next[idx] = { ...field, required: e.target.checked };
                          setFields(next);
                        }}
                      />
                      req
                    </label>
                    <button
                      type="button"
                      disabled={fields.length <= 1}
                      onClick={() => setFields(fields.filter((_, i) => i !== idx))}
                    >
                      Remove
                    </button>
                  </div>
                ))}
                <div>
                  <button
                    type="button"
                    onClick={() =>
                      setFields([...fields, { name: "", type: "string", required: false }])
                    }
                  >
                    + Add Field
                  </button>
                </div>
              </div>
            ) : (
              <div className="form-row">
                <label htmlFor="ext-schema-json">JSON Schema (`schema_definition`)</label>
                <textarea
                  id="ext-schema-json"
                  className="code-input"
                  rows={8}
                  value={schemaText}
                  onChange={(e) => setSchemaText(e.target.value)}
                />
                {schemaParseError && (
                  <div className="error-banner" role="alert">
                    {schemaParseError}
                  </div>
                )}
              </div>
            )}

            <div className="inline-grid">
              <div className="form-row">
                <label htmlFor="ext-model">Ollama Model Override</label>
                <input
                  id="ext-model"
                  type="text"
                  value={ollamaModel}
                  onChange={(e) => setOllamaModel(e.target.value)}
                  placeholder="Default (qwen2.5:3b)"
                />
              </div>
              <div className="form-row">
                <label htmlFor="ext-temp">Temperature</label>
                <input
                  id="ext-temp"
                  type="number"
                  step="0.1"
                  min={0.0}
                  max={2.0}
                  value={extractTemp}
                  onChange={(e) => setExtractTemp(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="form-row">
              <label htmlFor="ext-instructions">Optional Instructions</label>
              <input
                id="ext-instructions"
                type="text"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="Additional grounding hints..."
              />
            </div>
          </>
        )}

        <div>
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Running Request..." : `Send POST /api/v1/nlp/${capability}`}
          </button>
        </div>
      </form>
    </section>
  );
}
