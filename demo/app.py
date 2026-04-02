import { useState, useRef, useEffect, useCallback } from "react";

// ════════════════════════════════════════════════════════════
// CONFIG & CONSTANTS
// ════════════════════════════════════════════════════════════

const API_BASE = "http://localhost:8000";

const ATTR_ORDER = [
  "primary_color", "secondary_color",
  "primary_material", "secondary_material",
  "style", "shape", "assembly",
];

const ATTR_LABELS = {
  primary_color: "Primary Color",
  secondary_color: "Secondary Color",
  primary_material: "Primary Material",
  secondary_material: "Secondary Material",
  style: "Style",
  shape: "Shape",
  assembly: "Assembly",
};

const DEFAULT_VOCAB = {
  primary_color: ["beige","black","blue","brown","burgundy","clear","cream","dark_brown","dark_gray","gold","gold_metal","green","gray","light_blue","light_brown","light_gray","multi","natural","navy","orange","pink","purple","red","rust","sage","silver","teal","white","yellow"],
  secondary_color: ["beige","black","blue","brown","burgundy","clear","cream","dark_brown","dark_gray","gold","gold_metal","green","gray","light_blue","light_brown","light_gray","multi","natural","navy","orange","pink","purple","red","rust","sage","silver","teal","white","yellow"],
  primary_material: ["brass_metal","ceramic","dark_wood","faux_leather","foam","glass","iron","leather","light_wood","linen","manufactured_wood","metal","microfiber","natural_fiber","plastic","stone","synthetics","velvet","wood"],
  secondary_material: ["brass_metal","ceramic","dark_wood","faux_leather","foam","glass","iron","leather","light_wood","linen","manufactured_wood","metal","microfiber","natural_fiber","plastic","stone","synthetics","velvet","wood"],
  style: ["bohemian","coastal","farmhouse","industrial","mid-century modern","modern","rustic","traditional"],
  shape: ["hexagon","irregular","l-shaped","oval","rectangular","round","runner","square","u-shaped"],
  assembly: ["full","none","partial"],
};

const MODES = [
  { key: "classifier", label: "Classifier", icon: "⚡", desc: "~50ms" },
  { key: "hybrid", label: "Hybrid", icon: "⚡+🔄", desc: "50ms + VLM" },
  { key: "vlm", label: "VLM Only", icon: "🔄", desc: "~2-5s" },
];

// ════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════

const fmt = (s) => s ? s.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : "—";

function ConfBadge({ conf, source }) {
  if (conf === "HV") return <span style={styles.badgeHV}>✓ Human</span>;
  if (conf === null || conf === undefined) {
    if (source === "vlm") return <span style={styles.badgeVLM}>VLM</span>;
    return <span style={styles.badgeNone}>—</span>;
  }
  const pct = `${Math.round(conf * 100)}%`;
  if (conf >= 0.7) return <span style={styles.badgeHigh}>{pct}</span>;
  if (conf >= 0.4) return <span style={styles.badgeMed}>{pct}</span>;
  return <span style={styles.badgeLow}>{pct}</span>;
}

function fileToBase64(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.readAsDataURL(file);
  });
}

// ════════════════════════════════════════════════════════════
// MAIN APP
// ════════════════════════════════════════════════════════════

export default function App() {
  const [mode, setMode] = useState("classifier");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [result, setResult] = useState(null);
  const [editedAttrs, setEditedAttrs] = useState({});
  const [vocab, setVocab] = useState(DEFAULT_VOCAB);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [form, setForm] = useState({
    product_name: "Modern Walnut Wood Dining Table with Metal Legs",
    product_class: "Dining Tables",
    description: "Solid walnut table top with sleek black metal hairpin legs. Mid-century modern style. Assembly required.",
    confidence_threshold: 0.5,
  });
  const fileRef = useRef(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => { setMounted(true); }, []);

  // Fetch vocab on mount
  useEffect(() => {
    fetch(`${API_BASE}/config`).then(r => r.json()).then(d => {
      if (d.vocab) setVocab(d.vocab);
    }).catch(() => {});
  }, []);

  const handleImage = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImageFile(f);
    setImagePreview(URL.createObjectURL(f));
  };

  const dropImage = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) {
      setImageFile(f);
      setImagePreview(URL.createObjectURL(f));
    }
  };

  const predict = async () => {
    setLoading(true);
    setResult(null);
    setEditedAttrs({});

    const payload = { ...form };
    if (imageFile) {
      payload.image_base64 = await fileToBase64(imageFile);
    }

    try {
      if (mode === "hybrid") {
        setLoadingStep("classifier");
        // Use streaming endpoint
        const res = await fetch(`${API_BASE}/predict/hybrid/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") break;
              try {
                const parsed = JSON.parse(data);
                if (parsed.step === "classifier") {
                  setResult(parsed);
                  setLoadingStep("vlm");
                } else if (parsed.step === "vlm_complete") {
                  setResult(parsed);
                  setLoadingStep("");
                }
              } catch {}
            }
          }
        }
      } else {
        setLoadingStep(mode);
        const endpoint = mode === "vlm" ? "vlm" : "classifier";
        const res = await fetch(`${API_BASE}/predict/${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        setResult(data);
        setLoadingStep("");
      }
    } catch (err) {
      console.error(err);
      setResult({ error: err.message });
    }
    setLoading(false);
    setLoadingStep("");
  };

  const updateAttr = (attr, value) => {
    setEditedAttrs(prev => ({ ...prev, [attr]: value }));
  };

  // Build export JSON
  const exportJSON = () => {
    if (!result) return {};
    const out = {};
    const tax = result.taxonomy || {};
    Object.keys(tax).sort().forEach(lk => { out[lk] = tax[lk]?.value; });
    if (result.product_class?.value) out.product_class = result.product_class.value;
    ATTR_ORDER.forEach(attr => {
      if (editedAttrs[attr]) {
        out[attr] = { value: editedAttrs[attr], confidence: "HV" };
      } else {
        const info = result.attributes?.[attr];
        if (info?.value) out[attr] = info;
      }
    });
    return out;
  };

  return (
    <div style={{
      ...styles.root,
      opacity: mounted ? 1 : 0,
      transition: "opacity 0.6s ease"
    }}>
      {/* ── HEADER ── */}
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div>
            <h1 style={styles.logo}>
              <span style={styles.logoIcon}>◆</span>
              Product Intelligence
            </h1>
            <p style={styles.subtitle}>Multi-Tower Classifier · VLM Fallback · Search Relevance</p>
          </div>
          <div style={styles.modeSelector}>
            {MODES.map(m => (
              <button
                key={m.key}
                onClick={() => setMode(m.key)}
                style={{
                  ...styles.modeBtn,
                  ...(mode === m.key ? styles.modeBtnActive : {}),
                }}
              >
                <span style={styles.modeIcon}>{m.icon}</span>
                <span style={styles.modeLabel}>{m.label}</span>
                <span style={styles.modeDesc}>{m.desc}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT ── */}
      <main style={styles.main}>
        <div style={styles.grid}>

          {/* ── LEFT: INPUT ── */}
          <section style={styles.inputSection}>
            <div style={styles.sectionHeader}>
              <span style={styles.sectionIcon}>📦</span>
              <h2 style={styles.sectionTitle}>Product Input</h2>
            </div>

            {/* Image Upload */}
            <div
              style={{
                ...styles.dropzone,
                ...(imagePreview ? styles.dropzoneHasImage : {}),
              }}
              onClick={() => fileRef.current?.click()}
              onDrop={dropImage}
              onDragOver={e => e.preventDefault()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Product" style={styles.previewImg} />
              ) : (
                <div style={styles.dropzoneInner}>
                  <span style={styles.dropzoneIcon}>🖼</span>
                  <p style={styles.dropzoneText}>Drop image or click to upload</p>
                  <p style={styles.dropzoneHint}>JPG, PNG, WebP</p>
                </div>
              )}
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                onChange={handleImage}
                style={{ display: "none" }}
              />
            </div>

            {/* Text Inputs */}
            <div style={styles.inputGroup}>
              <label style={styles.label}>Product Name</label>
              <input
                style={styles.input}
                value={form.product_name}
                onChange={e => setForm(f => ({...f, product_name: e.target.value}))}
                placeholder="e.g. Modern Walnut Dining Table"
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Category</label>
              <input
                style={styles.input}
                value={form.product_class}
                onChange={e => setForm(f => ({...f, product_class: e.target.value}))}
                placeholder="e.g. Dining Tables"
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Description</label>
              <textarea
                style={styles.textarea}
                value={form.description}
                onChange={e => setForm(f => ({...f, description: e.target.value}))}
                rows={3}
                placeholder="Product description..."
              />
            </div>

            {mode !== "vlm" && (
              <div style={styles.inputGroup}>
                <label style={styles.label}>
                  Confidence Threshold: {form.confidence_threshold}
                </label>
                <input
                  type="range"
                  min="0.1" max="0.9" step="0.05"
                  value={form.confidence_threshold}
                  onChange={e => setForm(f => ({...f, confidence_threshold: parseFloat(e.target.value)}))}
                  style={styles.slider}
                />
              </div>
            )}

            <button
              style={{
                ...styles.predictBtn,
                opacity: loading ? 0.7 : 1,
              }}
              onClick={predict}
              disabled={loading}
            >
              {loading ? (
                <span style={styles.btnLoading}>
                  <span style={styles.spinner} />
                  {loadingStep === "classifier" ? "Running Classifier..." :
                   loadingStep === "vlm" ? "Running VLM..." : "Processing..."}
                </span>
              ) : (
                "Extract Attributes"
              )}
            </button>
          </section>

          {/* ── RIGHT: RESULTS ── */}
          <section style={styles.resultSection}>
            <div style={styles.sectionHeader}>
              <span style={styles.sectionIcon}>📊</span>
              <h2 style={styles.sectionTitle}>Results</h2>
              {result?.latency_ms && (
                <span style={styles.latencyBadge}>
                  ⚡ {result.latency_ms}ms
                  {result.vlm_ms ? ` + 🔄 ${result.vlm_ms}ms` : ""}
                </span>
              )}
            </div>

            {!result && !loading && (
              <div style={styles.emptyState}>
                <span style={styles.emptyIcon}>◇</span>
                <p style={styles.emptyText}>Upload a product and click Extract</p>
              </div>
            )}

            {loading && !result && (
              <div style={styles.emptyState}>
                <span style={{...styles.spinner, width: 28, height: 28}} />
                <p style={styles.emptyText}>Analyzing product...</p>
              </div>
            )}

            {result?.error && (
              <div style={styles.errorBox}>{result.error}</div>
            )}

            {result && !result.error && (
              <div style={{
                opacity: 1,
                animation: "fadeSlideIn 0.4s ease",
              }}>
                {/* Taxonomy Table */}
                {result.taxonomy && Object.keys(result.taxonomy).length > 0 && (
                  <TaxonomyTable taxonomy={result.taxonomy} />
                )}

                {/* Product Class */}
                {result.product_class?.value && (
                  <ProductClassCard pc={result.product_class} />
                )}

                {/* Attributes Table */}
                {result.attributes && (
                  <AttributesTable
                    attributes={result.attributes}
                    sources={result.sources || {}}
                    vocab={vocab}
                    editedAttrs={editedAttrs}
                    onUpdate={updateAttr}
                    vlmAttrs={result.vlm_attrs || []}
                    vlmLoading={loadingStep === "vlm"}
                  />
                )}

                {/* Status */}
                {result.vlm_attrs?.length > 0 && loadingStep === "vlm" && (
                  <div style={styles.vlmLoadingBanner}>
                    <span style={styles.spinnerSmall} />
                    Running VLM for: {result.vlm_attrs.map(a => fmt(a)).join(", ")}
                  </div>
                )}

                {result.mode === "hybrid" && !loading && (
                  <div style={styles.successBanner}>
                    ⚡ Classifier ({result.cls_ms?.toFixed(0)}ms) + 🔄 VLM ({result.vlm_ms?.toFixed(0)}ms)
                  </div>
                )}

                {/* Export JSON */}
                <details style={styles.exportDetails}>
                  <summary style={styles.exportSummary}>Export JSON</summary>
                  <pre style={styles.exportPre}>
                    {JSON.stringify(exportJSON(), null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </section>
        </div>
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0c0f14; }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        select:focus, input:focus, textarea:focus {
          outline: none;
          border-color: #6366f1 !important;
          box-shadow: 0 0 0 2px rgba(99,102,241,0.15);
        }
      `}</style>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// TAXONOMY TABLE
// ════════════════════════════════════════════════════════════

function TaxonomyTable({ taxonomy }) {
  const levels = Object.entries(taxonomy).sort(([a], [b]) => a.localeCompare(b));
  return (
    <div style={styles.tableWrap}>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={{...styles.th, width: 50}}>Level</th>
            <th style={styles.th}>Category</th>
            <th style={{...styles.th, textAlign: "right", width: 90}}>Conf.</th>
          </tr>
        </thead>
        <tbody>
          {levels.map(([lk, info], idx) => (
            <tr key={lk} style={{ background: idx % 2 === 0 ? "#14181f" : "#111419" }}>
              <td style={styles.tdLevel}>{lk.replace("level_", "L")}</td>
              <td style={styles.td}>{fmt(info.value)}</td>
              <td style={{...styles.td, textAlign: "right"}}>
                <ConfBadge conf={info.confidence} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// PRODUCT CLASS
// ════════════════════════════════════════════════════════════

function ProductClassCard({ pc }) {
  return (
    <div style={styles.pcCard}>
      <div style={styles.pcLabel}>Product Class</div>
      <div style={styles.pcRow}>
        <span style={styles.pcValue}>{fmt(pc.value)}</span>
        <ConfBadge conf={pc.confidence} />
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// ATTRIBUTES TABLE
// ════════════════════════════════════════════════════════════

function AttributesTable({ attributes, sources, vocab, editedAttrs, onUpdate, vlmAttrs, vlmLoading }) {
  return (
    <div style={styles.tableWrap}>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>Attribute</th>
            <th style={styles.th}>Value</th>
            <th style={{...styles.th, textAlign: "right", width: 90}}>Conf.</th>
          </tr>
        </thead>
        <tbody>
          {ATTR_ORDER.map((attr, idx) => {
            const info = attributes[attr];
            if (!info?.value) return null;

            const original = info.value?.toLowerCase?.().replace(/ /g, "_") || info.value;
            const edited = editedAttrs[attr];
            const current = edited || original;
            const isVlmLoading = vlmLoading && vlmAttrs.includes(attr);
            const source = sources[attr];
            const isEdited = edited && edited !== original;

            const options = vocab[attr] || [];
            const allOptions = options.includes(original) ? options : [original, ...options];

            return (
              <tr key={attr} style={{ background: idx % 2 === 0 ? "#14181f" : "#111419" }}>
                <td style={styles.tdAttrName}>{ATTR_LABELS[attr]}</td>
                <td style={styles.td}>
                  {isVlmLoading ? (
                    <div style={styles.loadingValue}>
                      {fmt(original)} <span style={styles.spinnerSmall} />
                    </div>
                  ) : (
                    <select
                      value={current}
                      onChange={e => onUpdate(attr, e.target.value)}
                      style={styles.select}
                    >
                      {allOptions.map(opt => (
                        <option key={opt} value={opt}>{fmt(opt)}</option>
                      ))}
                    </select>
                  )}
                </td>
                <td style={{...styles.td, textAlign: "right"}}>
                  {isVlmLoading ? (
                    <span style={styles.badgeLoading}>Loading</span>
                  ) : isEdited ? (
                    <span style={styles.badgeHV}>✓ Human</span>
                  ) : (
                    <ConfBadge conf={info.confidence} source={source} />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ════════════════════════════════════════════════════════════
// STYLES
// ════════════════════════════════════════════════════════════

const styles = {
  root: {
    minHeight: "100vh",
    background: "#0c0f14",
    color: "#e2e8f0",
    fontFamily: "'DM Sans', sans-serif",
  },

  // Header
  header: {
    borderBottom: "1px solid #1e2330",
    background: "linear-gradient(180deg, #10141c 0%, #0c0f14 100%)",
    padding: "20px 0",
  },
  headerInner: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "0 32px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 16,
  },
  logo: {
    fontSize: 22,
    fontWeight: 700,
    color: "#f1f5f9",
    letterSpacing: "-0.5px",
  },
  logoIcon: {
    color: "#818cf8",
    marginRight: 10,
    fontSize: 18,
  },
  subtitle: {
    fontSize: 12,
    color: "#64748b",
    marginTop: 4,
    letterSpacing: "0.5px",
    textTransform: "uppercase",
  },
  modeSelector: {
    display: "flex",
    gap: 6,
    background: "#111419",
    borderRadius: 10,
    padding: 4,
    border: "1px solid #1e2330",
  },
  modeBtn: {
    background: "transparent",
    border: "none",
    borderRadius: 8,
    padding: "8px 16px",
    cursor: "pointer",
    color: "#94a3b8",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 2,
    transition: "all 0.2s",
    minWidth: 90,
  },
  modeBtnActive: {
    background: "#1e293b",
    color: "#f1f5f9",
    boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
  },
  modeIcon: { fontSize: 14 },
  modeLabel: { fontSize: 13, fontWeight: 600 },
  modeDesc: { fontSize: 10, opacity: 0.6 },

  // Main
  main: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "24px 32px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1.2fr",
    gap: 28,
    alignItems: "start",
  },

  // Sections
  sectionHeader: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    marginBottom: 20,
  },
  sectionIcon: { fontSize: 18 },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 600,
    color: "#f1f5f9",
    flex: 1,
  },
  latencyBadge: {
    fontSize: 11,
    color: "#94a3b8",
    background: "#1e293b",
    padding: "4px 10px",
    borderRadius: 20,
    fontFamily: "'JetBrains Mono', monospace",
  },
  inputSection: {
    background: "#111419",
    borderRadius: 14,
    padding: 24,
    border: "1px solid #1e2330",
  },
  resultSection: {
    background: "#111419",
    borderRadius: 14,
    padding: 24,
    border: "1px solid #1e2330",
    minHeight: 400,
  },

  // Dropzone
  dropzone: {
    border: "2px dashed #2d3348",
    borderRadius: 12,
    padding: 24,
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.2s",
    marginBottom: 16,
    background: "#0c0f14",
    overflow: "hidden",
  },
  dropzoneHasImage: {
    padding: 0,
    border: "2px solid #2d3348",
  },
  dropzoneInner: { padding: "20px 0" },
  dropzoneIcon: { fontSize: 32, display: "block", marginBottom: 8 },
  dropzoneText: { color: "#94a3b8", fontSize: 13, fontWeight: 500 },
  dropzoneHint: { color: "#475569", fontSize: 11, marginTop: 4 },
  previewImg: {
    width: "100%",
    maxHeight: 280,
    objectFit: "cover",
    display: "block",
  },

  // Inputs
  inputGroup: { marginBottom: 14 },
  label: {
    display: "block",
    fontSize: 11,
    fontWeight: 600,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 6,
  },
  input: {
    width: "100%",
    background: "#0c0f14",
    border: "1px solid #2d3348",
    borderRadius: 8,
    padding: "10px 14px",
    color: "#e2e8f0",
    fontSize: 14,
    fontFamily: "'DM Sans', sans-serif",
    transition: "border-color 0.2s",
  },
  textarea: {
    width: "100%",
    background: "#0c0f14",
    border: "1px solid #2d3348",
    borderRadius: 8,
    padding: "10px 14px",
    color: "#e2e8f0",
    fontSize: 14,
    fontFamily: "'DM Sans', sans-serif",
    resize: "vertical",
    transition: "border-color 0.2s",
  },
  slider: {
    width: "100%",
    accentColor: "#818cf8",
  },

  // Predict Button
  predictBtn: {
    width: "100%",
    background: "linear-gradient(135deg, #6366f1 0%, #818cf8 100%)",
    color: "#fff",
    border: "none",
    borderRadius: 10,
    padding: "14px 0",
    fontSize: 15,
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s",
    fontFamily: "'DM Sans', sans-serif",
    marginTop: 8,
    letterSpacing: "0.3px",
  },
  btnLoading: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  spinner: {
    display: "inline-block",
    width: 18,
    height: 18,
    border: "2px solid rgba(255,255,255,0.3)",
    borderTopColor: "#fff",
    borderRadius: "50%",
    animation: "spin 0.6s linear infinite",
  },
  spinnerSmall: {
    display: "inline-block",
    width: 14,
    height: 14,
    border: "2px solid rgba(129,140,248,0.3)",
    borderTopColor: "#818cf8",
    borderRadius: "50%",
    animation: "spin 0.6s linear infinite",
    marginLeft: 6,
  },

  // Empty State
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "60px 0",
    gap: 12,
  },
  emptyIcon: { fontSize: 36, color: "#2d3348" },
  emptyText: { color: "#475569", fontSize: 14 },
  errorBox: {
    background: "#1c1012",
    border: "1px solid #5c1d1d",
    color: "#fca5a5",
    borderRadius: 10,
    padding: "12px 16px",
    fontSize: 13,
  },

  // Tables
  tableWrap: {
    borderRadius: 10,
    overflow: "hidden",
    border: "1px solid #1e2330",
    marginBottom: 12,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 13,
  },
  th: {
    background: "#1a2035",
    color: "#94a3b8",
    padding: "10px 14px",
    textAlign: "left",
    fontWeight: 600,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: "0.6px",
    borderBottom: "1px solid #1e2330",
  },
  td: {
    padding: "9px 14px",
    borderBottom: "1px solid #1a1e28",
    color: "#e2e8f0",
    fontSize: 13,
  },
  tdLevel: {
    padding: "9px 14px",
    borderBottom: "1px solid #1a1e28",
    color: "#64748b",
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
  },
  tdAttrName: {
    padding: "9px 14px",
    borderBottom: "1px solid #1a1e28",
    color: "#cbd5e1",
    fontSize: 13,
    fontWeight: 500,
  },

  // Select
  select: {
    width: "100%",
    background: "#0c0f14",
    border: "1px solid #2d3348",
    borderRadius: 6,
    padding: "6px 10px",
    color: "#e2e8f0",
    fontSize: 13,
    fontFamily: "'DM Sans', sans-serif",
    cursor: "pointer",
    transition: "border-color 0.2s",
    appearance: "auto",
  },
  loadingValue: {
    display: "flex",
    alignItems: "center",
    color: "#64748b",
    fontSize: 13,
  },

  // Product Class Card
  pcCard: {
    background: "#14181f",
    border: "1px solid #1e2330",
    borderRadius: 10,
    padding: "12px 16px",
    marginBottom: 12,
  },
  pcLabel: {
    fontSize: 10,
    fontWeight: 600,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.8px",
    marginBottom: 6,
  },
  pcRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  pcValue: {
    fontSize: 15,
    fontWeight: 600,
    color: "#f1f5f9",
  },

  // Badges
  badgeHigh: {
    color: "#34d399",
    fontWeight: 600,
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
  },
  badgeMed: {
    color: "#fbbf24",
    fontWeight: 600,
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
  },
  badgeLow: {
    color: "#f87171",
    fontWeight: 600,
    fontSize: 12,
    fontFamily: "'JetBrains Mono', monospace",
  },
  badgeVLM: {
    color: "#a78bfa",
    fontWeight: 600,
    fontSize: 11,
    padding: "2px 8px",
    background: "rgba(167,139,250,0.1)",
    borderRadius: 4,
  },
  badgeHV: {
    color: "#34d399",
    fontWeight: 600,
    fontSize: 11,
    padding: "2px 8px",
    background: "rgba(52,211,153,0.1)",
    borderRadius: 4,
  },
  badgeLoading: {
    color: "#a78bfa",
    fontWeight: 600,
    fontSize: 11,
    padding: "2px 8px",
    background: "rgba(167,139,250,0.1)",
    borderRadius: 4,
  },
  badgeNone: {
    color: "#475569",
    fontSize: 12,
  },

  // Banners
  vlmLoadingBanner: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    background: "rgba(129,140,248,0.08)",
    border: "1px solid rgba(129,140,248,0.15)",
    borderRadius: 8,
    padding: "10px 14px",
    fontSize: 12,
    color: "#a5b4fc",
    marginBottom: 12,
  },
  successBanner: {
    background: "rgba(52,211,153,0.08)",
    border: "1px solid rgba(52,211,153,0.15)",
    borderRadius: 8,
    padding: "10px 14px",
    fontSize: 12,
    color: "#6ee7b7",
    marginBottom: 12,
    fontFamily: "'JetBrains Mono', monospace",
  },

  // Export
  exportDetails: {
    marginTop: 8,
  },
  exportSummary: {
    fontSize: 12,
    color: "#64748b",
    cursor: "pointer",
    padding: "8px 0",
    fontWeight: 500,
  },
  exportPre: {
    background: "#0c0f14",
    border: "1px solid #1e2330",
    borderRadius: 8,
    padding: 14,
    fontSize: 11,
    color: "#94a3b8",
    overflow: "auto",
    maxHeight: 300,
    fontFamily: "'JetBrains Mono', monospace",
  },
};