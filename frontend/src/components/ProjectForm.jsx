import { useState } from "react";

const USD_TO_INR = 83.5;

const DEFAULTS = {
  team_size:               10,
  duration_months:         12,
  budget_usd:              8350000,
  requirements_clarity:    3,
  client_involvement:      3,
  tech_complexity:         3,
  risk_level:              "medium",
  project_type:            "web",
  team_experience:         3,
  regulatory_compliance:   0,
  geographic_distribution: 1,
};

// Preset profiles — each produces a different SDLC
const PRESETS = [
  {
    label: "🏢 Enterprise",
    hint:  "SAFe",
    values: { team_size: 60, duration_months: 24, budget_usd: 91850000, requirements_clarity: 4, client_involvement: 3, tech_complexity: 4, risk_level: "medium", project_type: "web",      team_experience: 4, regulatory_compliance: 0, geographic_distribution: 3 },
  },
  {
    label: "🚀 Startup",
    hint:  "Scrum",
    values: { team_size: 6,  duration_months: 6,  budget_usd: 5010000,  requirements_clarity: 2, client_involvement: 5, tech_complexity: 3, risk_level: "medium", project_type: "mobile",   team_experience: 3, regulatory_compliance: 0, geographic_distribution: 2 },
  },
  {
    label: "🏥 Medical",
    hint:  "V-Model",
    values: { team_size: 20, duration_months: 30, budget_usd: 37575000, requirements_clarity: 5, client_involvement: 1, tech_complexity: 4, risk_level: "medium", project_type: "embedded", team_experience: 4, regulatory_compliance: 1, geographic_distribution: 1 },
  },
  {
    label: "⚡ Prototype",
    hint:  "RAD",
    values: { team_size: 6,  duration_months: 4,  budget_usd: 5010000,  requirements_clarity: 2, client_involvement: 4, tech_complexity: 2, risk_level: "low",    project_type: "web",      team_experience: 3, regulatory_compliance: 0, geographic_distribution: 1 },
  },
  {
    label: "🛡 Defence",
    hint:  "Spiral",
    values: { team_size: 20, duration_months: 30, budget_usd: 41750000, requirements_clarity: 3, client_involvement: 2, tech_complexity: 5, risk_level: "high",   project_type: "embedded", team_experience: 5, regulatory_compliance: 1, geographic_distribution: 1 },
  },
  {
    label: "🔬 R&D",
    hint:  "Iterative",
    values: { team_size: 8,  duration_months: 20, budget_usd: 10020000, requirements_clarity: 1, client_involvement: 1, tech_complexity: 3, risk_level: "medium", project_type: "data",     team_experience: 2, regulatory_compliance: 0, geographic_distribution: 2 },
  },
];

const SLIDER_LABELS = {
  requirements_clarity: { 1: "Vague", 2: "Unclear", 3: "Moderate", 4: "Clear", 5: "Frozen" },
  client_involvement:   { 1: "Absent", 2: "Rare", 3: "Occasional", 4: "Regular", 5: "Daily" },
  tech_complexity:      { 1: "Simple", 2: "Basic", 3: "Moderate", 4: "Complex", 5: "Very Complex" },
  team_experience:      { 1: "Junior", 2: "Beginner", 3: "Mid-level", 4: "Senior", 5: "Expert" },
};

export default function ProjectForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS);

  const set    = (k) => (e) => setForm((p) => ({ ...p, [k]: e.target.value }));
  const setNum = (k) => (e) => setForm((p) => ({ ...p, [k]: Number(e.target.value) }));

  const handleSubmit = (e) => {
    e.preventDefault();
    // Convert INR budget to USD for the API
    onSubmit({ ...form, budget_usd: Number(form.budget_usd) / USD_TO_INR });
  };

  const applyPreset = (preset) => setForm({ ...DEFAULTS, ...preset.values });

  const slider = (key, label, min = 1, max = 5) => {
    const hint = SLIDER_LABELS[key]?.[form[key]];
    return (
      <div key={key}>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}:&nbsp;
          <span className="font-bold text-indigo-600">{form[key]}</span>
          {hint && <span className="ml-1 text-xs text-gray-400">({hint})</span>}
        </label>
        <input
          type="range" min={min} max={max} value={form[key]}
          onChange={setNum(key)}
          className="w-full accent-indigo-600"
        />
        <div className="flex justify-between text-xs text-gray-400">
          <span>{min}</span><span>{max}</span>
        </div>
      </div>
    );
  };

  const inputCls  = "w-full border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400";

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-800">Project Parameters</h2>
        <button type="button" onClick={() => setForm(DEFAULTS)}
          className="text-xs text-gray-400 hover:text-indigo-600 transition">
          ↺ Reset
        </button>
      </div>

      {/* ── Presets ── */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Quick Presets</p>
        <div className="grid grid-cols-3 gap-2">
          {PRESETS.map((p) => (
            <button key={p.label} type="button" onClick={() => applyPreset(p)}
              className="py-1.5 px-2 rounded-lg text-xs font-medium border border-gray-200 hover:border-indigo-400 hover:bg-indigo-50 transition text-left">
              <span>{p.label}</span>
              <span className="block text-indigo-500 font-semibold">{p.hint}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Basic info ── */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Team Size</label>
          <input type="number" min={1} max={100} value={form.team_size} onChange={set("team_size")}
            className={inputCls} />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Duration (months)</label>
          <input type="number" min={1} max={60} value={form.duration_months} onChange={set("duration_months")}
            className={inputCls} />
        </div>
        <div className="col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">Budget (₹ INR)</label>
          <input type="number" min={1000} value={form.budget_usd} onChange={set("budget_usd")}
            className={inputCls} />
        </div>
      </div>

      {/* ── Sliders ── */}
      <div className="space-y-4">
        {slider("requirements_clarity", "Requirements Clarity")}
        {slider("client_involvement",   "Client Involvement")}
        {slider("tech_complexity",      "Technical Complexity")}
        {slider("team_experience",      "Team Experience")}
      </div>

      {/* ── Dropdowns ── */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Risk Level</label>
          <select value={form.risk_level} onChange={set("risk_level")} className={inputCls}>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Project Type</label>
          <select value={form.project_type} onChange={set("project_type")} className={inputCls}>
            <option value="web">Web</option>
            <option value="mobile">Mobile</option>
            <option value="embedded">Embedded</option>
            <option value="data">Data / ML</option>
          </select>
        </div>
      </div>

      {/* ── Advanced ── */}
      <div className="border-t pt-4 space-y-4">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Advanced Parameters</p>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Team Distribution</label>
          <div className="grid grid-cols-3 gap-2">
            {[
              { val: 1, label: "🏢 Co-located" },
              { val: 2, label: "🔀 Hybrid" },
              { val: 3, label: "🌍 Distributed" },
            ].map(({ val, label }) => (
              <button key={val} type="button"
                onClick={() => setForm((p) => ({ ...p, geographic_distribution: val }))}
                className={`py-2 px-2 rounded-lg text-xs font-medium border transition ${
                  form.geographic_distribution === val
                    ? "bg-indigo-600 text-white border-indigo-600"
                    : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300"
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-700">Regulatory Compliance</p>
            <p className="text-xs text-gray-400">HIPAA, ISO 27001, SOX, GDPR, etc.</p>
          </div>
          <button type="button"
            onClick={() => setForm((p) => ({ ...p, regulatory_compliance: p.regulatory_compliance === 1 ? 0 : 1 }))}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              form.regulatory_compliance === 1 ? "bg-indigo-600" : "bg-gray-200"
            }`}>
            <span className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform ${
              form.regulatory_compliance === 1 ? "translate-x-7" : "translate-x-1"
            }`} />
          </button>
        </div>
      </div>

      <button type="submit" disabled={loading}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold py-2.5 rounded-xl transition">
        {loading ? "Analyzing..." : "Predict SDLC & Estimate Cost"}
      </button>
    </form>
  );
}
