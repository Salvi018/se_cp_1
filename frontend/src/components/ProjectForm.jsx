import { useState } from "react";

const DEFAULTS = {
  team_size: 10,
  duration_months: 12,
  budget_usd: 100000,
  requirements_clarity: 3,
  client_involvement: 3,
  tech_complexity: 3,
  risk_level: "medium",
  project_type: "web",
};

export default function ProjectForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS);

  const set = (k) => (e) => setForm((p) => ({ ...p, [k]: e.target.value }));

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(form);
  };

  const slider = (key, label, min = 1, max = 5) => (
    <div key={key}>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}: <span className="font-bold text-indigo-600">{form[key]}</span>
      </label>
      <input
        type="range" min={min} max={max} value={form[key]}
        onChange={set(key)}
        className="w-full accent-indigo-600"
      />
      <div className="flex justify-between text-xs text-gray-400">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  );

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow p-6 space-y-5">
      <h2 className="text-xl font-bold text-gray-800">Project Parameters</h2>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Team Size</label>
          <input type="number" min={1} max={100} value={form.team_size} onChange={set("team_size")}
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Duration (months)</label>
          <input type="number" min={1} max={60} value={form.duration_months} onChange={set("duration_months")}
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400" />
        </div>
        <div className="col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">Budget (USD)</label>
          <input type="number" min={1000} value={form.budget_usd} onChange={set("budget_usd")}
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400" />
        </div>
      </div>

      <div className="space-y-4">
        {slider("requirements_clarity", "Requirements Clarity")}
        {slider("client_involvement",   "Client Involvement")}
        {slider("tech_complexity",      "Technical Complexity")}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Risk Level</label>
          <select value={form.risk_level} onChange={set("risk_level")}
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400">
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Project Type</label>
          <select value={form.project_type} onChange={set("project_type")}
            className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400">
            <option value="web">Web</option>
            <option value="mobile">Mobile</option>
            <option value="embedded">Embedded</option>
            <option value="data">Data/ML</option>
          </select>
        </div>
      </div>

      <button type="submit" disabled={loading}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold py-2.5 rounded-xl transition">
        {loading ? "Analyzing..." : "Predict SDLC & Estimate Cost"}
      </button>
    </form>
  );
}
