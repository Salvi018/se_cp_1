import { fmtINR } from "../utils/currency";

const SDLC_COLOR = {
  Waterfall: "bg-blue-100 text-blue-800",
  Agile:     "bg-green-100 text-green-800",
  Scrum:     "bg-emerald-100 text-emerald-800",
  Kanban:    "bg-yellow-100 text-yellow-800",
  Spiral:    "bg-red-100 text-red-800",
  Iterative: "bg-purple-100 text-purple-800",
  RAD:       "bg-orange-100 text-orange-800",
  XP:        "bg-pink-100 text-pink-800",
  SAFe:      "bg-cyan-100 text-cyan-800",
  "V-Model": "bg-teal-100 text-teal-800",
};


export default function ResultCard({ result }) {
  if (!result) return null;

  // Support both flat and nested cost_range shapes
  const costLower = result.cost_range?.lower ?? result.cost_lower ?? 0;
  const costUpper = result.cost_range?.upper ?? result.cost_upper ?? 0;
  const color     = SDLC_COLOR[result.recommended_sdlc] || "bg-gray-100 text-gray-800";

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4">
      <h2 className="text-xl font-bold text-gray-800">Recommendation</h2>

      {/* SDLC badge + confidence */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className={`text-2xl font-extrabold px-4 py-2 rounded-xl ${color}`}>
          {result.recommended_sdlc}
        </span>
        <div className="flex flex-col">
          <span className="text-xs text-gray-400">Confidence</span>
          <span className="font-bold text-gray-700">
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>
        {/* Confidence bar */}
        <div className="flex-1 min-w-[80px] bg-gray-100 rounded-full h-2">
          <div
            className="bg-indigo-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
          />
        </div>
      </div>

      {/* Alternatives */}
      {result.alternatives?.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-1 uppercase tracking-wide">Alternatives</p>
          <div className="flex gap-2 flex-wrap">
            {result.alternatives.map((a) => (
              <span
                key={a}
                className={`text-xs px-2 py-1 rounded-full ${SDLC_COLOR[a] || "bg-gray-100 text-gray-600"}`}
              >
                {a}
              </span>
            ))}
          </div>
        </div>
      )}

      <hr />

      {/* Cost + Effort */}
      <div className="grid grid-cols-2 gap-4 text-center">
        <div className="bg-indigo-50 rounded-xl p-3">
          <p className="text-xs text-gray-500 mb-1">Estimated Cost</p>
          <p className="text-lg font-bold text-indigo-700">{fmtINR(result.estimated_cost_usd)}</p>
          <p className="text-xs text-gray-400 mt-0.5">
            {fmtINR(costLower)} – {fmtINR(costUpper)}
          </p>
        </div>
        <div className="bg-indigo-50 rounded-xl p-3">
          <p className="text-xs text-gray-500 mb-1">Effort</p>
          <p className="text-lg font-bold text-indigo-700">
            {result.effort_person_months}
          </p>
          <p className="text-xs text-gray-400 mt-0.5">person-months</p>
        </div>
      </div>
    </div>
  );
}
