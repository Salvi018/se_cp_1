const fmt = (n) => `$${Number(n).toLocaleString()}`;

const BADGE = {
  Waterfall: "bg-blue-100 text-blue-700",
  Agile:     "bg-green-100 text-green-700",
  Scrum:     "bg-emerald-100 text-emerald-700",
  Kanban:    "bg-yellow-100 text-yellow-700",
  Spiral:    "bg-red-100 text-red-700",
  Iterative: "bg-purple-100 text-purple-700",
};

export default function HistoryTable({ data, page, pages, onPage }) {
  if (!data?.length) return <p className="text-gray-400 text-center py-8">No predictions yet.</p>;

  return (
    <div className="bg-white rounded-2xl shadow overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
          <tr>
            {["Date", "Type", "Team", "Duration", "SDLC", "Confidence", "Est. Cost"].map(h => (
              <th key={h} className="px-4 py-3 text-left">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {data.map((p) => (
            <tr key={p.id} className="hover:bg-gray-50">
              <td className="px-4 py-3 text-gray-500">{new Date(p.created_at).toLocaleDateString()}</td>
              <td className="px-4 py-3 capitalize">{p.project_type}</td>
              <td className="px-4 py-3">{p.team_size}</td>
              <td className="px-4 py-3">{p.duration_months}m</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${BADGE[p.recommended_sdlc] || "bg-gray-100"}`}>
                  {p.recommended_sdlc}
                </span>
              </td>
              <td className="px-4 py-3">{(p.confidence * 100).toFixed(1)}%</td>
              <td className="px-4 py-3 font-medium">{fmt(p.estimated_cost_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {pages > 1 && (
        <div className="flex justify-center gap-2 p-4">
          {Array.from({ length: pages }, (_, i) => i + 1).map((p) => (
            <button key={p} onClick={() => onPage(p)}
              className={`px-3 py-1 rounded-lg text-sm ${p === page ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-600 hover:bg-gray-200"}`}>
              {p}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
