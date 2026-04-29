import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";
import { fmtINR, fmtINRShort } from "../utils/currency";

const COLORS = ["#a5b4fc", "#4f46e5", "#a5b4fc"];

export default function CostChart({ result, budget }) {
  if (!result) return null;

  const costLower = result.cost_range?.lower ?? result.cost_lower ?? 0;
  const costUpper = result.cost_range?.upper ?? result.cost_upper ?? 0;

  const data = [
    { name: "Lower",     value: costLower },
    { name: "Estimated", value: result.estimated_cost_usd },
    { name: "Upper",     value: costUpper },
  ];

  const overBudget = budget && result.estimated_cost_usd > budget;

  return (
    <div className="bg-white rounded-2xl shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800">Cost Breakdown</h2>
        {overBudget && (
          <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full font-medium">
            ⚠ Over budget
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tickFormatter={fmtINRShort} tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(v) => [fmtINR(v), "Cost"]}
            contentStyle={{ borderRadius: "8px", border: "1px solid #e5e7eb" }}
          />
          {budget && (
            <ReferenceLine
              y={budget}
              stroke="#ef4444"
              strokeDasharray="4 4"
              label={{ value: "Budget", fill: "#ef4444", fontSize: 11, position: "insideTopRight" }}
            />
          )}
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {data.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 mt-2 text-center">
        Red dashed line = your budget
      </p>
    </div>
  );
}
