import { useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

const fmtCost   = (v) => `$${(v / 1000).toFixed(0)}k`;
const fmtEffort = (v) => `${v.toFixed(0)}pm`;

const CustomTooltip = ({ active, payload, mode }) => {
  if (!active || !payload?.length) return null;
  const { actual, predicted } = payload[0].payload;
  const fmt = mode === "cost" ? (v) => `$${Number(v).toLocaleString()}` : (v) => `${v} person-months`;
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 shadow text-xs space-y-1">
      <p className="text-gray-500">Actual:    <span className="font-semibold text-gray-800">{fmt(actual)}</span></p>
      <p className="text-gray-500">Predicted: <span className="font-semibold text-indigo-600">{fmt(predicted)}</span></p>
      <p className="text-gray-400">Error: {Math.abs(((predicted - actual) / actual) * 100).toFixed(1)}%</p>
    </div>
  );
};

export default function PredictedVsActualChart({ data }) {
  const [mode, setMode] = useState("cost");
  if (!data) return null;

  const points = data[mode]?.points ?? [];
  const max    = data[mode]?.max    ?? 0;
  const isCost = mode === "cost";

  const perfectLine = [
    { actual: 0, predicted: 0 },
    { actual: max, predicted: max },
  ];

  return (
    <div className="bg-white rounded-2xl shadow p-6">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div>
          <h2 className="text-xl font-bold text-gray-800">Predicted vs Actual</h2>
          <p className="text-sm text-gray-400 mt-0.5">
            Each dot = one test sample. Perfect model = all dots on the line.
          </p>
        </div>
        <div className="flex gap-2">
          {["cost", "effort"].map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${
                mode === m
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {m === "cost" ? "Cost (USD)" : "Effort (pm)"}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="actual"
            name="Actual"
            tickFormatter={isCost ? fmtCost : fmtEffort}
            tick={{ fontSize: 11 }}
            label={{ value: "Actual", position: "insideBottom", offset: -5, fontSize: 12, fill: "#9ca3af" }}
          />
          <YAxis
            type="number"
            dataKey="predicted"
            name="Predicted"
            tickFormatter={isCost ? fmtCost : fmtEffort}
            tick={{ fontSize: 11 }}
            label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 12, fill: "#9ca3af" }}
          />
          <Tooltip content={<CustomTooltip mode={mode} />} />

          {/* Perfect prediction reference line */}
          <Scatter
            name="Perfect prediction"
            data={perfectLine}
            line={{ stroke: "#e5e7eb", strokeDasharray: "5 5", strokeWidth: 1.5 }}
            shape={() => null}
            legendType="line"
          />

          {/* Actual scatter points */}
          <Scatter
            name={isCost ? "Cost prediction" : "Effort prediction"}
            data={points}
            fill="#4f46e5"
            fillOpacity={0.6}
            r={3}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
        </ScatterChart>
      </ResponsiveContainer>

      <div className="mt-3 flex gap-6 justify-center text-xs text-gray-400">
        <span>● Prediction point</span>
        <span>- - Perfect prediction line</span>
      </div>
    </div>
  );
}
