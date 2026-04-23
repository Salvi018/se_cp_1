import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts";

const COLORS = [
  "#4f46e5", "#6366f1", "#818cf8", "#a5b4fc",
  "#c7d2fe", "#ddd6fe", "#ede9fe", "#f5f3ff",
  "#e0e7ff", "#eef2ff",
];

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 shadow text-sm">
      <p className="font-semibold text-gray-800">{d.feature}</p>
      <p className="text-indigo-600">{d.pct}% importance</p>
    </div>
  );
};

export default function FeatureImportanceChart({ data }) {
  if (!data?.length) return null;

  return (
    <div className="bg-white rounded-2xl shadow p-6">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-gray-800">Feature Importance</h2>
        <p className="text-sm text-gray-400 mt-0.5">
          Which inputs drive the SDLC recommendation most
        </p>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 60, left: 10, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={130}
            tick={{ fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
            <LabelList
              dataKey="pct"
              position="right"
              formatter={(v) => `${v}%`}
              style={{ fontSize: 11, fill: "#6b7280" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
