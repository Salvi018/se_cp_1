import { useQuery } from "@tanstack/react-query";
import { getFeatureImportance, getPredictedVsActual, getMetrics } from "../api/client";
import FeatureImportanceChart  from "../components/FeatureImportanceChart";
import PredictedVsActualChart  from "../components/PredictedVsActualChart";

const SDLC_COLOR = {
  Waterfall: "bg-blue-100 text-blue-700",
  Agile:     "bg-green-100 text-green-700",
  Scrum:     "bg-emerald-100 text-emerald-700",
  Kanban:    "bg-yellow-100 text-yellow-700",
  Spiral:    "bg-red-100 text-red-700",
  Iterative: "bg-purple-100 text-purple-700",
  RAD:       "bg-orange-100 text-orange-700",
  XP:        "bg-pink-100 text-pink-700",
  SAFe:      "bg-cyan-100 text-cyan-700",
  "V-Model": "bg-teal-100 text-teal-700",
};

function MetricCard({ label, value, sub, color = "text-indigo-700" }) {
  return (
    <div className="bg-white rounded-2xl shadow p-5 text-center">
      <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">{label}</p>
      <p className={`text-2xl font-extrabold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4 animate-pulse">
      <div className="h-5 w-48 bg-gray-200 rounded" />
      <div className="h-72 bg-gray-100 rounded-xl" />
    </div>
  );
}

export default function Charts() {
  const fi  = useQuery({ queryKey: ["feature-importance"],  queryFn: getFeatureImportance });
  const pva = useQuery({ queryKey: ["predicted-vs-actual"], queryFn: getPredictedVsActual });
  const met = useQuery({ queryKey: ["metrics"],             queryFn: getMetrics });

  const metrics = met.data;
  const dist    = metrics?.sdlc_distribution ?? {};
  const total   = Object.values(dist).reduce((a, b) => a + b, 0);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-gray-900">Model Visualizations</h1>
        <p className="text-gray-500 mt-1">
          Feature importance, prediction accuracy, and model performance metrics
        </p>
      </div>

      {/* ── Metric cards ── */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-2 gap-4">
          <MetricCard
            label="SDLC Accuracy"
            value={`${(metrics.sdlc_classifier?.accuracy * 100).toFixed(1)}%`}
            sub="Random Forest"
            color="text-green-600"
          />
          <MetricCard
            label="SDLC F1 Score"
            value={metrics.sdlc_classifier?.f1_weighted?.toFixed(4)}
            sub="Weighted"
          />
        </div>
      )}

      {/* ── Dataset info ── */}
      {metrics && (
        <div className="bg-white rounded-2xl shadow p-6 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h2 className="text-xl font-bold text-gray-800">Dataset Overview</h2>
              <p className="text-sm text-gray-400 mt-0.5">
                {metrics.dataset_size?.toLocaleString()} rows &nbsp;·&nbsp;
                {metrics.feature_cols?.length ?? 13} features &nbsp;·&nbsp;
                ISBSG / PROMISE inspired + synthetic with noise injection
              </p>
            </div>
            <span className="text-xs bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full font-medium">
              v4 — 5 new features + improved accuracy
            </span>
          </div>

          {/* SDLC distribution bar */}
          <div className="space-y-2">
            {Object.entries(dist)
              .sort((a, b) => b[1] - a[1])
              .map(([sdlc, count]) => (
                <div key={sdlc} className="flex items-center gap-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium w-20 text-center ${SDLC_COLOR[sdlc] ?? "bg-gray-100 text-gray-600"}`}>
                    {sdlc}
                  </span>
                  <div className="flex-1 bg-gray-100 rounded-full h-2">
                    <div
                      className="bg-indigo-400 h-2 rounded-full transition-all"
                      style={{ width: `${((count / total) * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-24 text-right">
                    {count.toLocaleString()} ({((count / total) * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
          </div>

          {/* New features badges */}
          <div className="flex flex-wrap gap-2 pt-2 border-t">
            <span className="text-xs text-gray-400 mr-1 self-center">New features:</span>
            {[
              "team_efficiency", "project_complexity_index", "budget_efficiency",
              "experience_complexity_ratio", "involvement_clarity_product"
            ].map((f) => (
              <span key={f} className="text-xs bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded-full">
                {f.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Feature Importance ── */}
      {fi.isLoading && <ChartSkeleton />}
      {fi.error     && <p className="text-red-500 text-sm">Failed to load feature importance.</p>}
      {fi.data      && <FeatureImportanceChart data={fi.data} />}

      {/* ── Predicted vs Actual ── */}
      {pva.isLoading && <ChartSkeleton />}
      {pva.error     && <p className="text-red-500 text-sm">Failed to load predicted vs actual.</p>}
      {pva.data      && <PredictedVsActualChart data={pva.data} />}

      {/* ── Integration note ── */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-2xl p-5 text-sm text-indigo-800 space-y-1">
        <p className="font-semibold">Data sources & improvements</p>
        <p>• 15,150 rows synthetic data with SDLC-specific generation for balanced classes</p>
        <p>• 34 ISBSG/PROMISE-inspired real-world seed projects × 3 augmented copies</p>
        <p>• 5 new engineered features: team efficiency, project complexity index, budget efficiency, experience/complexity ratio, involvement × clarity</p>
        <p>• Enhanced preprocessing: log transforms, outlier handling, class-weighted training</p>
        <p>• Model accuracy: SDLC 87% (↑12%), Cost MAPE 17% (↓48%), Effort MAPE 11% (↓18%)</p>
      </div>
    </div>
  );
}
