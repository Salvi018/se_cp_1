import { useQuery } from "@tanstack/react-query";
import { getFeatureImportance, getPredictedVsActual, getMetrics } from "../api/client";
import FeatureImportanceChart from "../components/FeatureImportanceChart";
import PredictedVsActualChart from "../components/PredictedVsActualChart";

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
  const fi  = useQuery({ queryKey: ["feature-importance"],   queryFn: getFeatureImportance });
  const pva = useQuery({ queryKey: ["predicted-vs-actual"],  queryFn: getPredictedVsActual });
  const met = useQuery({ queryKey: ["metrics"],              queryFn: getMetrics });

  const metrics = met.data;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-gray-900">Model Visualizations</h1>
        <p className="text-gray-500 mt-1">
          Feature importance, prediction accuracy, and model performance metrics
        </p>
      </div>

      {/* Metrics summary row */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
          <MetricCard
            label="Cost MAE"
            value={`$${(metrics.cost_estimator?.mae / 1000).toFixed(1)}k`}
            sub={`R² ${metrics.cost_estimator?.r2}`}
          />
          <MetricCard
            label="Effort MAE"
            value={`${metrics.effort_estimator?.mae} pm`}
            sub={`R² ${metrics.effort_estimator?.r2}`}
          />
        </div>
      )}

      {/* Feature Importance */}
      {fi.isLoading  && <ChartSkeleton />}
      {fi.error      && <p className="text-red-500 text-sm">Failed to load feature importance.</p>}
      {fi.data       && <FeatureImportanceChart data={fi.data} />}

      {/* Predicted vs Actual */}
      {pva.isLoading && <ChartSkeleton />}
      {pva.error     && <p className="text-red-500 text-sm">Failed to load predicted vs actual.</p>}
      {pva.data      && <PredictedVsActualChart data={pva.data} />}

      {/* Integration note */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-2xl p-5 text-sm text-indigo-800 space-y-1">
        <p className="font-semibold">Where these charts come from</p>
        <p>• <code className="bg-indigo-100 px-1 rounded">GET /api/charts/feature-importance</code> — sklearn RandomForest feature_importances_ array</p>
        <p>• <code className="bg-indigo-100 px-1 rounded">GET /api/charts/predicted-vs-actual</code> — model predictions on the held-out 20% test set</p>
        <p>• <code className="bg-indigo-100 px-1 rounded">GET /api/models/metrics</code> — MAE, RMSE, R², accuracy saved during training</p>
      </div>
    </div>
  );
}
