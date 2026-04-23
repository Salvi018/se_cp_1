import { useState } from "react";
import { usePrediction } from "../hooks/usePrediction";
import ProjectForm from "../components/ProjectForm";
import ResultCard  from "../components/ResultCard";
import CostChart   from "../components/CostChart";
import LoadingSkeleton from "../components/LoadingSkeleton";

export default function Home() {
  const { mutate, data, isPending, error, reset } = usePrediction();
  const [budget, setBudget] = useState(null);

  const handleSubmit = (formData) => {
    reset();
    setBudget(Number(formData.budget_usd));
    mutate(formData);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-extrabold text-gray-900">SDLC Model Selector</h1>
        <p className="text-gray-500 mt-1">ML-powered SDLC recommendation with cost &amp; effort estimation</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl px-4 py-3 text-sm flex items-start gap-2">
          <span className="mt-0.5">⚠</span>
          <span>{error.message}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ProjectForm onSubmit={handleSubmit} loading={isPending} />

        <div className="space-y-6">
          {isPending && <LoadingSkeleton />}
          {!isPending && data && (
            <>
              <ResultCard result={data} />
              <CostChart  result={data} budget={budget} />
            </>
          )}
          {!isPending && !data && !error && (
            <div className="bg-white rounded-2xl shadow p-8 text-center text-gray-400">
              <p className="text-4xl mb-3">🎯</p>
              <p className="font-medium">Fill in the form and click Predict</p>
              <p className="text-sm mt-1">Results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
