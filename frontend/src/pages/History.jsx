import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHistory } from "../api/client";
import HistoryTable from "../components/HistoryTable";

export default function History() {
  const [page, setPage] = useState(1);
  const { data, isLoading, error } = useQuery({
    queryKey: ["history", page],
    queryFn:  () => getHistory(page),
  });

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-extrabold text-gray-900">Prediction History</h1>
        <p className="text-gray-500 mt-1">All past SDLC predictions and cost estimates</p>
      </div>

      {isLoading && <p className="text-gray-400 text-center py-8">Loading...</p>}
      {error    && <p className="text-red-500 text-center py-8">Failed to load history.</p>}

      {data && (
        <HistoryTable
          data={data.predictions}
          page={data.page}
          pages={data.pages}
          onPage={setPage}
        />
      )}
    </div>
  );
}
