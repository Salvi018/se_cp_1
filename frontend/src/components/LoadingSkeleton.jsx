function Bone({ className = "" }) {
  return (
    <div className={`bg-gray-200 rounded-lg animate-pulse ${className}`} />
  );
}

export default function LoadingSkeleton() {
  return (
    <div className="space-y-6">
      {/* ResultCard skeleton */}
      <div className="bg-white rounded-2xl shadow p-6 space-y-4">
        <Bone className="h-5 w-32" />
        <div className="flex items-center gap-3">
          <Bone className="h-10 w-28" />
          <Bone className="h-4 w-20" />
          <Bone className="flex-1 h-2" />
        </div>
        <div className="flex gap-2">
          <Bone className="h-6 w-16" />
          <Bone className="h-6 w-16" />
          <Bone className="h-6 w-16" />
        </div>
        <hr />
        <div className="grid grid-cols-2 gap-4">
          <Bone className="h-20" />
          <Bone className="h-20" />
        </div>
      </div>

      {/* CostChart skeleton */}
      <div className="bg-white rounded-2xl shadow p-6 space-y-4">
        <Bone className="h-5 w-36" />
        <Bone className="h-48 w-full" />
      </div>
    </div>
  );
}
