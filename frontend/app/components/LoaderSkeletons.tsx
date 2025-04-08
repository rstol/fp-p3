import { Skeleton } from '~/components/ui/skeleton';

export function ClusterViewSkeleton() {
  return (
    <div className="h-[600px] flex-1">
      <Skeleton className="h-full w-full rounded-lg" />
      <div className="mt-4 flex justify-between">
        <Skeleton className="h-6 w-72" />
        <div className="flex gap-2">
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
          <Skeleton className="h-8 w-8 rounded-full" />
        </div>
      </div>
    </div>
  );
}

export function ScatterPlotSkeleton() {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center p-8">
      <div className="mb-4 flex w-full justify-between">
        <Skeleton className="h-8 w-40" />
        <Skeleton className="h-8 w-32" />
      </div>
      <div className="relative h-96 w-full rounded-md border p-4">
        <Skeleton className="h-full w-full" />
        {/* Fake dots */}
        {Array(20)
          .fill(0)
          .map((_, i) => (
            <Skeleton
              key={i}
              className="absolute h-3 w-3 rounded-full"
              style={{
                left: `${10 + Math.random() * 80}%`,
                top: `${10 + Math.random() * 80}%`,
              }}
            />
          ))}
      </div>
      <div className="mt-4 flex w-full justify-between">
        <Skeleton className="h-6 w-24" />
        <Skeleton className="h-6 w-24" />
      </div>
    </div>
  );
}

export function PlayDetailsSkeleton() {
  return (
    <div className="w-96 p-4">
      <Skeleton className="mb-4 h-10 w-full" />

      {/* Court diagram */}
      <Skeleton className="mb-4 h-64 w-full" />

      {/* Game details */}
      <div className="mb-4 space-y-4">
        <div className="flex justify-between">
          <Skeleton className="h-5 w-28" />
          <Skeleton className="h-5 w-28" />
        </div>
        <div className="flex justify-between">
          <Skeleton className="h-5 w-36" />
          <Skeleton className="h-5 w-36" />
        </div>
        <div className="flex justify-between">
          <Skeleton className="h-5 w-36" />
          <Skeleton className="h-5 w-10" />
        </div>
      </div>

      {/* Tag input */}
      <div className="mb-4">
        <Skeleton className="mb-2 h-5 w-20" />
        <div className="flex items-center">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="ml-2 h-6 w-6" />
        </div>
      </div>

      {/* Note input */}
      <div className="mb-6">
        <Skeleton className="mb-2 h-5 w-20" />
        <div className="flex items-center">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="ml-2 h-6 w-6" />
        </div>
      </div>

      {/* Statistics */}
      <Skeleton className="mb-4 h-8 w-48" />
      <div className="space-y-4">
        <div className="flex justify-between">
          <Skeleton className="h-5 w-36" />
          <Skeleton className="h-5 w-10" />
        </div>
        <div className="flex justify-between">
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-5 w-16" />
        </div>
      </div>
    </div>
  );
}

export function BasketballUISkeleton() {
  return (
    <div className="h-full w-full">
      {/* Team selector and filter */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-5 w-5 rounded-full" />
        </div>
        <Skeleton className="h-10 w-40" />
      </div>

      {/* Main content section */}
      <div className="flex gap-6">
        <ClusterViewSkeleton />

        {/* Right panel (details) */}
        <PlayDetailsSkeleton />
      </div>
    </div>
  );
}
