import type { SearchMatch } from "@/lib/types";

interface VectorResultsProps {
  matches: SearchMatch[];
}

export function VectorResults({ matches }: VectorResultsProps) {
  if (matches.length === 0) {
    return <p className="mt-3 text-sm text-muted">No nearest neighbors yet.</p>;
  }

  return (
    <div className="mt-3 grid gap-2">
      {matches.map((match) => (
        <div key={match.id} className="flex items-center justify-between rounded-lg bg-white/[0.055] px-3 py-2.5 text-sm text-muted">
          <span className="truncate font-mono text-xs">{match.id}</span>
          <strong className="text-foreground">{(match.score * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}
