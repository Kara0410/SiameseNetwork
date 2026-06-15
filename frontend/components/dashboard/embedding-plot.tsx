import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { VectorResults } from "./vector-results";
import type { SearchResponse } from "@/lib/types";

interface EmbeddingPlotProps {
  result: SearchResponse | null;
  highlightId?: string;
}

export function EmbeddingPlot({ result, highlightId }: EmbeddingPlotProps) {
  const points = result?.embedding_map ?? [];
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  return (
    <Card>
      <CardHeader>
        <div>
          <CardDescription>Vector search</CardDescription>
          <CardTitle>Embedding map</CardTitle>
        </div>
        <Badge variant="accent">FAISS</Badge>
      </CardHeader>
      <CardContent>
        <div className="relative h-[260px] overflow-hidden rounded-lg border border-border bg-[radial-gradient(circle_at_center,rgba(104,232,255,0.1),transparent_58%)]">
          {points.length === 0 ? (
            <div className="grid h-full place-items-center text-sm text-muted">
              Run a verification to populate the embedding map
            </div>
          ) : (
            points.map((point) => {
              const left = 6 + ((point.x - minX) / rangeX) * 88;
              const top = 6 + ((point.y - minY) / rangeY) * 88;
              const active = point.id === highlightId;
              return (
                <span
                  key={point.id}
                  className={
                    active
                      ? "absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow-[0_0_28px_#68E8FF] animate-float"
                      : "absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-success shadow-[0_0_16px_#8BFFCA] animate-float"
                  }
                  style={{ left: `${left}%`, top: `${top}%` }}
                />
              );
            })
          )}
        </div>
        <VectorResults matches={result?.matches ?? []} />
      </CardContent>
    </Card>
  );
}
