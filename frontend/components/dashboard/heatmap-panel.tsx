import Image from "next/image";

import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { VerifyResponse } from "@/lib/types";

interface HeatmapPanelProps {
  result: VerifyResponse | null;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/[0.055] p-3.5">
      <small className="text-[11px] uppercase tracking-[0.14em] text-muted">{label}</small>
      <strong className="block text-xl text-foreground">{value}</strong>
    </div>
  );
}

export function HeatmapPanel({ result }: HeatmapPanelProps) {
  return (
    <Card>
      <CardHeader>
        <div>
          <CardDescription>Explainability</CardDescription>
          <CardTitle>Attention heatmap</CardTitle>
        </div>
        <div className="flex flex-col items-end gap-1.5">
          {result && <Badge variant="accent">{result.model}</Badge>}
          <Badge variant="success">{result ? (result.anomalies[0] ?? "no spoof signal") : "idle"}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        {result?.heatmap_a && result?.heatmap_b ? (
          <div className="grid grid-cols-2 gap-3">
            <div className="overflow-hidden rounded-lg border border-border">
              <Image src={result.heatmap_a} alt="Image A heatmap" width={200} height={200} className="h-full w-full object-cover" unoptimized />
            </div>
            <div className="overflow-hidden rounded-lg border border-border">
              <Image src={result.heatmap_b} alt="Image B heatmap" width={200} height={200} className="h-full w-full object-cover" unoptimized />
            </div>
          </div>
        ) : (
          <div className="grid h-48 place-items-center rounded-lg border border-dashed border-accent/30 text-sm text-muted">
            Run a verification to see attention overlays
          </div>
        )}

        <div className="mt-3.5 grid grid-cols-2 gap-2.5">
          <Stat label="similarity" value={result ? `${(result.similarity * 100).toFixed(1)}%` : "—"} />
          <Stat label="distance" value={result ? result.distance.toFixed(3) : "—"} />
          <Stat label="spoof risk" value={result ? result.spoof_risk.toFixed(2) : "—"} />
          <Stat label="latency" value={result ? `${result.latency_ms.toFixed(0)}ms` : "—"} />
        </div>

        {result && result.anomalies.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {result.anomalies.map((flag) => (
              <Badge key={flag} variant={flag === "no spoof signal" ? "success" : "warning"}>
                {flag}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
