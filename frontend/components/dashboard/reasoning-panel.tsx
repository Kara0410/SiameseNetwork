import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LatencyPlot } from "@/components/architecture/latency-plot";
import type { LatencyStage, VerifyResponse } from "@/lib/types";

interface ReasoningPanelProps {
  result: VerifyResponse | null;
  latency: LatencyStage[];
}

export function ReasoningPanel({ result, latency }: ReasoningPanelProps) {
  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <div>
          <CardDescription>LLM trace</CardDescription>
          <CardTitle>Decision summary</CardTitle>
        </div>
        <Badge variant="accent">{result?.model ?? "policy"}</Badge>
      </CardHeader>
      <CardContent>
        <p className="text-xl leading-snug tracking-tight text-foreground">
          {result?.reasoning ?? "Run a verification to generate a reasoning trace."}
        </p>
        <LatencyPlot stages={latency} />
      </CardContent>
    </Card>
  );
}
