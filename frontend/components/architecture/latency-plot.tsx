import type { LatencyStage } from "@/lib/types";

interface LatencyPlotProps {
  stages: LatencyStage[];
}

export function LatencyPlot({ stages }: LatencyPlotProps) {
  const max = Math.max(...stages.map((stage) => stage.ms), 1);

  return (
    <div className="mt-4 grid gap-2.5">
      {stages.map((stage) => (
        <div key={stage.stage} className="grid grid-cols-[72px_1fr_48px] items-center gap-2.5">
          <span className="text-xs text-muted">{stage.stage}</span>
          <div className="h-2.5 overflow-hidden rounded-pill bg-white/[0.05]">
            <em
              className="block h-full rounded-pill bg-gradient-to-r from-accent to-success shadow-[0_0_18px_rgba(104,232,255,0.4)]"
              style={{ width: `${(stage.ms / max) * 100}%` }}
            />
          </div>
          <b className="text-right text-xs font-medium text-muted">{stage.ms.toFixed(0)}ms</b>
        </div>
      ))}
    </div>
  );
}
