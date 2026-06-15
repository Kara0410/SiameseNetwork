"use client";

import { cn } from "@/lib/utils";
import type { PipelineStage } from "@/lib/types";

interface PipelineMotionProps {
  stages: PipelineStage[];
  activeId: string;
  onPick: (id: string) => void;
}

export function PipelineMotion({ stages, activeId, onPick }: PipelineMotionProps) {
  return (
    <div
      className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6"
      aria-label="Architecture pipeline"
    >
      {stages.map((stage, index) => (
        <button
          key={stage.id}
          onClick={() => onPick(stage.id)}
          className={cn(
            "relative min-h-[116px] rounded-lg border border-border bg-white/[0.045] p-4 text-left transition-colors",
            activeId === stage.id && "border-accent/45 bg-accent/[0.17]"
          )}
        >
          <small className="text-[11px] uppercase tracking-[0.14em] text-muted">{stage.layer}</small>
          <strong className="mt-2.5 block text-foreground">{stage.title}</strong>
          <span className="text-xs text-muted">{stage.metric}</span>
          {index < stages.length - 1 && (
            <i className="absolute right-[-13px] top-1/2 hidden h-0.5 w-4 -translate-y-1/2 bg-accent shadow-[0_0_14px_#68E8FF] lg:block" />
          )}
        </button>
      ))}
    </div>
  );
}
