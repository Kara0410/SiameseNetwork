import { Card } from "@/components/ui/card";
import type { PipelineStage } from "@/lib/types";

interface NodeDetailProps {
  node: PipelineStage;
}

export function NodeDetail({ node }: NodeDetailProps) {
  return (
    <Card className="min-h-[260px] p-6">
      <div className="grid h-[50px] w-[50px] place-items-center rounded-md bg-accent text-sm font-extrabold text-[#061018]">
        {node.layer}
      </div>
      <h2 className="mt-4 text-4xl font-extrabold tracking-tight text-foreground">{node.title}</h2>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">{node.detail}</p>
      <div className="mt-5 flex flex-wrap gap-2">
        {node.stack.map((chip) => (
          <span key={chip} className="rounded-pill bg-white/[0.07] px-2.5 py-2 text-xs text-muted">
            {chip}
          </span>
        ))}
      </div>
    </Card>
  );
}
