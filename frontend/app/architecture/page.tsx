"use client";

import { useEffect, useState } from "react";

import { Card } from "@/components/ui/card";
import { PipelineMotion } from "@/components/architecture/pipeline-motion";
import { NodeDetail } from "@/components/architecture/node-detail";
import { DeploymentMap } from "@/components/architecture/deployment-map";
import { getArchitecture } from "@/lib/api-client";
import type { ArchitectureResponse } from "@/lib/types";

export default function ArchitecturePage() {
  const [data, setData] = useState<ArchitectureResponse | null>(null);
  const [activeId, setActiveId] = useState("encoder");

  useEffect(() => {
    getArchitecture()
      .then((response) => {
        setData(response);
        const first = response.pipeline[0];
        if (first && !response.pipeline.some((stage) => stage.id === activeId)) {
          setActiveId(first.id);
        }
      })
      .catch(() => undefined);
  }, [activeId]);

  const stages = data?.pipeline ?? [];
  const node = stages.find((stage) => stage.id === activeId) ?? stages[0];

  return (
    <main className="grid gap-4 pb-10">
      <Card className="max-w-3xl p-6">
        <span className="text-xs uppercase tracking-[0.14em] text-muted">Technical overview</span>
        <h1 className="mt-1.5 text-4xl font-extrabold leading-[0.95] tracking-tight text-foreground sm:text-6xl">
          Six systems. One decision path.
        </h1>
        <p className="mt-2.5 text-muted">
          An educational walkthrough of the full pipeline, from a raw image to a served
          decision - including the model architectures, loss functions, and infrastructure
          behind each stage.
        </p>
      </Card>

      {stages.length > 0 && <PipelineMotion stages={stages} activeId={activeId} onPick={setActiveId} />}

      {node && <NodeDetail node={node} />}

      <DeploymentMap groups={data?.deployment ?? []} />
    </main>
  );
}
