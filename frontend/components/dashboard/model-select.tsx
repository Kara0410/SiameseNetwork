"use client";

import { useEffect, useState } from "react";

import { getModels } from "@/lib/api-client";
import type { ModelInfo } from "@/lib/types";

export function useEmbeddingModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");

  useEffect(() => {
    getModels()
      .then((response) => {
        setModels(response.models);
        setModel(response.default);
      })
      .catch(() => undefined);
  }, []);

  return { models, model, setModel };
}

interface ModelSelectProps {
  models: ModelInfo[];
  model: string;
  onChange: (model: string) => void;
}

export function ModelSelect({ models, model, onChange }: ModelSelectProps) {
  if (models.length === 0) return null;

  return (
    <label className="flex items-center justify-between gap-3 text-xs text-muted">
      <span className="uppercase tracking-[0.14em]">Embedding model</span>
      <select
        value={model}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-md border border-border bg-surface px-2.5 py-1.5 text-sm text-foreground"
      >
        {models.map((item) => (
          <option key={item.name} value={item.name}>
            {item.display_name}
          </option>
        ))}
      </select>
    </label>
  );
}
