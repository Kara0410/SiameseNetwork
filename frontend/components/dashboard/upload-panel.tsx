"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ModelSelect, useEmbeddingModels } from "@/components/dashboard/model-select";
import { searchImage, verifyImages } from "@/lib/api-client";
import type { SearchResponse, VerifyResponse } from "@/lib/types";

interface UploadPanelProps {
  onResult: (result: VerifyResponse) => void;
  onSearchResult: (result: SearchResponse) => void;
}

interface SlotState {
  file: File | null;
  preview: string | null;
}

const EMPTY_SLOT: SlotState = { file: null, preview: null };

function DropZone({
  label,
  hint,
  slot,
  onFile,
}: {
  label: string;
  hint: string;
  slot: SlotState;
  onFile: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div
      role="button"
      tabIndex={0}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        const file = event.dataTransfer.files?.[0];
        if (file) onFile(file);
      }}
      onClick={() => inputRef.current?.click()}
      className="relative grid h-36 cursor-pointer place-items-center overflow-hidden rounded-lg border border-dashed border-accent/35 bg-white/[0.02] text-center transition-colors hover:border-accent/60"
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
        }}
      />
      {slot.preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={slot.preview} alt={label} className="h-full w-full object-cover" />
      ) : (
        <div className="px-3 text-sm text-muted">
          <UploadCloud className="mx-auto mb-2 h-5 w-5 text-accent" />
          <strong className="text-foreground">{label}</strong>
          <p className="mt-1 text-xs">{hint}</p>
        </div>
      )}
    </div>
  );
}

export function UploadPanel({ onResult, onSearchResult }: UploadPanelProps) {
  const [slotA, setSlotA] = useState<SlotState>(EMPTY_SLOT);
  const [slotB, setSlotB] = useState<SlotState>(EMPTY_SLOT);
  const { models, model, setModel } = useEmbeddingModels();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (slotA.preview) URL.revokeObjectURL(slotA.preview);
      if (slotB.preview) URL.revokeObjectURL(slotB.preview);
    };
  }, [slotA.preview, slotB.preview]);

  function setSlot(set: (slot: SlotState) => void, file: File) {
    set({ file, preview: URL.createObjectURL(file) });
  }

  async function handleRun() {
    if (!slotA.file || !slotB.file) {
      setError("Provide both a reference and a candidate image.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await verifyImages(slotA.file, slotB.file, model || undefined);
      onResult(result);
      const search = await searchImage(slotB.file, model || undefined);
      onSearchResult(search);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-3.5">
      <div className="grid grid-cols-2 gap-3">
        <DropZone label="Reference image" hint="Trusted identity" slot={slotA} onFile={(file) => setSlot(setSlotA, file)} />
        <DropZone label="Candidate image" hint="Live capture / upload" slot={slotB} onFile={(file) => setSlot(setSlotB, file)} />
      </div>

      <ModelSelect models={models} model={model} onChange={setModel} />

      <Button onClick={handleRun} disabled={loading} className="w-full">
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        {loading ? "Running inference…" : "Run verification"}
      </Button>

      {error && <p className="text-sm text-danger">{error}</p>}
    </div>
  );
}
