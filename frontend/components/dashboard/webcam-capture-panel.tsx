"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Loader2, RefreshCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ModelSelect, useEmbeddingModels } from "@/components/dashboard/model-select";
import { searchImage, verifyImages } from "@/lib/api-client";
import type { SearchResponse, VerifyResponse } from "@/lib/types";

interface WebcamCapturePanelProps {
  onResult: (result: VerifyResponse) => void;
  onSearchResult: (result: SearchResponse) => void;
}

interface Snapshot {
  blob: Blob | null;
  preview: string | null;
}

const EMPTY_SNAPSHOT: Snapshot = { blob: null, preview: null };

function SnapshotSlot({
  label,
  hint,
  snapshot,
  disabled,
  onCapture,
}: {
  label: string;
  hint: string;
  snapshot: Snapshot;
  disabled: boolean;
  onCapture: () => void;
}) {
  return (
    <div className="grid gap-2">
      <div className="relative grid h-28 place-items-center overflow-hidden rounded-lg border border-dashed border-accent/35 bg-white/[0.02]">
        {snapshot.preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={snapshot.preview} alt={label} className="h-full w-full object-cover" />
        ) : (
          <div className="px-3 text-center text-sm text-muted">
            <strong className="text-foreground">{label}</strong>
            <p className="mt-1 text-xs">{hint}</p>
          </div>
        )}
      </div>
      <Button variant="outline" size="sm" disabled={disabled} onClick={onCapture}>
        <RefreshCcw className="h-4 w-4" />
        {snapshot.preview ? `Retake ${label.toLowerCase()}` : `Capture ${label.toLowerCase()}`}
      </Button>
    </div>
  );
}

export function WebcamCapturePanel({ onResult, onSearchResult }: WebcamCapturePanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reference, setReference] = useState<Snapshot>(EMPTY_SNAPSHOT);
  const [candidate, setCandidate] = useState<Snapshot>(EMPTY_SNAPSHOT);
  const [loading, setLoading] = useState(false);
  const { models, model, setModel } = useEmbeddingModels();

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    return () => {
      if (reference.preview) URL.revokeObjectURL(reference.preview);
      if (candidate.preview) URL.revokeObjectURL(candidate.preview);
    };
  }, [reference.preview, candidate.preview]);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setStreaming(true);
    } catch {
      setError("Camera access denied or unavailable.");
    }
  }

  function stop() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStreaming(false);
  }

  function capture(set: (next: (prev: Snapshot) => Snapshot) => void) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        set((prev) => {
          if (prev.preview) URL.revokeObjectURL(prev.preview);
          return { blob, preview: URL.createObjectURL(blob) };
        });
      },
      "image/jpeg",
      0.92
    );
  }

  async function handleRun() {
    if (!reference.blob || !candidate.blob) {
      setError("Capture both a reference and a candidate frame.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await verifyImages(reference.blob, candidate.blob, model || undefined);
      onResult(result);
      const search = await searchImage(candidate.blob, model || undefined);
      onSearchResult(search);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-3.5">
      <div className="relative overflow-hidden rounded-lg border border-border bg-[radial-gradient(circle_at_center,rgba(104,232,255,0.14),rgba(255,255,255,0.03))]">
        <video ref={videoRef} autoPlay playsInline muted className="h-48 w-full object-cover" />
        <canvas ref={canvasRef} className="hidden" />
        {!streaming && (
          <div className="absolute inset-0 grid place-items-center text-sm text-muted">
            Camera preview is idle
          </div>
        )}
        {streaming && (
          <span className="absolute left-0 right-0 top-0 h-0.5 animate-scan bg-gradient-to-r from-transparent via-accent to-transparent" />
        )}
      </div>

      {streaming ? (
        <Button variant="ghost" onClick={stop}>
          <CameraOff className="h-4 w-4" />
          Stop camera
        </Button>
      ) : (
        <Button onClick={start}>
          <Camera className="h-4 w-4" />
          Start camera
        </Button>
      )}

      <div className="grid grid-cols-2 gap-3">
        <SnapshotSlot
          label="Reference"
          hint="Trusted identity"
          snapshot={reference}
          disabled={!streaming}
          onCapture={() => capture(setReference)}
        />
        <SnapshotSlot
          label="Candidate"
          hint="Live capture"
          snapshot={candidate}
          disabled={!streaming}
          onCapture={() => capture(setCandidate)}
        />
      </div>

      <ModelSelect models={models} model={model} onChange={setModel} />

      <Button onClick={handleRun} disabled={loading || !reference.blob || !candidate.blob} className="w-full">
        {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        {loading ? "Running inference…" : "Run verification"}
      </Button>

      {error && <p className="text-sm text-danger">{error}</p>}
    </div>
  );
}
