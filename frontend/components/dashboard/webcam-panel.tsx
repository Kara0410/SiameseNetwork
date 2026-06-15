"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useLiveSocket } from "@/lib/use-live-socket";

const FRAME_INTERVAL_MS = 900;

export function WebcamPanel() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const resetNextFrameRef = useRef(true);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { connected, latest, sendFrame } = useLiveSocket({ enabled: streaming });

  useEffect(() => {
    if (!streaming) return;

    const interval = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUri = canvas.toDataURL("image/jpeg", 0.7);
      sendFrame(dataUri, resetNextFrameRef.current);
      resetNextFrameRef.current = false;
    }, FRAME_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [streaming, sendFrame]);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      resetNextFrameRef.current = true;
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

  useEffect(() => () => stop(), []);

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

      <div className="flex items-center gap-2.5">
        {streaming ? (
          <Button variant="ghost" onClick={stop} className="flex-1">
            <CameraOff className="h-4 w-4" />
            Stop camera
          </Button>
        ) : (
          <Button onClick={start} className="flex-1">
            <Camera className="h-4 w-4" />
            Start camera
          </Button>
        )}
        <Button
          variant="outline"
          size="icon"
          disabled={!streaming}
          onClick={() => {
            resetNextFrameRef.current = true;
          }}
          title="Reset reference frame"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      {error && <p className="text-sm text-danger">{error}</p>}

      <div className="grid grid-cols-2 gap-2.5">
        <div className="rounded-lg bg-white/[0.055] p-3.5">
          <small className="text-[11px] uppercase tracking-[0.14em] text-muted">live similarity</small>
          <strong className="block text-xl text-foreground">
            {latest ? `${(latest.similarity * 100).toFixed(1)}%` : "—"}
          </strong>
        </div>
        <div className="rounded-lg bg-white/[0.055] p-3.5">
          <small className="text-[11px] uppercase tracking-[0.14em] text-muted">spoof risk</small>
          <strong className="block text-xl text-foreground">{latest ? latest.spoof_risk.toFixed(2) : "—"}</strong>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-muted">
        <span>{streaming ? (connected ? "stream connected" : "connecting…") : "idle"}</span>
        {latest && latest.flags.length > 0 && <Badge variant="warning">{latest.flags[0]}</Badge>}
      </div>
    </div>
  );
}
