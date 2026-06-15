"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { liveSocketUrl } from "./api-client";
import type { LiveFrameError, LiveFrameResult } from "./types";

interface UseLiveSocketOptions {
  enabled: boolean;
}

interface UseLiveSocketState {
  connected: boolean;
  latest: LiveFrameResult | null;
  error: string | null;
}

/** Connects to `/api/v1/live` and exposes a `sendFrame` function for streaming
 * base64 JPEG frames; `enabled` controls connect/disconnect. */
export function useLiveSocket({ enabled }: UseLiveSocketOptions) {
  const socketRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<UseLiveSocketState>({
    connected: false,
    latest: null,
    error: null,
  });

  useEffect(() => {
    if (!enabled) {
      socketRef.current?.close();
      socketRef.current = null;
      setState((prev) => ({ ...prev, connected: false }));
      return;
    }

    const socket = new WebSocket(liveSocketUrl());
    socketRef.current = socket;

    socket.onopen = () => setState((prev) => ({ ...prev, connected: true, error: null }));
    socket.onclose = () => setState((prev) => ({ ...prev, connected: false }));
    socket.onerror = () => setState((prev) => ({ ...prev, error: "Live socket connection error" }));
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as LiveFrameResult | LiveFrameError;
      if ("error" in data) {
        setState((prev) => ({ ...prev, error: data.error }));
      } else {
        setState((prev) => ({ ...prev, latest: data, error: null }));
      }
    };

    return () => socket.close();
  }, [enabled]);

  const sendFrame = useCallback((dataUri: string, reset = false) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ image: dataUri, reset }));
    }
  }, []);

  return { ...state, sendFrame };
}
