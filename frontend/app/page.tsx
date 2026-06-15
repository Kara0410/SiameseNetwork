"use client";

import { useEffect, useState } from "react";

import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ScoreOrb } from "@/components/dashboard/score-orb";
import { TelemetryRow } from "@/components/dashboard/telemetry-row";
import { UploadPanel } from "@/components/dashboard/upload-panel";
import { WebcamCapturePanel } from "@/components/dashboard/webcam-capture-panel";
import { WebcamPanel } from "@/components/dashboard/webcam-panel";
import { HeatmapPanel } from "@/components/dashboard/heatmap-panel";
import { EmbeddingPlot } from "@/components/dashboard/embedding-plot";
import { ReasoningPanel } from "@/components/dashboard/reasoning-panel";
import { getArchitecture } from "@/lib/api-client";
import type { ArchitectureResponse, SearchResponse, VerifyResponse } from "@/lib/types";

export default function DashboardPage() {
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [architecture, setArchitecture] = useState<ArchitectureResponse | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    getArchitecture().then(setArchitecture).catch(() => undefined);
  }, []);

  function handleResult(next: VerifyResponse) {
    setResult(next);
    setRefreshToken((token) => token + 1);
  }

  const score = result ? result.similarity * 100 : 0;
  const verdict = result?.verdict ?? "review";

  return (
    <main className="grid gap-4 pb-10 lg:grid-cols-[1.15fr_0.85fr_360px]">
      <Card className="p-6 lg:col-span-2">
        <span className="text-xs uppercase tracking-[0.14em] text-muted">Live verification</span>
        <h1 className="mt-1.5 text-4xl font-extrabold leading-[0.95] tracking-tight text-foreground sm:text-6xl">
          Identity intelligence, reduced to signal.
        </h1>
        <p className="mt-2.5 max-w-xl text-muted">
          Snap a reference and candidate photo with your webcam, or upload images directly. Every
          run produces a similarity score, explainability overlay, vector neighbors, and an LLM
          reasoning trace.
        </p>
      </Card>

      <Card className="grid grid-cols-[128px_1fr] items-center gap-4 p-6">
        <ScoreOrb score={score} verdict={verdict} />
        <div>
          <div className="text-2xl font-extrabold capitalize tracking-tight text-foreground">
            {result ? result.verdict : "Awaiting input"}
          </div>
          <p className="mt-1 text-sm text-muted">
            {result ? result.reasoning : "Run a verification to see a live decision."}
          </p>
        </div>
      </Card>

      <div className="lg:col-span-3">
        <TelemetryRow refreshToken={refreshToken} />
      </div>

      <Card>
        <CardHeader>
          <div>
            <CardDescription>Input stream</CardDescription>
            <CardTitle>Capture</CardTitle>
          </div>
          <Badge variant={result ? "success" : "default"}>{result ? "ready" : "idle"}</Badge>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="webcam-capture">
            <TabsList>
              <TabsTrigger value="webcam-capture">Webcam</TabsTrigger>
              <TabsTrigger value="upload">Upload</TabsTrigger>
              <TabsTrigger value="live">Live stream</TabsTrigger>
            </TabsList>
            <TabsContent value="webcam-capture">
              <WebcamCapturePanel onResult={handleResult} onSearchResult={setSearchResult} />
            </TabsContent>
            <TabsContent value="upload">
              <UploadPanel onResult={handleResult} onSearchResult={setSearchResult} />
            </TabsContent>
            <TabsContent value="live">
              <WebcamPanel />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <HeatmapPanel result={result} />

      <div className="lg:row-span-2">
        <EmbeddingPlot result={searchResult} highlightId={result?.trace_id} />
      </div>

      <ReasoningPanel result={result} latency={architecture?.latency ?? []} />
    </main>
  );
}
