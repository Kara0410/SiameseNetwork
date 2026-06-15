"use client";

import { useEffect, useState } from "react";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Sparkline } from "./sparkline";
import { getHistory } from "@/lib/api-client";
import type { HistoryItem } from "@/lib/types";

interface Metric {
  label: string;
  value: string;
  values: number[];
  tone: "accent" | "mint";
}

function normalize(values: number[]): number[] {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return values.map(() => 50);
  return values.map((value) => ((value - min) / (max - min)) * 100);
}

function buildMetrics(items: HistoryItem[]): Metric[] {
  const chrono = [...items].reverse();
  const similarity = chrono.map((item) => item.similarity * 100);
  const latency = chrono.map((item) => item.latency_ms);
  const spoofRisk = chrono.map((item) => item.spoof_risk * 100);
  const verifiedRate = chrono.map((_, index) => {
    const slice = chrono.slice(0, index + 1);
    const verified = slice.filter((item) => item.verdict === "verified").length;
    return (verified / slice.length) * 100;
  });

  return [
    {
      label: "similarity",
      value: similarity.length ? `${(similarity.at(-1) ?? 0).toFixed(1)}%` : "—",
      values: similarity,
      tone: "accent",
    },
    {
      label: "latency",
      value: latency.length ? `${(latency.at(-1) ?? 0).toFixed(0)}ms` : "—",
      values: normalize(latency),
      tone: "accent",
    },
    {
      label: "spoof risk",
      value: spoofRisk.length ? (spoofRisk.at(-1) ?? 0).toFixed(1) : "—",
      values: spoofRisk,
      tone: "mint",
    },
    {
      label: "verified rate",
      value: verifiedRate.length ? `${(verifiedRate.at(-1) ?? 0).toFixed(0)}%` : "—",
      values: verifiedRate,
      tone: "mint",
    },
  ];
}

interface TelemetryRowProps {
  refreshToken?: number;
}

export function TelemetryRow({ refreshToken }: TelemetryRowProps) {
  const [metrics, setMetrics] = useState<Metric[] | null>(null);

  useEffect(() => {
    let active = true;
    getHistory(20)
      .then((response) => {
        if (active) setMetrics(buildMetrics(response.items));
      })
      .catch(() => {
        if (active) setMetrics([]);
      });
    return () => {
      active = false;
    };
  }, [refreshToken]);

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {(metrics ?? Array.from({ length: 4 })).map((metric, index) => (
        <Card key={metric?.label ?? index}>
          <CardContent className="p-4 pt-5">
            {metric ? (
              <>
                <span className="text-xs uppercase tracking-[0.14em] text-muted">{metric.label}</span>
                <strong className="mt-1.5 block text-2xl text-foreground">{metric.value}</strong>
                <Sparkline values={metric.values} tone={metric.tone} />
              </>
            ) : (
              <>
                <Skeleton className="h-3 w-16" />
                <Skeleton className="mt-2 h-7 w-20" />
                <Skeleton className="mt-3 h-12 w-full" />
              </>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
