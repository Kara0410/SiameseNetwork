import type { Verdict } from "@/lib/types";

const VERDICT_COLOR: Record<Verdict, string> = {
  verified: "#8BFFCA",
  review: "#FFD183",
  blocked: "#FF7C8A",
};

interface ScoreOrbProps {
  score: number;
  verdict: Verdict;
}

export function ScoreOrb({ score, verdict }: ScoreOrbProps) {
  const sweep = Math.max(0, Math.min(100, score));
  const color = VERDICT_COLOR[verdict];

  return (
    <div
      className="relative grid h-32 w-32 shrink-0 place-items-center rounded-full animate-breathe"
      style={{ background: `conic-gradient(${color} ${sweep}%, rgba(255,255,255,0.09) 0)` }}
    >
      <div className="absolute inset-[9px] rounded-full bg-background-raised" />
      <span className="z-10 text-3xl font-extrabold text-foreground">{score.toFixed(1)}</span>
      <small className="absolute z-10 mt-9 text-[11px] uppercase tracking-[0.14em] text-muted">match</small>
    </div>
  );
}
