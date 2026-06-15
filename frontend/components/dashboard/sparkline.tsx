interface SparklineProps {
  values: number[];
  tone?: "accent" | "mint";
}

/** Renders a small trend line; `values` are expected in the 0-100 range. */
export function Sparkline({ values, tone = "accent" }: SparklineProps) {
  if (values.length === 0) {
    return <svg className="mt-1.5 h-12 w-full" viewBox="0 0 100 52" aria-hidden="true" />;
  }

  const color = tone === "mint" ? "#8BFFCA" : "#68E8FF";
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 46 - (Math.max(0, Math.min(100, value)) / 100) * 38;
      return `${x},${y}`;
    })
    .join(" ");
  const last = Math.max(0, Math.min(100, values.at(-1) ?? 0));
  const lastY = 46 - (last / 100) * 38;

  return (
    <svg className="mt-1.5 h-12 w-full" viewBox="0 0 100 52" aria-hidden="true">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 0 8px ${color})` }}
      />
      <circle cx={100} cy={lastY} r={2.8} fill={color} />
    </svg>
  );
}
