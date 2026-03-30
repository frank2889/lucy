// Sparkline — Tiny inline chart for telemetry history
import { useStore } from '../store';

interface Props {
  getValue: (t: { emotions: { curiosity: number; satisfaction: number; frustration: number }; neuro: { dopamine: number; norepinephrine: number; serotonin: number; acetylcholine: number }; tension: number; consciousness: { phi: number; coherence: number } }) => number;
  color: string;
  label: string;
  height?: number;
}

export default function Sparkline({ getValue, color, label, height = 32 }: Props) {
  const history = useStore((s) => s.telemetryHistory);
  if (history.length < 2) return null;

  const values = history.map(getValue);
  const max = Math.max(...values, 0.001);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const w = 120;

  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x},${y}`;
    })
    .join(' ');

  const current = values[values.length - 1];

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-[--text-secondary] w-20 text-right">{label}</span>
      <svg width={w} height={height} className="opacity-80">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="text-[10px] font-mono w-12" style={{ color }}>{current.toFixed(3)}</span>
    </div>
  );
}
