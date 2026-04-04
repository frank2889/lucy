// Sparkline — Inline SVG chart with gradient fill
import { useStore } from '../store';

interface Props {
  getValue: (t: { emotions: { curiosity: number; satisfaction: number; frustration: number }; neuro: { dopamine: number; norepinephrine: number; serotonin: number; acetylcholine: number }; tension: number; consciousness: { phi: number; coherence: number } }) => number;
  color: string;
  label: string;
  height?: number;
}

export default function Sparkline({ getValue, color, label, height = 28 }: Props) {
  const history = useStore((s) => s.telemetryHistory);
  const id = `spark-${label}`;
  if (history.length < 2) return null;

  const values = history.map(getValue);
  const max = Math.max(...values, 0.001);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const w = 140;

  const linePoints = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x},${y}`;
    })
    .join(' ');

  // Area fill path
  const areaPath = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${i === 0 ? 'M' : 'L'}${x},${y}`;
    })
    .join(' ') + ` L${w},${height} L0,${height} Z`;

  const current = values[values.length - 1];

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-[--text-secondary] w-20 text-right truncate">{label}</span>
      <svg width={w} height={height} className="flex-shrink-0">
        <defs>
          <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill={`url(#${id})`} />
        <polyline
          points={linePoints}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.9}
        />
        {/* Current value dot */}
        {values.length > 0 && (
          <circle
            cx={w}
            cy={height - ((current - min) / range) * (height - 4) - 2}
            r="2.5"
            fill={color}
          />
        )}
      </svg>
      <span className="text-[10px] font-mono w-12 text-right" style={{ color }}>{current.toFixed(3)}</span>
    </div>
  );
}
