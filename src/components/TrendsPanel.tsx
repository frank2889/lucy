// TrendsPanel — Sparkline trend charts
import { useStore } from '../store';
import type { Telemetry } from '../types';

interface ChartProps {
  label: string;
  getValue: (t: Telemetry) => number;
  color: string;
}

function Chart({ label, getValue, color }: ChartProps) {
  const history = useStore((s) => s.telemetryHistory);
  if (history.length < 2) return null;

  const values = history.map(getValue);
  const max = Math.max(...values, 0.001);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const w = 500;
  const h = 48;

  const linePoints = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 8) - 4;
      return `${x},${y}`;
    })
    .join(' ');

  const areaPath = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 8) - 4;
      return `${i === 0 ? 'M' : 'L'}${x},${y}`;
    })
    .join(' ') + ` L${w},${h} L0,${h} Z`;

  const current = values[values.length - 1];
  const id = `trend-${label}`;

  return (
    <div className="flex items-center gap-4">
      <span className="text-sm text-[--text2] w-28 text-right shrink-0 font-medium">{label}</span>
      <svg viewBox={`0 0 ${w} ${h}`} className="flex-1 h-10" preserveAspectRatio="none">
        <defs>
          <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.15" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill={`url(#${id})`} />
        <polyline points={linePoints} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span className="text-sm mono w-16 text-right shrink-0 font-semibold" style={{ color }}>{current.toFixed(4)}</span>
    </div>
  );
}

export default function TrendsPanel() {
  const history = useStore((s) => s.telemetryHistory);
  if (history.length < 3) return (
    <div className="card text-sm text-[--text2] py-8 text-center">Collecting data...</div>
  );

  return (
    <div className="card fade-in flex flex-col gap-2">
      <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider mb-2">Trends</h3>
      <Chart label="Curiosity" getValue={(t) => t.emotions.curiosity} color="var(--cyan)" />
      <Chart label="Satisfaction" getValue={(t) => t.emotions.satisfaction} color="var(--green)" />
      <Chart label="Frustration" getValue={(t) => t.emotions.frustration} color="var(--red)" />
      <Chart label="Tension" getValue={(t) => t.tension} color="var(--orange)" />
      <Chart label="Dopamine" getValue={(t) => t.neuro.dopamine} color="#e58e32" />
      <Chart label="Norepineph." getValue={(t) => t.neuro.norepinephrine} color="var(--pink)" />
      <Chart label={'\u03A6 Integrated'} getValue={(t) => t.consciousness.phi} color="var(--purple)" />
      <Chart label="Coherence" getValue={(t) => t.consciousness.coherence} color="var(--blue)" />
    </div>
  );
}
