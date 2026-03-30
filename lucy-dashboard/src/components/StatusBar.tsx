// StatusBar — Top bar with tick, phase, and key metrics
import { useStore } from '../store';

export default function StatusBar() {
  const t = useStore((s) => s.telemetry);
  if (!t) return <div className="glass p-2 text-center text-[--text-secondary]">Connecting to Lucy...</div>;

  return (
    <div className="glass px-4 py-2 flex items-center justify-between gap-4 text-sm">
      <div className="flex items-center gap-3">
        <span className="text-[--accent-cyan] font-bold text-base">LUCY</span>
        <span className="text-[--text-secondary]">v2.0</span>
        <span className="border-l border-[--border] pl-3">
          tick <span className="text-[--accent-blue] font-mono">{t.tick.toLocaleString()}</span>
        </span>
        <span className="text-[--text-secondary]">{t.phase}</span>
      </div>
      <div className="flex items-center gap-4">
        <Metric label="points" value={t.points.toLocaleString()} />
        <Metric label="active" value={t.active.toLocaleString()} color="--accent-green" />
        <Metric label="tension" value={t.tension.toFixed(4)} />
        <Metric label="Φ" value={t.consciousness.phi.toFixed(3)} color="--accent-purple" />
        {t.consciousness.is_critical && (
          <span className="text-[--accent-orange] animate-pulse-slow">⚡ CRITICAL</span>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-[--text-secondary]">{label}</span>
      <span className={`font-mono ${color ? `text-[${color}]` : ''}`} style={color ? { color: `var(${color})` } : undefined}>
        {value}
      </span>
    </div>
  );
}
