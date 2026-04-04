// StatusBar — Compact header
import { useStore } from '../store';

export default function StatusBar() {
  const t = useStore((s) => s.telemetry);

  if (!t) return (
    <div className="text-center text-[--text2] text-lg py-8">
      Connecting to Lucy...
    </div>
  );

  return (
    <div className="flex items-center justify-between fade-in">
      <div className="flex items-center gap-3">
        <div className="w-3 h-3 rounded-full bg-[--green]" />
        <h1 className="text-2xl font-bold tracking-tight">Lucy</h1>
        <span className="text-[--text2] text-sm">v2.0</span>
      </div>
      <div className="flex items-center gap-6 text-sm text-[--text2]">
        <span>tick <strong className="text-[--text] mono">{t.tick.toLocaleString()}</strong></span>
        <span className="px-2.5 py-1 rounded-full text-xs font-semibold"
          style={{
            color: t.phase === 'Idle' ? 'var(--green)' : t.phase === 'Processing' ? 'var(--blue)' : 'var(--orange)',
            background: t.phase === 'Idle' ? 'rgba(52,199,89,0.12)' : t.phase === 'Processing' ? 'rgba(0,122,255,0.12)' : 'rgba(255,149,0,0.12)',
          }}>
          {t.phase}
        </span>
        <span><strong className="text-[--text] mono">{t.points.toLocaleString()}</strong> nodes</span>
        <span>tension <strong className="text-[--text] mono">{(t.tension ?? 0).toFixed(4)}</strong></span>
      </div>
    </div>
  );
}
