// EmotionsPanel — Apple Health style emotion rings + neuromodulators
import { useStore } from '../store';

function Ring({ value, label, color, size = 100 }: { value: number; label: string; color: string; size?: number }) {
  const r = (size - 12) / 2;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(Math.max(value ?? 0, 0), 1);
  const offset = circ * (1 - pct);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size/2} cy={size/2} r={r} className="ring-track" />
          <circle
            cx={size/2} cy={size/2} r={r}
            className="ring-value"
            stroke={color}
            strokeDasharray={circ}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold mono" style={{ color }}>
            {(pct * 100).toFixed(0)}
          </span>
        </div>
      </div>
      <span className="text-sm font-medium text-[--text2]">{label}</span>
    </div>
  );
}

function MiniBar({ label, value, color }: { label: string; value: number; color: string }) {
  const v = value ?? 0;
  const pct = Math.min(100, Math.max(0, v * 100));
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-[--text2] w-32 truncate">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-[--bg] overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.max(pct, 1)}%`, background: color }} />
      </div>
      <span className="text-sm mono w-14 text-right font-medium" style={{ color }}>{v.toFixed(3)}</span>
    </div>
  );
}

export default function EmotionsPanel() {
  const t = useStore((s) => s.telemetry);
  if (!t) return null;

  return (
    <div className="card fade-in">
      {/* Emotion Rings — hero element */}
      <div className="flex items-center justify-center gap-12 py-4 mb-6">
        <Ring value={t.emotions.curiosity} label="Curiosity" color="var(--cyan)" size={120} />
        <Ring value={t.emotions.satisfaction} label="Satisfaction" color="var(--green)" size={120} />
        <Ring value={t.emotions.frustration} label="Frustration" color="var(--red)" size={120} />
      </div>

      {/* Neuromodulators + Consciousness — compact bars */}
      <div className="grid grid-cols-2 gap-x-8 gap-y-2">
        <div>
          <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider mb-2">Neuromodulators</h3>
          <MiniBar label="Dopamine" value={t.neuro.dopamine} color="var(--orange)" />
          <MiniBar label="Norepinephrine" value={t.neuro.norepinephrine} color="var(--red)" />
          <MiniBar label="Serotonin" value={t.neuro.serotonin} color="var(--blue)" />
          <MiniBar label="Acetylcholine" value={t.neuro.acetylcholine} color="var(--purple)" />
        </div>
        <div>
          <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider mb-2">Consciousness</h3>
          <MiniBar label="Spectral Radius" value={t.consciousness.spectral_radius} color="var(--purple)" />
          <MiniBar label="Coherence" value={t.consciousness.coherence} color="var(--blue)" />
          <MiniBar label="Free Energy" value={Math.min(t.consciousness.free_energy, 1)} color="var(--orange)" />
          <MiniBar label={'\u03A6 Integrated'} value={t.consciousness.phi} color="var(--teal)" />
        </div>
      </div>
    </div>
  );
}
