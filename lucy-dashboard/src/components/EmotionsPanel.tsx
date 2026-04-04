import { useStore } from '../store';

const RING = 130;
const SW = 9;

function Ring({ value, label, color, bg }: { value: number; label: string; color: string; bg: string }) {
  const r = (RING - SW) / 2;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(Math.max(value ?? 0, 0), 1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <div style={{ position: 'relative', width: RING, height: RING }}>
        <svg width={RING} height={RING} style={{ transform: 'rotate(-90deg)' }}>
          <circle cx={RING/2} cy={RING/2} r={r} fill="none" stroke={bg} strokeWidth={SW} />
          <circle cx={RING/2} cy={RING/2} r={r} fill="none" stroke={color} strokeWidth={SW}
            strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={circ * (1 - pct)}
            style={{ transition: 'stroke-dashoffset 1s ease' }} />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <span className="mono" style={{ fontSize: 28, fontWeight: 700, color }}>{(pct * 100).toFixed(0)}</span>
          <span style={{ fontSize: 11, color: 'var(--text-disabled)' }}>%</span>
        </div>
      </div>
      <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)' }}>{label}</span>
    </div>
  );
}

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  const v = value ?? 0;
  const pct = v > 0 ? Math.min(100, Math.max(4, Math.sqrt(v) * 100)) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '4px 0' }}>
      <span style={{ fontSize: 13, color: 'var(--text-secondary)', width: 120, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--page-bg)', overflow: 'hidden' }}>
        <div style={{ height: '100%', borderRadius: 3, background: color, width: `${pct}%`, transition: 'width 0.7s ease' }} />
      </div>
      <span className="mono" style={{ width: 56, textAlign: 'right', flexShrink: 0, color }}>{v.toFixed(3)}</span>
    </div>
  );
}

export default function EmotionsPanel() {
  const t = useStore((s) => s.telemetry);
  if (!t) return null;
  const e = t.emotions ?? { curiosity: 0, satisfaction: 0, frustration: 0 };
  const n = t.neuro ?? { dopamine: 0, norepinephrine: 0, serotonin: 0, acetylcholine: 0 };
  const c = t.consciousness ?? { spectral_radius: 0, coherence: 0, free_energy: 0, phi: 0 };

  return (
    <div>
      {/* Rings */}
      <div className="card" style={{ padding: '32px 20px', display: 'flex', justifyContent: 'center', gap: 48 }}>
        <Ring value={e.curiosity ?? 0} label="Curiosity" color="var(--cyan)" bg="var(--cyan-bg)" />
        <Ring value={e.satisfaction ?? 0} label="Satisfaction" color="var(--green)" bg="var(--green-bg)" />
        <Ring value={e.frustration ?? 0} label="Frustration" color="var(--red)" bg="var(--red-bg)" />
      </div>
      <div style={{ height: 16 }} />
      {/* Neuro + Consciousness */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12 }}>Neuromodulators</div>
          <Bar label="Dopamine" value={n.dopamine ?? 0} color="var(--orange)" />
          <Bar label="Norepinephrine" value={n.norepinephrine ?? 0} color="var(--red)" />
          <Bar label="Serotonin" value={n.serotonin ?? 0} color="var(--blue)" />
          <Bar label="Acetylcholine" value={n.acetylcholine ?? 0} color="var(--purple)" />
        </div>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 12 }}>Consciousness</div>
          <Bar label="Spectral Radius" value={c.spectral_radius ?? 0} color="var(--purple)" />
          <Bar label="Coherence" value={c.coherence ?? 0} color="var(--blue)" />
          <Bar label="Free Energy" value={Math.min(c.free_energy ?? 0, 1)} color="var(--orange)" />
          <Bar label={'\u03A6 Integrated'} value={c.phi ?? 0} color="var(--cyan)" />
        </div>
      </div>
    </div>
  );
}
