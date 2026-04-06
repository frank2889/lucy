import { useStore } from '../store';
import { useIsMobile } from '../hooks/useIsMobile';

const RING_DESKTOP = 130;
const RING_MOBILE = 90;
const SW = 9;

function Ring({ value, label, color, bg, size }: { value: number; label: string; color: string; bg: string; size: number }) {
  const r = (size - SW) / 2;
  const circ = 2 * Math.PI * r;
  const pct = Math.min(Math.max(value ?? 0, 0), 1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={bg} strokeWidth={SW} />
          <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={SW}
            strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={circ * (1 - pct)}
            style={{ transition: 'stroke-dashoffset 1s ease' }} />
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <span className="mono" style={{ fontSize: size < 100 ? 20 : 28, fontWeight: 700, color }}>{(pct * 100).toFixed(0)}</span>
          <span style={{ fontSize: 11, color: 'var(--text-disabled)' }}>%</span>
        </div>
      </div>
      <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)' }}>{label}</span>
    </div>
  );
}

function Bar({ label, value, color, compact }: { label: string; value: number; color: string; compact?: boolean }) {
  const v = value ?? 0;
  const pct = v > 0 ? Math.min(100, Math.max(4, Math.sqrt(v) * 100)) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: compact ? 6 : 12, padding: '3px 0' }}>
      <span style={{ fontSize: compact ? 11 : 13, color: 'var(--text-secondary)', width: compact ? 80 : 120, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--page-bg)', overflow: 'hidden' }}>
        <div style={{ height: '100%', borderRadius: 3, background: color, width: `${pct}%`, transition: 'width 0.7s ease' }} />
      </div>
      <span className="mono" style={{ width: compact ? 44 : 56, textAlign: 'right', flexShrink: 0, fontSize: compact ? 11 : 13, color }}>{v.toFixed(compact ? 2 : 3)}</span>
    </div>
  );
}

export default function EmotionsPanel() {
  const t = useStore((s) => s.telemetry);
  if (!t) return null;
  const e = t.emotions ?? { curiosity: 0, satisfaction: 0, frustration: 0 };
  const n = t.neuro ?? { dopamine: 0, norepinephrine: 0, serotonin: 0, acetylcholine: 0 };
  const c = t.consciousness ?? { spectral_radius: 0, coherence: 0, free_energy: 0, phi: 0,
    theta_gamma_coupling: 0, branching_ratio: 0, ignition_active: false, ignition_strength: 0, expected_free_energy: 0 };
  const mobile = useIsMobile();
  const ringSize = mobile ? RING_MOBILE : RING_DESKTOP;
  const compact = mobile;

  return (
    <div>
      {/* Rings */}
      <div className="card" style={{ padding: mobile ? '16px 10px' : '32px 20px', display: 'flex', justifyContent: 'center', gap: mobile ? 16 : 48 }}>
        <Ring value={e.curiosity ?? 0} label="Curiosity" color="var(--cyan)" bg="var(--cyan-bg)" size={ringSize} />
        <Ring value={e.satisfaction ?? 0} label="Satisfaction" color="var(--green)" bg="var(--green-bg)" size={ringSize} />
        <Ring value={e.frustration ?? 0} label="Frustration" color="var(--red)" bg="var(--red-bg)" size={ringSize} />
      </div>
      <div style={{ height: mobile ? 8 : 16 }} />
      {/* Neuro + Consciousness */}
      <div style={{ display: 'grid', gridTemplateColumns: mobile ? '1fr' : '1fr 1fr', gap: mobile ? 8 : 16 }}>
        <div className="card" style={{ padding: mobile ? 12 : 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>Neuromodulators</div>
          <Bar label="Dopamine" value={n.dopamine ?? 0} color="var(--orange)" compact={compact} />
          <Bar label="NE" value={n.norepinephrine ?? 0} color="var(--red)" compact={compact} />
          <Bar label="Serotonin" value={n.serotonin ?? 0} color="var(--blue)" compact={compact} />
          <Bar label="ACh" value={n.acetylcholine ?? 0} color="var(--purple)" compact={compact} />
        </div>
        <div className="card" style={{ padding: mobile ? 12 : 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>Consciousness</div>
          <Bar label={'\u03A6 Causal'} value={c.phi ?? 0} color="var(--cyan)" compact={compact} />
          <Bar label="Coherence" value={c.coherence ?? 0} color="var(--blue)" compact={compact} />
          <Bar label={'\u03B8-\u03B3 Coupl'} value={c.theta_gamma_coupling ?? 0} color="var(--green)" compact={compact} />
          <Bar label="Branch \u03C3" value={c.branching_ratio ?? 0} color="var(--orange)" compact={compact} />
          <Bar label="Spectral R" value={c.spectral_radius ?? 0} color="var(--purple)" compact={compact} />
          <Bar label={`Ignition ${c.ignition_active ? '\u26A1' : '\u25CB'}`} value={c.ignition_strength ?? 0} color={c.ignition_active ? 'var(--red)' : 'var(--text-disabled)'} compact={compact} />
          <Bar label="Exp FE" value={Math.min(c.expected_free_energy ?? 0, 1)} color="var(--orange)" compact={compact} />
        </div>
      </div>
    </div>
  );
}
