// EmotionsPanel — Real-time emotion bars + neuromodulator gauges
import { useStore } from '../store';

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.min(value * 100, 100);
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 text-right text-[--text-secondary]">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-[--bg-secondary] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-12 text-right font-mono" style={{ color }}>{value.toFixed(3)}</span>
    </div>
  );
}

export default function EmotionsPanel() {
  const t = useStore((s) => s.telemetry);
  if (!t) return null;

  return (
    <div className="glass p-3 flex flex-col gap-2">
      <h3 className="text-xs uppercase tracking-widest text-[--text-secondary] mb-1">Emotions</h3>
      <Bar label="curiosity" value={t.emotions.curiosity} color="var(--accent-cyan)" />
      <Bar label="satisfaction" value={t.emotions.satisfaction} color="var(--accent-green)" />
      <Bar label="frustration" value={t.emotions.frustration} color="var(--accent-red)" />

      <h3 className="text-xs uppercase tracking-widest text-[--text-secondary] mt-2 mb-1">Neuromodulators</h3>
      <Bar label="dopamine" value={t.neuro.dopamine} color="#ffaa00" />
      <Bar label="norepinephrine" value={t.neuro.norepinephrine} color="#ff6644" />
      <Bar label="serotonin" value={t.neuro.serotonin} color="#44aaff" />
      <Bar label="acetylcholine" value={t.neuro.acetylcholine} color="#44ff88" />

      <h3 className="text-xs uppercase tracking-widest text-[--text-secondary] mt-2 mb-1">Consciousness</h3>
      <Bar label="spectral radius" value={t.consciousness.spectral_radius} color="var(--accent-purple)" />
      <Bar label="coherence" value={t.consciousness.coherence} color="var(--accent-blue)" />
      <Bar label="free energy" value={Math.min(t.consciousness.free_energy, 1)} color="var(--accent-orange)" />
    </div>
  );
}
