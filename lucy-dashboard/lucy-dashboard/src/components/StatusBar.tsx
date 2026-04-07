import { useStore } from '../store';

export default function StatusBar() {
  const t = useStore((s) => s.telemetry);
  const activity = useStore((s) => s.activity);
  if (!t) return <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-secondary)' }}>Connecting to Lucy...</div>;

  const mood = activity?.mood ?? 'loading';
  const moodColors: Record<string, { color: string; bg: string }> = {
    kalm: { color: 'var(--green)', bg: 'var(--green-bg)' },
    curious: { color: 'var(--cyan)', bg: 'var(--cyan-bg)' },
    gefrustreerd: { color: 'var(--red)', bg: 'var(--red-bg)' },
    loading: { color: 'var(--text-disabled)', bg: 'var(--page-bg)' },
  };
  const mc = moodColors[mood] ?? moodColors.kalm;

  return (
    <div className="card" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--green)' }} />
        <span style={{ fontSize: 20, fontWeight: 600 }}>Lucy</span>
        <span style={{ fontSize: 12, color: mc.color, background: mc.bg, padding: '3px 10px', borderRadius: 20, fontWeight: 600 }}>{mood}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20, fontSize: 13, color: 'var(--text-secondary)' }}>
        <span>tick <strong className="mono" style={{ color: 'var(--text-primary)' }}>{(t.tick ?? 0).toLocaleString()}</strong></span>
        <span style={{ padding: '3px 10px', borderRadius: 8, fontSize: 12, fontWeight: 600,
          color: t.phase === 'Idle' ? 'var(--green)' : 'var(--blue)',
          background: t.phase === 'Idle' ? 'var(--green-bg)' : 'var(--blue-bg)',
        }}>{t.phase}</span>
        <span><strong className="mono" style={{ color: 'var(--text-primary)' }}>{(t.points ?? 0).toLocaleString()}</strong> nodes</span>
        <span>vocab <strong className="mono" style={{ color: 'var(--blue)' }}>{(activity?.exploring?.tokens_processed ?? 0).toLocaleString()}</strong> tokens</span>
      </div>
    </div>
  );
}
