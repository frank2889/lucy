import { useStore } from '../store';

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="card" style={{ padding: '16px 20px' }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
      <div className="mono" style={{ fontSize: 22, fontWeight: 700, color: color ?? 'var(--text-primary)' }}>{value}</div>
    </div>
  );
}

export default function FieldStats() {
  const s = useStore((s) => s.status);
  const t = useStore((s) => s.telemetry);
  if (!s && !t) return null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
      <Stat label="Points" value={s?.field?.points?.toLocaleString() ?? t?.points?.toLocaleString() ?? '-'} />
      <Stat label="Couplings" value={s?.field?.couplings?.toLocaleString() ?? '-'} />
      <Stat label="Vocabulary" value={s?.vocabulary?.size?.toLocaleString() ?? '-'} color="var(--blue)" />
      <Stat label="Pages Read" value={s?.explorer?.pages_read?.toString() ?? '-'} color="var(--green)" />
      <Stat label="Tokens" value={s?.explorer?.tokens_processed?.toLocaleString() ?? '-'} />
      <Stat label="Reflections" value={s?.self_modification?.reflection_count?.toString() ?? '-'} color="var(--purple)" />
    </div>
  );
}
