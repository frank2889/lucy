import { useStore } from '../store';
import { useIsMobile } from '../hooks/useIsMobile';

function Stat({ label, value, color, compact }: { label: string; value: string; color?: string; compact?: boolean }) {
  return (
    <div className="card" style={{ padding: compact ? '10px 12px' : '16px 20px' }}>
      <div style={{ fontSize: compact ? 11 : 12, color: 'var(--text-secondary)', marginBottom: 2 }}>{label}</div>
      <div className="mono" style={{ fontSize: compact ? 16 : 22, fontWeight: 700, color: color ?? 'var(--text-primary)' }}>{value}</div>
    </div>
  );
}

export default function FieldStats() {
  const s = useStore((s) => s.status);
  const t = useStore((s) => s.telemetry);
  const mobile = useIsMobile();
  if (!s && !t) return null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: mobile ? 'repeat(3, 1fr)' : 'repeat(6, 1fr)', gap: mobile ? 8 : 16 }}>
      <Stat label="Points" value={s?.field?.points?.toLocaleString() ?? t?.points?.toLocaleString() ?? '-'} compact={mobile} />
      <Stat label="Couplings" value={s?.field?.couplings?.toLocaleString() ?? '-'} compact={mobile} />
      <Stat label="Vocab" value={s?.vocabulary?.size?.toLocaleString() ?? '-'} color="var(--blue)" compact={mobile} />
      <Stat label="Pages" value={s?.explorer?.pages_read?.toString() ?? '-'} color="var(--green)" compact={mobile} />
      <Stat label="Tokens" value={s?.explorer?.tokens_processed?.toLocaleString() ?? '-'} compact={mobile} />
      <Stat label="Reflect" value={s?.self_modification?.reflection_count?.toString() ?? '-'} color="var(--purple)" compact={mobile} />
    </div>
  );
}
