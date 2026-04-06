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
  const t = useStore((s) => s.telemetry);
  const mobile = useIsMobile();
  if (!t) return null;

  // v2.5: Alles uit /telemetry (lock-free, altijd beschikbaar)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: mobile ? 'repeat(3, 1fr)' : 'repeat(6, 1fr)', gap: mobile ? 8 : 16 }}>
      <Stat label="Points" value={(t.points ?? 0).toLocaleString()} compact={mobile} />
      <Stat label="Couplings" value={(t as any).couplings?.toLocaleString() ?? '-'} compact={mobile} />
      <Stat label="Vocab" value={(t as any).vocab_size?.toLocaleString() ?? '-'} color="var(--blue)" compact={mobile} />
      <Stat label="Pages" value={(t as any).pages_read?.toLocaleString() ?? '-'} color="var(--green)" compact={mobile} />
      <Stat label="Tokens" value={(t as any).tokens_processed?.toLocaleString() ?? '-'} compact={mobile} />
      <Stat label="Reflect" value={(t as any).reflection_count?.toLocaleString() ?? '-'} color="var(--purple)" compact={mobile} />
    </div>
  );
}
