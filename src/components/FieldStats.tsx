// FieldStats — Key metrics as big number cards
import { useStore } from '../store';

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="card flex flex-col gap-1 items-center text-center py-4">
      <span className="text-2xl font-bold mono" style={color ? { color } : undefined}>{value}</span>
      <span className="text-xs text-[--text2] font-medium">{label}</span>
    </div>
  );
}

export default function FieldStats() {
  const status = useStore((s) => s.status);
  const t = useStore((s) => s.telemetry);
  if (!status && !t) return null;

  return (
    <div className="grid grid-cols-6 gap-3 fade-in">
      <Stat label="Points" value={status?.field?.points?.toLocaleString() ?? t?.points?.toLocaleString() ?? '-'} />
      <Stat label="Couplings" value={status?.field?.couplings?.toLocaleString() ?? '-'} />
      <Stat label="Vocabulary" value={status?.vocabulary?.size?.toLocaleString() ?? '-'} color="var(--blue)" />
      <Stat label="Pages Read" value={status?.explorer?.pages_read?.toString() ?? '-'} color="var(--green)" />
      <Stat label="Tokens" value={status?.explorer?.tokens_processed?.toLocaleString() ?? '-'} />
      <Stat label="Reflections" value={status?.self_modification?.reflection_count?.toString() ?? '-'} color="var(--purple)" />
    </div>
  );
}
