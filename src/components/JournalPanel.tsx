// JournalPanel — Lucy's journal entries
import { useStore } from '../store';

const KIND_COLORS: Record<string, string> = {
  milestone: 'var(--green)',
  thought: 'var(--blue)',
  satisfaction_spike: 'var(--green)',
  curiosity_spike: 'var(--cyan)',
  frustration_spike: 'var(--red)',
  error: 'var(--red)',
  goal: 'var(--purple)',
};

export default function JournalPanel() {
  const journal = useStore((s) => s.journal);
  if (!journal) return (
    <div className="card text-[--text2] text-sm py-8 text-center">Loading journal...</div>
  );

  return (
    <div className="card fade-in flex flex-col gap-3 max-h-80 overflow-hidden">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider">Journal</h3>
        <span className="text-xs text-[--text2] mono">{journal.total_entries} entries</span>
      </div>

      {journal.milestones.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {journal.milestones.slice(0, 8).map((m) => (
            <span key={m} className="text-xs px-2.5 py-1 rounded-full bg-[--bg] text-[--purple] font-medium mono">
              {m}
            </span>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-y-auto flex flex-col gap-2">
        {journal.entries.slice(-15).reverse().map((entry, i) => {
          const color = KIND_COLORS[entry.kind] || 'var(--border)';
          return (
            <div key={i} className="border-l-3 pl-3 py-1" style={{ borderColor: color }}>
              <div className="flex items-center gap-2 text-xs">
                <span className="mono font-medium" style={{ color }}>t{entry.tick}</span>
                <span className="text-[--text2]">{entry.kind}</span>
              </div>
              <p className="text-sm text-[--text] opacity-70 mt-0.5 leading-relaxed">{entry.message}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
