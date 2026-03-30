// JournalPanel — Lucy's journal entries and milestones
import { useStore } from '../store';

export default function JournalPanel() {
  const journal = useStore((s) => s.journal);
  if (!journal) return <div className="glass p-3 text-[--text-secondary] text-sm">Loading journal...</div>;

  return (
    <div className="glass p-3 flex flex-col gap-2 max-h-full overflow-hidden">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-widest text-[--text-secondary]">Journal</h3>
        <span className="text-xs text-[--text-secondary]">{journal.total_entries} entries</span>
      </div>

      {journal.milestones.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-1">
          {journal.milestones.map((m) => (
            <span key={m} className="text-[10px] px-1.5 py-0.5 rounded bg-[--accent-purple]/20 text-[--accent-purple] border border-[--accent-purple]/30">
              {m}
            </span>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-y-auto flex flex-col gap-1.5">
        {journal.entries.slice(-20).reverse().map((entry, i) => (
          <div key={i} className="text-xs border-l-2 border-[--border] pl-2 py-0.5">
            <div className="flex items-center gap-2">
              <span className="text-[--accent-blue] font-mono">t{entry.tick}</span>
              <span className="text-[--text-secondary]">{entry.kind}</span>
            </div>
            <p className="text-[--text-primary] mt-0.5 leading-relaxed">{entry.message}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
