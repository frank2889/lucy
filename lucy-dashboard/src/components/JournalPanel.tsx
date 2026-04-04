import { useStore } from '../store';

const KIND_COLORS: Record<string, string> = {
  milestone: 'var(--green)', thought: 'var(--blue)',
  satisfaction_spike: 'var(--green)', curiosity_spike: 'var(--cyan)',
  frustration_spike: 'var(--red)', error: 'var(--red)', goal: 'var(--purple)',
};

export default function JournalPanel() {
  const journal = useStore((s) => s.journal);

  if (!journal) return (
    <div className="card" style={{ padding: 20, textAlign: 'center', color: 'var(--text-disabled)' }}>Loading journal...</div>
  );

  const entries = journal.entries?.slice(-15).reverse() ?? [];

  return (
    <div className="card" style={{ padding: 20, display: 'flex', flexDirection: 'column', maxHeight: 400 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>Journal</span>
        <span className="mono" style={{ fontSize: 12, color: 'var(--text-disabled)' }}>{journal.total_entries ?? 0} entries</span>
      </div>

      {(journal.milestones ?? []).length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
          {journal.milestones.slice(0, 8).map((m) => (
            <span key={m} className="mono" style={{ fontSize: 11, padding: '2px 8px', borderRadius: 6, background: 'var(--purple-bg)', color: 'var(--purple)', border: '1px solid transparent' }}>{m}</span>
          ))}
        </div>
      )}

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {entries.map((entry, i) => {
          const color = KIND_COLORS[entry.kind] ?? 'var(--border)';
          return (
            <div key={i} style={{ paddingLeft: 12, borderLeft: `3px solid ${color}`, padding: '4px 0 4px 12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                <span className="mono" style={{ fontWeight: 600, color }}>t{entry.tick ?? '?'}</span>
                <span style={{ color: 'var(--text-disabled)' }}>{entry.kind ?? 'unknown'}</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2, lineHeight: 1.4 }}>{entry.message ?? ''}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
