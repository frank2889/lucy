import { useStore } from '../store';

export default function GoalsPanel() {
  const goals = useStore((s) => s.telemetry?.goals) ?? [];
  if (goals.length === 0) return null;

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16 }}>Active Goals</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {goals.map((g, i) => {
          const pct = Math.max(Math.min((g.progress ?? 0) * 100, 100), g.progress > 0 ? 2 : 0);
          return (
            <div key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 500 }}>{g.description ?? 'Unnamed'}</span>
                <span className="mono" style={{ fontSize: 12, color: 'var(--text-disabled)' }}>P{(g.priority ?? 0).toFixed(1)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--page-bg)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', borderRadius: 3, background: 'var(--green)', width: `${pct}%`, transition: 'width 0.7s ease' }} />
                </div>
                <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', width: 36, textAlign: 'right' }}>{pct.toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
