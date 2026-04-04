import { useStore } from '../store';

export default function ExplorationPanel() {
  const activity = useStore((s) => s.activity);
  if (!activity) return null;

  const interests = (activity.exploring?.current_interests ?? [])
    .filter((s) => /^[\x20-\x7E]+$/.test(s) && s.length < 30);

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16 }}>Exploration</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, textAlign: 'center', marginBottom: interests.length > 0 ? 16 : 0 }}>
        <div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--blue)' }}>{(activity.exploring?.pages_read ?? 0).toLocaleString()}</div>
          <div style={{ fontSize: 12, color: 'var(--text-disabled)' }}>Pages</div>
        </div>
        <div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700 }}>{(activity.exploring?.tokens_processed ?? 0).toLocaleString()}</div>
          <div style={{ fontSize: 12, color: 'var(--text-disabled)' }}>Tokens</div>
        </div>
        <div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--orange)' }}>{(activity.exploring?.frontier_size ?? 0).toLocaleString()}</div>
          <div style={{ fontSize: 12, color: 'var(--text-disabled)' }}>Frontier</div>
        </div>
      </div>
      {interests.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {interests.slice(0, 12).map((s, i) => (
            <span key={i} style={{ fontSize: 12, padding: '3px 10px', borderRadius: 6, background: 'var(--page-bg)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>{s}</span>
          ))}
        </div>
      )}
    </div>
  );
}
