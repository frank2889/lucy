import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useStore } from '../store';

const LINES = [
  { key: 'curiosity', color: 'var(--cyan)', label: 'Curiosity' },
  { key: 'satisfaction', color: 'var(--green)', label: 'Satisfaction' },
  { key: 'frustration', color: 'var(--red)', label: 'Frustration' },
  { key: 'tension', color: 'var(--orange)', label: 'Tension' },
];

export default function TrendsPanel() {
  const history = useStore((s) => s.telemetryHistory);

  if (history.length < 3) {
    return (
      <div className="card" style={{ padding: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>Trends</div>
        <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-disabled)', fontSize: 13 }}>
          Collecting data...
        </div>
      </div>
    );
  }

  const data = history.map((t, i) => ({
    i,
    curiosity: t.emotions.curiosity,
    satisfaction: t.emotions.satisfaction,
    frustration: t.emotions.frustration,
    tension: t.tension,
  }));

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 16 }}>Emotions & Tension</div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <defs>
            {LINES.map(l => (
              <linearGradient key={l.key} id={`g-${l.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={l.color} stopOpacity={0.15} />
                <stop offset="100%" stopColor={l.color} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          <XAxis dataKey="i" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ background: '#fff', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12, boxShadow: 'var(--card-shadow)' }}
            labelStyle={{ display: 'none' }}
          />
          {LINES.map(l => (
            <Area key={l.key} type="monotone" dataKey={l.key} name={l.label}
              stroke={l.color} fill={`url(#g-${l.key})`} strokeWidth={2} dot={false} />
          ))}
        </AreaChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 20, marginTop: 12 }}>
        {LINES.map(l => (
          <div key={l.key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: l.color }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  );
}
