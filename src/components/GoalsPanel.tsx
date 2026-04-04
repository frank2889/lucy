// GoalsPanel — Active goals with progress
import { useStore } from '../store';

export default function GoalsPanel() {
  const goals = useStore((s) => s.telemetry?.goals) ?? [];
  if (goals.length === 0) return null;

  return (
    <div className="card fade-in">
      <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider mb-3">Active Goals</h3>
      <div className="flex flex-col gap-3">
        {goals.map((goal, i) => (
          <div key={i}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium text-[--text] truncate">{goal.description}</span>
              <span className="text-xs mono text-[--purple] font-semibold">{(goal.progress * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 rounded-full bg-[--bg] overflow-hidden">
              <div className="h-full rounded-full bg-[--green] transition-all duration-700"
                style={{ width: `${Math.max(Math.min(goal.progress * 100, 100), goal.progress > 0 ? 2 : 0)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
