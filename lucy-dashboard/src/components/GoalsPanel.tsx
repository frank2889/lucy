// GoalsPanel — Lucy's active goals with progress bars
import { useStore } from '../store';

export default function GoalsPanel() {
  const goals = useStore((s) => s.telemetry?.goals) ?? [];

  if (goals.length === 0) return null;

  return (
    <div className="glass p-3 flex flex-col gap-2">
      <h3 className="text-xs uppercase tracking-widest text-[--text-secondary]">Active Goals</h3>
      {goals.map((goal, i) => (
        <div key={i} className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[--text-primary] truncate flex-1">{goal.description}</span>
            <span className="text-[--text-secondary] ml-2">
              p{goal.priority.toFixed(1)}
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-[--bg-secondary] overflow-hidden">
            <div
              className="h-full rounded-full bg-[--accent-green] transition-all duration-700"
              style={{ width: `${Math.min(goal.progress * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
