// ExplorationPanel — What Lucy is exploring
import { useStore } from '../store';

export default function ExplorationPanel() {
  const activity = useStore((s) => s.activity);
  if (!activity) return null;

  const interests = activity.exploring?.current_interests ?? [];

  return (
    <div className="card fade-in">
      <h3 className="text-xs font-semibold text-[--text2] uppercase tracking-wider mb-3">Exploration</h3>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="text-center">
          <div className="text-lg font-bold mono text-[--blue]">{(activity.exploring?.pages_read ?? 0).toLocaleString()}</div>
          <div className="text-xs text-[--text2]">Pages</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold mono">{(activity.exploring?.tokens_processed ?? 0).toLocaleString()}</div>
          <div className="text-xs text-[--text2]">Tokens</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold mono text-[--orange]">{(activity.exploring?.frontier_size ?? 0).toLocaleString()}</div>
          <div className="text-xs text-[--text2]">Frontier</div>
        </div>
      </div>

      {interests.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {interests.slice(0, 10).map((interest, i) => (
            <span key={i} className="text-xs px-2.5 py-1 rounded-full bg-[--bg] text-[--text2] font-medium">
              {interest}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
