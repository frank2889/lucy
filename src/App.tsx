// Lucy Dashboard — Apple Health inspired
import { useEffect } from 'react';
import { useStore } from './store';
import StatusBar from './components/StatusBar';
import EmotionsPanel from './components/EmotionsPanel';
import GoalsPanel from './components/GoalsPanel';
import JournalPanel from './components/JournalPanel';
import ChatPanel from './components/ChatPanel';
import FieldStats from './components/FieldStats';
import ExplorationPanel from './components/ExplorationPanel';
import TrendsPanel from './components/TrendsPanel';

export default function App() {
  const fetchTelemetry = useStore((s) => s.fetchTelemetry);
  const fetchJournal = useStore((s) => s.fetchJournal);
  const fetchStatus = useStore((s) => s.fetchStatus);
  const fetchActivity = useStore((s) => s.fetchActivity);

  useEffect(() => {
    fetchTelemetry();
    fetchStatus();
    fetchActivity();
    fetchJournal();

    const tInt = setInterval(fetchTelemetry, 2000);
    const sInt = setInterval(fetchStatus, 5000);
    const aInt = setInterval(fetchActivity, 10000);
    const jInt = setInterval(fetchJournal, 15000);

    return () => {
      clearInterval(tInt);
      clearInterval(sInt);
      clearInterval(aInt);
      clearInterval(jInt);
    };
  }, [fetchTelemetry, fetchStatus, fetchActivity, fetchJournal]);

  return (
    <div className="w-full min-h-screen">
      <div className="max-w-[1400px] mx-auto px-6 py-6 flex flex-col gap-5">
        <StatusBar />

        {/* Hero: Emotion Rings */}
        <EmotionsPanel />

        {/* Stats grid */}
        <FieldStats />

        {/* Two columns: trends + goals/exploration */}
        <div className="grid grid-cols-3 gap-5">
          <div className="col-span-2">
            <TrendsPanel />
          </div>
          <div className="flex flex-col gap-5">
            <GoalsPanel />
            <ExplorationPanel />
          </div>
        </div>

        {/* Bottom: Journal + Chat */}
        <div className="grid grid-cols-2 gap-5">
          <JournalPanel />
          <ChatPanel />
        </div>
      </div>
    </div>
  );
}
