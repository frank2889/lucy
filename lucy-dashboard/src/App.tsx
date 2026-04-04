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
    fetchTelemetry(); fetchStatus(); fetchActivity(); fetchJournal();
    const a = setInterval(fetchTelemetry, 2000);
    const b = setInterval(fetchStatus, 5000);
    const c = setInterval(fetchActivity, 10000);
    const d = setInterval(fetchJournal, 15000);
    return () => { clearInterval(a); clearInterval(b); clearInterval(c); clearInterval(d); };
  }, [fetchTelemetry, fetchStatus, fetchActivity, fetchJournal]);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 20px' }}>
      <StatusBar />
      <div style={{ height: 20 }} />
      <EmotionsPanel />
      <div style={{ height: 20 }} />
      <FieldStats />
      <div style={{ height: 20 }} />
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        <TrendsPanel />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <GoalsPanel />
          <ExplorationPanel />
        </div>
      </div>
      <div style={{ height: 20 }} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <JournalPanel />
        <ChatPanel />
      </div>
    </div>
  );
}
