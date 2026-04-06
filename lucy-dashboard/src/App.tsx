import { useEffect } from 'react';
import { useStore } from './store';
import { useIsMobile } from './hooks/useIsMobile';
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
  const mobile = useIsMobile();

  useEffect(() => {
    fetchTelemetry(); fetchStatus(); fetchActivity(); fetchJournal();
    const a = setInterval(fetchTelemetry, 2000);
    const b = setInterval(fetchStatus, 5000);
    const c = setInterval(fetchActivity, 10000);
    const d = setInterval(fetchJournal, 15000);
    return () => { clearInterval(a); clearInterval(b); clearInterval(c); clearInterval(d); };
  }, [fetchTelemetry, fetchStatus, fetchActivity, fetchJournal]);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: mobile ? '12px 10px' : '24px 20px' }}>
      <StatusBar />
      <div style={{ height: mobile ? 12 : 20 }} />
      <EmotionsPanel />
      <div style={{ height: mobile ? 12 : 20 }} />
      <FieldStats />
      <div style={{ height: mobile ? 12 : 20 }} />
      <div style={{ display: 'grid', gridTemplateColumns: mobile ? '1fr' : '2fr 1fr', gap: mobile ? 12 : 20 }}>
        <TrendsPanel />
        <div style={{ display: 'flex', flexDirection: 'column', gap: mobile ? 12 : 20 }}>
          <GoalsPanel />
          <ExplorationPanel />
        </div>
      </div>
      <div style={{ height: mobile ? 12 : 20 }} />
      <div style={{ display: 'grid', gridTemplateColumns: mobile ? '1fr' : '1fr 1fr', gap: mobile ? 12 : 20 }}>
        <ChatPanel />
        <JournalPanel />
      </div>
    </div>
  );
}
