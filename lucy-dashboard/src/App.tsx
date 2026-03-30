// Lucy Brain Dashboard — Main App
import { useEffect } from 'react';
import { useStore } from './store';
import { updateAudio } from './audio';
import BrainRenderer from './components/BrainRenderer';
import StatusBar from './components/StatusBar';
import EmotionsPanel from './components/EmotionsPanel';
import JournalPanel from './components/JournalPanel';
import ChatPanel from './components/ChatPanel';
import GoalsPanel from './components/GoalsPanel';
import Sparkline from './components/Sparkline';
import AudioToggle from './components/AudioToggle';

export default function App() {
  const fetchTelemetry = useStore((s) => s.fetchTelemetry);
  const fetchTopology = useStore((s) => s.fetchTopology);
  const fetchJournal = useStore((s) => s.fetchJournal);
  const telemetry = useStore((s) => s.telemetry);

  // Polling loops
  useEffect(() => {
    // Initial fetch
    fetchTelemetry();
    fetchTopology();
    fetchJournal();

    // Telemetry: every 2s
    const tInt = setInterval(fetchTelemetry, 2000);
    // Topology: every 10s (heavier)
    const topoInt = setInterval(fetchTopology, 10000);
    // Journal: every 15s
    const jInt = setInterval(fetchJournal, 15000);

    return () => {
      clearInterval(tInt);
      clearInterval(topoInt);
      clearInterval(jInt);
    };
  }, [fetchTelemetry, fetchTopology, fetchJournal]);

  // Update audio sonification when telemetry changes
  useEffect(() => {
    if (!telemetry) return;
    updateAudio({
      curiosity: telemetry.emotions.curiosity,
      satisfaction: telemetry.emotions.satisfaction,
      frustration: telemetry.emotions.frustration,
      dopamine: telemetry.neuro.dopamine,
      serotonin: telemetry.neuro.serotonin,
      phi: telemetry.consciousness.phi,
      tension: telemetry.tension,
    });
  }, [telemetry]);

  return (
    <div className="w-full h-full flex flex-col">
      {/* Top status bar */}
      <div className="shrink-0 p-2">
        <StatusBar />
      </div>

      {/* Main content: 3D brain center, panels on sides */}
      <div className="flex-1 flex min-h-0 p-2 gap-2">
        {/* Left sidebar: emotions + goals + sparklines */}
        <div className="w-72 shrink-0 flex flex-col gap-2 overflow-y-auto">
          <EmotionsPanel />
          <GoalsPanel />

          {/* Sparklines */}
          <div className="glass p-3 flex flex-col gap-1">
            <h3 className="text-xs uppercase tracking-widest text-[--text-secondary] mb-1">Trends</h3>
            <Sparkline label="curiosity" getValue={(t) => t.emotions.curiosity} color="var(--accent-cyan)" />
            <Sparkline label="satisfaction" getValue={(t) => t.emotions.satisfaction} color="var(--accent-green)" />
            <Sparkline label="frustration" getValue={(t) => t.emotions.frustration} color="var(--accent-red)" />
            <Sparkline label="tension" getValue={(t) => t.tension} color="var(--accent-orange)" />
            <Sparkline label="dopamine" getValue={(t) => t.neuro.dopamine} color="#ffaa00" />
            <Sparkline label="Φ (phi)" getValue={(t) => t.consciousness.phi} color="var(--accent-purple)" />
            <Sparkline label="coherence" getValue={(t) => t.consciousness.coherence} color="var(--accent-blue)" />
          </div>
        </div>

        {/* Center: 3D Brain */}
        <div className="flex-1 relative rounded-lg overflow-hidden border border-[--border]">
          <BrainRenderer />
          {/* Audio toggle overlay */}
          <div className="absolute top-2 right-2 z-10">
            <AudioToggle />
          </div>
          {/* Legend overlay */}
          <div className="absolute bottom-2 left-2 z-10 glass px-3 py-2 text-[10px] flex gap-3 opacity-70">
            <span><span className="inline-block w-2 h-2 rounded-full bg-[#00ddff] mr-1" />Input</span>
            <span><span className="inline-block w-2 h-2 rounded-full bg-[#4466ff] mr-1" />Processing</span>
            <span><span className="inline-block w-2 h-2 rounded-full bg-[#aa44ff] mr-1" />Abstract</span>
            <span><span className="inline-block w-2 h-2 rounded-full bg-[#ff44aa] mr-1" />Meta</span>
          </div>
        </div>

        {/* Right sidebar: journal + chat */}
        <div className="w-80 shrink-0 flex flex-col gap-2 min-h-0">
          <div className="flex-1 min-h-0">
            <JournalPanel />
          </div>
          <div className="flex-1 min-h-0">
            <ChatPanel />
          </div>
        </div>
      </div>
    </div>
  );
}
