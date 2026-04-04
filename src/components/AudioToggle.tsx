// AudioToggle — Button to enable/disable brain sonification
import { useState } from 'react';
import { startAudio, stopAudio, isAudioRunning } from '../audio';

export default function AudioToggle() {
  const [active, setActive] = useState(isAudioRunning());

  const toggle = () => {
    if (active) {
      stopAudio();
    } else {
      startAudio();
    }
    setActive(!active);
  };

  return (
    <button
      onClick={toggle}
      className={`px-2 py-1 text-xs rounded border transition-colors ${
        active
          ? 'bg-[--accent-cyan]/20 border-[--accent-cyan]/40 text-[--accent-cyan]'
          : 'bg-[--bg-secondary] border-[--border] text-[--text-secondary] hover:text-[--text-primary]'
      }`}
      title={active ? 'Mute brain audio' : 'Enable brain sonification'}
    >
      {active ? '♫ Audio On' : '♪ Audio Off'}
    </button>
  );
}
