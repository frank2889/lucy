// AudioEngine — Sonification of Lucy's brain activity
// Creates ambient tones based on emotions, neurotransmitters, and field state

let audioCtx: AudioContext | null = null;
let masterGain: GainNode | null = null;
let oscillators: OscillatorNode[] = [];
let gains: GainNode[] = [];
let noiseSource: AudioBufferSourceNode | null = null;
let noiseGain: GainNode | null = null;
let isRunning = false;

// Frequencies mapped to different brain signals
const FREQ = {
  curiosity: 440,       // A4 — bright, seeking
  satisfaction: 330,     // E4 — warm, resonant
  frustration: 220,      // A3 — low, tense
  dopamine: 523,         // C5 — reward ping
  serotonin: 392,        // G4 — calm
  phi: 660,              // E5 — consciousness
};

function createNoiseBuffer(ctx: AudioContext, duration: number): AudioBuffer {
  const sampleRate = ctx.sampleRate;
  const length = sampleRate * duration;
  const buffer = ctx.createBuffer(1, length, sampleRate);
  const data = buffer.getChannelData(0);
  // Brown noise (low frequency rumble)
  let last = 0;
  for (let i = 0; i < length; i++) {
    const white = Math.random() * 2 - 1;
    last = (last + 0.02 * white) / 1.02;
    data[i] = last * 3.5;
  }
  return buffer;
}

export function startAudio() {
  if (isRunning) return;

  audioCtx = new AudioContext();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.15; // Keep it ambient
  masterGain.connect(audioCtx.destination);

  // Create oscillators for each signal
  const keys = Object.keys(FREQ) as (keyof typeof FREQ)[];
  for (const key of keys) {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = key === 'frustration' ? 'sawtooth' : 'sine';
    osc.frequency.value = FREQ[key];
    gain.gain.value = 0;
    osc.connect(gain);
    gain.connect(masterGain);
    osc.start();
    oscillators.push(osc);
    gains.push(gain);
  }

  // Background noise (field activity)
  const noiseBuffer = createNoiseBuffer(audioCtx, 4);
  noiseSource = audioCtx.createBufferSource();
  noiseSource.buffer = noiseBuffer;
  noiseSource.loop = true;
  noiseGain = audioCtx.createGain();
  noiseGain.gain.value = 0.02;
  noiseSource.connect(noiseGain);
  noiseGain.connect(masterGain);
  noiseSource.start();

  isRunning = true;
}

export function stopAudio() {
  if (!isRunning) return;
  for (const osc of oscillators) {
    try { osc.stop(); } catch { /* already stopped */ }
  }
  try { noiseSource?.stop(); } catch { /* already stopped */ }
  audioCtx?.close();
  oscillators = [];
  gains = [];
  noiseSource = null;
  noiseGain = null;
  audioCtx = null;
  masterGain = null;
  isRunning = false;
}

export function isAudioRunning() {
  return isRunning;
}

interface AudioParams {
  curiosity: number;
  satisfaction: number;
  frustration: number;
  dopamine: number;
  serotonin: number;
  phi: number;
  tension: number;
}

export function updateAudio(params: AudioParams) {
  if (!isRunning || !audioCtx) return;
  const t = audioCtx.currentTime;
  const ramp = 0.3; // smooth transitions

  // Map each signal to its oscillator gain
  const values = [
    params.curiosity * 0.4,       // curiosity
    params.satisfaction * 0.3,     // satisfaction
    params.frustration * 0.25,     // frustration (slightly quieter)
    params.dopamine * 0.2,         // dopamine
    params.serotonin * 0.15,       // serotonin
    params.phi * 0.35,             // phi (consciousness)
  ];

  for (let i = 0; i < gains.length && i < values.length; i++) {
    gains[i].gain.linearRampToValueAtTime(values[i], t + ramp);
  }

  // Noise volume tracks tension
  if (noiseGain) {
    noiseGain.gain.linearRampToValueAtTime(params.tension * 0.1, t + ramp);
  }

  // Detune based on coherence — more coherent = more in-tune
  if (oscillators.length > 0) {
    const detune = (1 - Math.min(params.phi, 1)) * 15; // 0-15 cents detune
    for (const osc of oscillators) {
      osc.detune.linearRampToValueAtTime(detune, t + ramp);
    }
  }
}
