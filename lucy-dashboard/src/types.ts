// Lucy Brain Dashboard — Type definitions

export interface TopoNode {
  id: number;
  pos: [number, number, number];
  depth: number;
  active: boolean;
  energy: number;
  tension: number;
  curiosity: number;
  satisfaction: number;
  frustration: number;
  phi_integrated: number;
  oscillation_phase: number;
}

export interface TopoLink {
  s: number;
  t: number;
  w: number;
}

export interface Topology {
  tick: number;
  total_points: number;
  sampled: number;
  nodes: TopoNode[];
  links: TopoLink[];
}

export interface Emotions {
  curiosity: number;
  satisfaction: number;
  frustration: number;
}

export interface Neuro {
  dopamine: number;
  norepinephrine: number;
  serotonin: number;
  acetylcholine: number;
}

export interface Consciousness {
  spectral_radius: number;
  phi: number;
  coherence: number;
  free_energy: number;
  is_critical: boolean;
}

export interface Goal {
  description: string;
  priority: number;
  progress: number;
}

export interface Telemetry {
  tick: number;
  phase: string;
  points: number;
  active: number;
  tension: number;
  emotions: Emotions;
  neuro: Neuro;
  consciousness: Consciousness;
  goals: Goal[];
}

export interface JournalEntry {
  tick: number;
  kind: string;
  message: string;
}

export interface JournalData {
  entries: JournalEntry[];
  milestones: string[];
  total_entries: number;
}

export interface AskResponse {
  answer: string;
  source: string;
  settle_ticks: number;
  tension: string;
  coherence: string;
  think_rounds: number;
  internal?: {
    emotions?: Emotions;
    goals?: Goal[];
  };
}
