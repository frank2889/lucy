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

export interface LucyStatus {
  alive: boolean;
  tick: number;
  phase: string;
  field: { points: number; active: number; couplings: number; avg_tension: string; input_ports: number; output_ports: number };
  vocabulary: { size: number };
  explorer: { frontier_size: number; pages_read: number; tokens_processed: number };
  replay_buffer: number;
  retrieval_store: number;
  self_modification: { recent: string[]; reflection_count: number; total_modifications: number };
  consciousness: Record<string, unknown>;
}

export interface LucyActivity {
  mood: string;
  emotions: Record<string, string>;
  neuromodulators: Record<string, string>;
  exploring: { current_interests: string[]; frontier_size: number; pages_read: number; tokens_processed: number };
  learning: { avg_tension: string; phase: string; teacher_interval: number; tick: number; vocabulary_size: number };
  recent_observations: Array<{ kind: string; message: string; tick: number }>;
  self_modifications: { reflection_count: number; total_modifications: number };
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
