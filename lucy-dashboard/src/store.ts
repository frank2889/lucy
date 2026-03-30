// Lucy Dashboard — Global state store
import { create } from 'zustand';
import type { Telemetry, Topology, JournalData, AskResponse } from './types';
import { api } from './api';

interface LucyStore {
  // Data
  telemetry: Telemetry | null;
  topology: Topology | null;
  journal: JournalData | null;
  chatHistory: { role: 'user' | 'lucy'; text: string; tick?: number }[];

  // Telemetry history (last 120 samples for sparklines)
  telemetryHistory: Telemetry[];

  // UI state
  selectedNode: number | null;
  setSelectedNode: (id: number | null) => void;

  // Actions
  fetchTelemetry: () => Promise<void>;
  fetchTopology: () => Promise<void>;
  fetchJournal: () => Promise<void>;
  askLucy: (question: string) => Promise<AskResponse>;
}

export const useStore = create<LucyStore>((set, get) => ({
  telemetry: null,
  topology: null,
  journal: null,
  chatHistory: [],
  telemetryHistory: [],
  selectedNode: null,

  setSelectedNode: (id) => set({ selectedNode: id }),

  fetchTelemetry: async () => {
    try {
      const t = await api.telemetry();
      set((s) => ({
        telemetry: t,
        telemetryHistory: [...s.telemetryHistory.slice(-119), t],
      }));
    } catch (e) {
      console.warn('telemetry fetch failed:', e);
    }
  },

  fetchTopology: async () => {
    try {
      const topo = await api.topology();
      set({ topology: topo });
    } catch (e) {
      console.warn('topology fetch failed:', e);
    }
  },

  fetchJournal: async () => {
    try {
      const j = await api.journal();
      set({ journal: j });
    } catch (e) {
      console.warn('journal fetch failed:', e);
    }
  },

  askLucy: async (question: string) => {
    set((s) => ({
      chatHistory: [...s.chatHistory, { role: 'user', text: question }],
    }));
    try {
      const resp = await api.ask(question);
      set((s) => ({
        chatHistory: [
          ...s.chatHistory,
          { role: 'lucy', text: resp.answer, tick: get().telemetry?.tick },
        ],
      }));
      return resp;
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : 'Unknown error';
      set((s) => ({
        chatHistory: [...s.chatHistory, { role: 'lucy', text: `[Error: ${errMsg}]` }],
      }));
      throw e;
    }
  },
}));
