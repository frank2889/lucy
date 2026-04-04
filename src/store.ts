// Lucy Dashboard — Global state store
import { create } from 'zustand';
import type { Telemetry, Topology, JournalData, AskResponse, LucyStatus, LucyActivity } from './types';
import { api } from './api';

interface LucyStore {
  telemetry: Telemetry | null;
  topology: Topology | null;
  journal: JournalData | null;
  status: LucyStatus | null;
  activity: LucyActivity | null;
  chatHistory: { role: 'user' | 'lucy'; text: string; tick?: number }[];
  telemetryHistory: Telemetry[];

  fetchTelemetry: () => Promise<void>;
  fetchTopology: () => Promise<void>;
  fetchJournal: () => Promise<void>;
  fetchStatus: () => Promise<void>;
  fetchActivity: () => Promise<void>;
  askLucy: (question: string) => Promise<AskResponse>;
}

export const useStore = create<LucyStore>((set, get) => ({
  telemetry: null,
  topology: null,
  journal: null,
  status: null,
  activity: null,
  chatHistory: [],
  telemetryHistory: [],

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

  fetchStatus: async () => {
    try {
      const s = await api.status();
      set({ status: s });
    } catch (e) {
      console.warn('status fetch failed:', e);
    }
  },

  fetchActivity: async () => {
    try {
      const a = await api.activity();
      set({ activity: a });
    } catch (e) {
      console.warn('activity fetch failed:', e);
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
