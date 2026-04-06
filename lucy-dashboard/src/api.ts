// Lucy API client
import type { Telemetry, Topology, JournalData, AskResponse, LucyStatus, LucyActivity } from './types';

const BASE = import.meta.env.DEV ? '/api' : '';

async function fetchWithRetry(input: RequestInfo, init?: RequestInit, retries = 2, timeoutMs = 10_000): Promise<Response> {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(input, { ...init, signal: AbortSignal.timeout(timeoutMs) });
      if (res.ok) return res;
      if (res.status >= 400 && res.status < 500) throw new Error(`${res.status} ${res.statusText}`);
    } catch (e) {
      if (i === retries - 1) throw e;
    }
    await new Promise(r => setTimeout(r, 1000 * 2 ** i));
  }
  throw new Error('Request failed after retries');
}

async function get<T>(path: string): Promise<T> {
  const res = await fetchWithRetry(`${BASE}${path}`);
  return res.json();
}

export const api = {
  telemetry: () => get<Telemetry>('/telemetry'),
  topology: () => get<Topology>('/topology'),
  journal: () => get<JournalData>('/journal'),
  // /ask duurt 30-60s (3 denk-rondes × 50 settle ticks) — lange timeout, geen retries
  ask: (question: string) =>
    fetch(`${BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: question,
      signal: AbortSignal.timeout(90_000),
    }).then(r => r.json()) as Promise<AskResponse>,
  status: () => get<LucyStatus>('/status'),
  activity: () => get<LucyActivity>('/activity'),
};
