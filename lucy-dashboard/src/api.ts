// Lucy API client
import type { Telemetry, Topology, JournalData, AskResponse, LucyStatus, LucyActivity } from './types';

const BASE = import.meta.env.DEV ? '/api' : '';

async function fetchWithRetry(input: RequestInfo, init?: RequestInit, retries = 3): Promise<Response> {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(input, { ...init, signal: AbortSignal.timeout(10_000) });
      if (res.ok) return res;
      if (res.status >= 400 && res.status < 500) throw new Error(`${res.status} ${res.statusText}`);
      // 5xx → retry
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

async function post<T>(path: string, body: string | object): Promise<T> {
  const isString = typeof body === 'string';
  const res = await fetchWithRetry(`${BASE}${path}`, {
    method: 'POST',
    headers: isString ? { 'Content-Type': 'text/plain' } : { 'Content-Type': 'application/json' },
    body: isString ? body : JSON.stringify(body),
  });
  return res.json();
}

export const api = {
  telemetry: () => get<Telemetry>('/telemetry'),
  topology: () => get<Topology>('/topology'),
  journal: () => get<JournalData>('/journal'),
  ask: (question: string) => post<AskResponse>('/ask', question),
  status: () => get<LucyStatus>('/status'),
  activity: () => get<LucyActivity>('/activity'),
};
