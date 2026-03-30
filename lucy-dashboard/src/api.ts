// Lucy API client
import type { Telemetry, Topology, JournalData, AskResponse } from './types';

const BASE = import.meta.env.DEV ? '/api' : '';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body: string | object): Promise<T> {
  const isString = typeof body === 'string';
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: isString ? { 'Content-Type': 'text/plain' } : { 'Content-Type': 'application/json' },
    body: isString ? body : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  telemetry: () => get<Telemetry>('/telemetry'),
  topology: () => get<Topology>('/topology'),
  journal: () => get<JournalData>('/journal'),
  ask: (question: string) => post<AskResponse>('/ask', question),
  status: () => get<Record<string, unknown>>('/status'),
};
