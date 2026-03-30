// Plugin system — Load and validate Lucy's self-created dashboard plugins
// Plugins are loaded from /plugins endpoint (data/plugins/*.json on server)
// Each plugin manifest defines: name, type (visual|audio|panel), code (JS as string)

export interface PluginManifest {
  name: string;
  version: string;
  type: 'visual' | 'audio' | 'panel';
  description: string;
  // Inline JavaScript code (sandboxed execution)
  code: string;
  // Required telemetry fields this plugin reads
  inputs?: string[];
  // DOM element ID to mount to (for panel plugins)
  mountId?: string;
  // Auto-enable on load
  enabled?: boolean;
}

interface LoadedPlugin {
  manifest: PluginManifest;
  cleanup?: () => void;
}

const loadedPlugins: Map<string, LoadedPlugin> = new Map();
const pluginErrors: Map<string, string> = new Map();

// Validation: max code size, no dangerous patterns
const MAX_CODE_SIZE = 50_000; // 50KB max per plugin
const BLOCKED_PATTERNS = [
  /eval\s*\(/,
  /Function\s*\(/,
  /import\s*\(/,
  /require\s*\(/,
  /document\.cookie/,
  /localStorage/,
  /sessionStorage/,
  /window\.open/,
  /fetch\s*\(/,
  /XMLHttpRequest/,
];

function validatePlugin(manifest: PluginManifest): string | null {
  if (!manifest.name || typeof manifest.name !== 'string') return 'Missing plugin name';
  if (!manifest.type || !['visual', 'audio', 'panel'].includes(manifest.type)) return 'Invalid plugin type';
  if (!manifest.code || typeof manifest.code !== 'string') return 'Missing plugin code';
  if (manifest.code.length > MAX_CODE_SIZE) return `Code too large (${manifest.code.length} > ${MAX_CODE_SIZE})`;

  for (const pattern of BLOCKED_PATTERNS) {
    if (pattern.test(manifest.code)) return `Blocked pattern: ${pattern.source}`;
  }

  return null;
}

// Execute plugin code in a restricted scope
function executePlugin(manifest: PluginManifest, telemetryData: Record<string, unknown>): (() => void) | undefined {
  try {
    // Create a sandboxed function with limited globals
    const sandbox = {
      console: { log: console.log, warn: console.warn },
      Math,
      JSON,
      Date,
      // Telemetry data (read-only snapshot)
      lucy: Object.freeze({ ...telemetryData }),
      // DOM access limited to plugin mount point
      mount: manifest.mountId ? document.getElementById(manifest.mountId) : null,
    };

    const keys = Object.keys(sandbox);
    const values = Object.values(sandbox);

    // Create function with sandbox as scope
    const fn = new Function(...keys, `"use strict";\n${manifest.code}`);
    const result = fn(...values);

    // If the plugin returns a cleanup function, keep it
    if (typeof result === 'function') return result;
    return undefined;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    pluginErrors.set(manifest.name, msg);
    console.warn(`Plugin "${manifest.name}" error:`, msg);
    return undefined;
  }
}

export async function loadPlugins(): Promise<PluginManifest[]> {
  try {
    const res = await fetch(import.meta.env.DEV ? '/api/plugins' : '/plugins');
    if (!res.ok) return [];
    const data = await res.json();
    return data.plugins || [];
  } catch {
    return [];
  }
}

export function activatePlugin(manifest: PluginManifest, telemetryData: Record<string, unknown>): boolean {
  const error = validatePlugin(manifest);
  if (error) {
    pluginErrors.set(manifest.name, error);
    console.warn(`Plugin "${manifest.name}" validation failed:`, error);
    return false;
  }

  // Deactivate if already loaded
  deactivatePlugin(manifest.name);

  const cleanup = executePlugin(manifest, telemetryData);
  loadedPlugins.set(manifest.name, { manifest, cleanup });
  return true;
}

export function deactivatePlugin(name: string) {
  const plugin = loadedPlugins.get(name);
  if (plugin?.cleanup) {
    try { plugin.cleanup(); } catch { /* ignore cleanup errors */ }
  }
  loadedPlugins.delete(name);
}

export function getPluginErrors(): Map<string, string> {
  return new Map(pluginErrors);
}

export function getLoadedPlugins(): PluginManifest[] {
  return Array.from(loadedPlugins.values()).map((p) => p.manifest);
}
