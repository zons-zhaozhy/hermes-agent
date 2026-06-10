/**
 * Process diagnostics for the /mem + /heapdump utility commands (Epic 3 port;
 * Ink ref `app/slash/commands/debug.ts` + `lib/memory.ts`). Pure formatters plus
 * the one impure seam (`performHeapdump` → `v8.writeHeapSnapshot`), kept out of
 * slash.ts so the dispatcher stays plain and tests can mock `node:v8`.
 */
import { mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { writeHeapSnapshot } from 'node:v8'

/** `123456789` → `117.7 MB` (binary units, one decimal above bytes). */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '0 B'
  if (n < 1024) return `${Math.round(n)} B`
  const units = ['KB', 'MB', 'GB', 'TB'] as const
  let v = n
  let i = -1
  do {
    v /= 1024
    i += 1
  } while (v >= 1024 && i < units.length - 1)
  return `${v.toFixed(1)} ${units[i]}`
}

/** Where heap snapshots land: `$HERMES_HOME`/`~/.hermes` + `logs/opentui-heap-<ts>.heapsnapshot`. */
export function heapSnapshotPath(now = new Date()): string {
  const home = process.env.HERMES_HOME?.trim() || join(homedir(), '.hermes')
  const ts = now.toISOString().replace(/[:.]/g, '-')
  return join(home, 'logs', `opentui-heap-${ts}.heapsnapshot`)
}

export interface HeapdumpResult {
  path: string
  before: { heapUsed: number; rss: number }
  after: { heapUsed: number; rss: number }
}

/**
 * Write a V8 heap snapshot (SYNCHRONOUS — blocks the event loop while V8 walks
 * the heap; that's inherent to writeHeapSnapshot) and report heap/rss before
 * vs after. Throws on I/O failure — the caller renders the error.
 */
export function performHeapdump(): HeapdumpResult {
  const before = process.memoryUsage()
  const path = heapSnapshotPath()
  mkdirSync(dirname(path), { recursive: true })
  const written = writeHeapSnapshot(path)
  const after = process.memoryUsage()
  return {
    after: { heapUsed: after.heapUsed, rss: after.rss },
    before: { heapUsed: before.heapUsed, rss: before.rss },
    path: written
  }
}

export interface MemSnapshot {
  heapUsed: number
  heapTotal: number
  external: number
  arrayBuffers: number
  rss: number
}

/**
 * The /mem system-line body (Ink's Memory panel as aligned rows). `renderables`
 * is the mounted-renderable count under the live renderer root (the store-cap
 * diagnostic) — omitted when unavailable (e.g. no renderer in tests).
 */
export function memReport(usage: MemSnapshot, uptimeSeconds: number, renderables?: number): string {
  const rows: Array<[string, string]> = [
    ['heap used', formatBytes(usage.heapUsed)],
    ['heap total', formatBytes(usage.heapTotal)],
    ['external', formatBytes(usage.external)],
    ['array buffers', formatBytes(usage.arrayBuffers)],
    ['rss', formatBytes(usage.rss)],
    ['uptime', `${Math.round(uptimeSeconds)}s`]
  ]
  if (renderables !== undefined) rows.push(['renderables', String(renderables)])
  const pad = Math.max(...rows.map(([k]) => k.length))
  return ['memory', ...rows.map(([k, v]) => `  ${k.padEnd(pad)}  ${v}`)].join('\n')
}
