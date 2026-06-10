/**
 * /replay — spawn-tree inspector logic (Epic 3 port; Ink ref
 * `app/slash/commands/ops.ts` /replay + `spawnHistoryStore.ts`). The gateway
 * archives each completed delegation fan-out as a JSON snapshot
 * (`spawn_tree.save`); these helpers read `spawn_tree.list` / `spawn_tree.load`
 * payloads and format them as PAGER TEXT — the native engine renders replays
 * through the existing pager overlay instead of Ink's agents overlay.
 *
 * All readers are defensive (wire JSON is loose, snapshots cross versions).
 */

export interface SpawnTreeEntry {
  path: string
  label: string
  count: number
  /** Epoch SECONDS (gateway convention). */
  finishedAt?: number
  sessionId?: string
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' && v ? v : undefined
}

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

/** Map a `spawn_tree.list` result ({entries:[…]}) into typed rows (pathless rows dropped). */
export function readSpawnTreeEntries(result: unknown): SpawnTreeEntry[] {
  if (!result || typeof result !== 'object') return []
  const entries = (result as { entries?: unknown }).entries
  if (!Array.isArray(entries)) return []
  const out: SpawnTreeEntry[] = []
  for (const e of entries) {
    if (!e || typeof e !== 'object') continue
    const o = e as { [k: string]: unknown }
    const path = str(o['path'])
    if (!path) continue
    const entry: SpawnTreeEntry = {
      count: num(o['count']) ?? 0,
      label: str(o['label']) ?? '',
      path
    }
    const finishedAt = num(o['finished_at'])
    if (finishedAt !== undefined) entry.finishedAt = finishedAt
    const sessionId = str(o['session_id'])
    if (sessionId !== undefined) entry.sessionId = sessionId
    out.push(entry)
  }
  return out
}

function fmtWhen(epochSeconds: number | undefined): string {
  if (epochSeconds === undefined) return '?'
  try {
    return new Date(epochSeconds * 1000).toLocaleString()
  } catch {
    return '?'
  }
}

/** The bare `/replay` listing: indexed rows the user replays by number. */
export function formatSpawnTreeList(entries: readonly SpawnTreeEntry[]): string {
  const lines: string[] = ['Archived spawn trees — /replay <n> to view, /replay <path> for any snapshot', '']
  entries.forEach((e, i) => {
    const label = e.label || `${e.count} subagent${e.count === 1 ? '' : 's'}`
    lines.push(`${String(i + 1).padStart(3)}. ${fmtWhen(e.finishedAt)} · ${e.count}× — ${label}`)
    lines.push(`     ${e.path}`)
  })
  return lines.join('\n')
}

/** Status glyph for an archived subagent row. */
function statusGlyph(status: string): string {
  if (status === 'completed') return '✓'
  if (status === 'error' || status === 'failed' || status === 'timeout') return '✗'
  if (status === 'interrupted') return '⏹'
  return '●'
}

/** One archived subagent → its pager lines (indented by spawn depth). */
function subagentLines(raw: unknown, index: number): string[] {
  const o = (raw && typeof raw === 'object' ? raw : {}) as { [k: string]: unknown }
  const depth = num(o['depth']) ?? 0
  const pad = '  '.repeat(Math.max(0, depth))
  const status = str(o['status']) ?? 'completed'
  const goal = str(o['goal']) ?? 'subagent'
  const lines = [`${pad}${statusGlyph(status)} [${index + 1}] ${goal}`]
  const meta: string[] = [status]
  const model = str(o['model'])
  if (model) meta.push(model)
  const duration = num(o['durationSeconds'])
  if (duration !== undefined) meta.push(`${Math.round(duration)}s`)
  const tools = num(o['toolCount'])
  if (tools) meta.push(`${tools} tool${tools === 1 ? '' : 's'}`)
  const tokIn = num(o['inputTokens'])
  const tokOut = num(o['outputTokens'])
  if (tokIn !== undefined || tokOut !== undefined) meta.push(`${tokIn ?? 0} in / ${tokOut ?? 0} out tok`)
  lines.push(`${pad}    ${meta.join(' · ')}`)
  const summary = str(o['summary'])
  if (summary) for (const s of summary.split('\n')) lines.push(`${pad}    ${s}`)
  const notes = o['notes']
  if (Array.isArray(notes)) {
    for (const note of notes) if (typeof note === 'string' && note) lines.push(`${pad}    · ${note}`)
  }
  return lines
}

/** A loaded snapshot (`spawn_tree.load` payload) → the full pager text. */
export function formatSpawnTree(payload: unknown): string {
  const o = (payload && typeof payload === 'object' ? payload : {}) as { [k: string]: unknown }
  const subagents = Array.isArray(o['subagents']) ? (o['subagents'] as unknown[]) : []
  const header: string[] = []
  const label = str(o['label'])
  header.push(label ?? 'spawn tree')
  const meta: string[] = []
  const sessionId = str(o['session_id'])
  if (sessionId) meta.push(`session ${sessionId}`)
  meta.push(`finished ${fmtWhen(num(o['finished_at']))}`)
  meta.push(`${subagents.length} subagent${subagents.length === 1 ? '' : 's'}`)
  header.push(meta.join(' · '))
  if (!subagents.length) return [...header, '', '(snapshot empty or unreadable)'].join('\n')
  const body = subagents.flatMap((s, i) => ['', ...subagentLines(s, i)])
  return [...header, ...body].join('\n')
}
