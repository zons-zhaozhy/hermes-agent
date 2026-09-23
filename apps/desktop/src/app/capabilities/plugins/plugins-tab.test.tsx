import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { type ComponentProps, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $notifications } from '@/store/notifications'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { PageSearchShell } from '../../page-search-shell'

import { CapabilityTabs } from './capability-tabs'
import { parseCatalog } from './catalog-data'
import { PluginActions, PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async () => ({ plugins: $agentPlugins.get() }))

// SkillsView owns navigation and search; exercise that controlled contract
// with the same primitives instead of giving PluginsTab private controls.
function PluginsHarness({
  view: initialView = 'installed',
  query: initialQuery = '',
  ...props
}: ComponentProps<typeof PluginsTab>) {
  const [view, setView] = useState(initialView)
  const [query, setQuery] = useState(initialQuery)

  return (
    <PageSearchShell onSearchChange={setQuery} searchPlaceholder="Search plugins" searchValue={query}>
      <CapabilityTabs actions={<PluginActions profile={props.profile} />} onChange={setView} value={view} />
      <PluginsTab {...props} onQueryChange={setQuery} query={query} view={view} />
    </PageSearchShell>
  )
}

function renderPlugins(props: ComponentProps<typeof PluginsTab>) {
  return render(
    <QueryClientProvider client={queryClient}>
      <PluginsHarness {...props} />
    </QueryClientProvider>
  )
}

const weatherEntry = {
  name: 'weather-plugin',
  repo: 'https://github.com/example/weather-plugin',
  sha: 'a'.repeat(40),
  subdir: '',
  tier: 'community',
  category: 'weather',
  description: 'Local weather forecasts'
}

function seedCatalog(entries = [weatherEntry]) {
  queryClient.setQueryData(['public-catalog', 'plugins'], parseCatalog('plugins', entries))
}

// The catalog picker is now an embedded iframe (docs site) that posts
// { type: 'hermes-plugin-pick', name, repo, sha?, subdir? } to the parent,
// which routes it through openCatalogPluginInstall. Drive that real link.
async function pickCatalogEntry(entry: { name: string, repo: string, sha?: string, subdir?: string }) {
  fireEvent(
    window,
    new MessageEvent('message', {
      data: { type: 'hermes-plugin-pick', ...entry },
      origin: 'https://hermes-agent.nousresearch.com'
    })
  )
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.unstubAllGlobals()
})

const connectionFixture = {
  baseUrl: 'http://localhost',
  isFullscreen: false,
  logs: [],
  nativeOverlayWidth: 0,
  token: '',
  windowButtonPosition: null,
  wsUrl: ''
}

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

const uninstallDiskPlugin = vi.fn(async (_id: string) => ({ ok: true }))

vi.mock('@/contrib/runtime-loader', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  uninstallDiskPlugin: (id: string) => uninstallDiskPlugin(id)
}))

describe('PluginsTab', () => {
  beforeEach(() => {
    $pluginRecords.set({})
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    requestGateway.mockReset()
    requestGateway.mockImplementation(async () => ({ plugins: $agentPlugins.get() }))
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
  })

  it('renders declared server pills and the unavailable sentence under the description', () => {
    $agentPlugins.set([
      {
        description: 'A test plugin',
        key: 'demo-plugin',
        name: 'demo-plugin',
        servers: [
          { name: 'ready-server', sentence: '', state: 'connected' },
          {
            name: 'setup-server',
            sentence: 'Example App is not installed. Install Example App, then try again.',
            state: 'missing_app'
          }
        ],
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])

    renderPlugins({ profile: 'workbot' })

    expect(screen.getByTestId('server-pill-ready-server').className).toContain('bg-emerald-500/10')
    expect(screen.getByTestId('server-pill-setup-server').className).toContain('bg-destructive/10')
    expect(screen.getByText('Example App is not installed. Install Example App, then try again.')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Agent: demo-plugin' }).getAttribute('aria-checked')).toBe('true')
  })

  it('hides bundled plugins (managed from their own surfaces)', () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'image_gen/fal',
        name: 'fal',
        source: 'bundled',
        status: 'enabled',
        version: ''
      }
    ])

    renderPlugins({ profile: null })

    expect(screen.queryByText('fal')).toBeNull()
    expect(screen.queryByRole('row')).toBeNull()
    // The empty state renders emptyAll + emptyHint as sibling text nodes in one <p>.
    expect(screen.getByText((_, element) => element?.textContent === 'No plugins yet. Browse the catalog below and install a reviewed plugin with one click.')).toBeTruthy()
  })

  // A desktop half can only be copied out of a backend that runs on THIS
  // machine; against a remote one the reconcile is a structural no-op, so the
  // row must say so instead of pending forever (#114079).
  it('marks a remote-backend desktop half unavailable instead of forever copying', () => {
    $connection.set({ ...connectionFixture, mode: 'remote' })
    $agentPlugins.set([
      {
        description: '',
        has_desktop_half: true,
        key: 'nous-prices',
        name: 'nous-prices',
        source: 'catalog',
        status: 'enabled',
        version: '1'
      }
    ])

    render(<PluginsTab profile={null} />)

    const detail = within(screen.getByRole('row', { name: /^nous-prices/ }))
    expect(detail.getByText('unavailable (remote backend)')).toBeTruthy()
    expect(detail.queryByText('copying…')).toBeNull()
  })

  it('keeps the pending desktop-half state on a local backend', () => {
    $agentPlugins.set([
      {
        description: '',
        has_desktop_half: true,
        key: 'nous-prices',
        name: 'nous-prices',
        source: 'catalog',
        status: 'enabled',
        version: '1'
      }
    ])

    render(<PluginsTab profile={null} />)

    const detail = within(screen.getByRole('row', { name: /^nous-prices/ }))
    expect(detail.getByText('copying…')).toBeTruthy()
    expect(detail.queryByText('unavailable (remote backend)')).toBeNull()
  })

  it('renders a unified package as ONE row with a Desktop switch and an Agent switch', () => {
    $pluginRecords.set({
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })
    $agentPlugins.set([
      {
        description: '',
        key: 'hermes-media-studio',
        name: 'hermes-media-studio',
        source: 'git',
        status: 'disabled',
        version: '1'
      }
    ])

    renderPlugins({ profile: 'workbot', scopeLabel: 'workbot' })

    expect(screen.getAllByRole('row', { name: /^Media Studio/ })).toHaveLength(1)
    expect(screen.getAllByRole('row')).toHaveLength(2)
    const detail = within(screen.getByRole('row', { name: /^Media Studio/ }))
    expect(detail.getByText('Agent + Desktop')).toBeTruthy()
    expect(detail.getByRole('switch', { name: 'Desktop: Media Studio' }).getAttribute('aria-checked')).toBe('true')
    expect(detail.getByRole('switch', { name: 'Agent: Media Studio' }).getAttribute('aria-checked')).toBe('false')
    expect(detail.getByRole('cell', { name: 'Agent in workbot' })).toBeTruthy()
  })

  it('offers "Install here" for a desktop half whose agent half is not in the selected profile', async () => {
    $pluginRecords.set({
      media: {
        id: 'media',
        name: 'Media Studio',
        kind: 'disk',
        status: 'loaded',
        packageName: 'hermes-media-studio',
        packageOrigin: { repo: 'https://github.com/NousResearch/hermes-media-studio.git', sha: 'abc' }
      }
    })

    renderPlugins({ profile: 'workbot', scopeLabel: 'workbot' })

    expect(screen.queryByRole('switch', { name: /^Agent:/ })).toBeNull()
    screen.getByRole('button', { name: 'Install here' }).click()
    // Pre-filled from the package marker: repo + pinned sha, agent half only.
    await waitFor(() => {
      expect($pluginInstallRequest.get()).toMatchObject({
        legacyHint: 'agent',
        profile: 'workbot',
        repo: 'https://github.com/NousResearch/hermes-media-studio.git',
        sha: 'abc'
      })
    })
  })

  it('disables "Install here" when the package has no known origin (hand-copied folder)', () => {
    $pluginRecords.set({
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })

    renderPlugins({ profile: 'workbot', scopeLabel: 'workbot' })

    expect((screen.getByRole('button', { name: 'Install here' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('loads the plugin list scoped to the selected profile', () => {
    renderPlugins({ profile: 'workbot' })

    expect(requestGateway).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'workbot' })
    )
  })

  it('opens the install modal for a picker selection in the scoped profile', async () => {
    renderPlugins({ profile: 'workbot' })

    expect($pluginInstallRequest.get()).toBeNull()
    await pickCatalogEntry({ name: weatherEntry.name, repo: weatherEntry.repo, sha: weatherEntry.sha })

    await waitFor(() =>
      expect($pluginInstallRequest.get()).toMatchObject({
        catalogName: weatherEntry.name,
        repo: weatherEntry.repo,
        profile: 'workbot',
        sha: weatherEntry.sha
      })
    )
  })

  it('toggles by canonical key through plugins.manage', async () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'image_gen/legacy',
        name: 'Legacy plugin',
        source: 'user',
        status: 'disabled',
        version: '0.20.0'
      }
    ])
    requestGateway.mockResolvedValueOnce({
      ok: true,
      plugin: { key: 'image_gen/legacy', name: 'Legacy plugin', status: 'enabled' }
    } as never)

    renderPlugins({ profile: null })

    screen.getByRole('switch', { name: 'Agent: Legacy plugin' }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'toggle', key: 'image_gen/legacy', enable: true })
      )
    )
  })

  it('renders keyless rows read-only (no name-addressed toggle RPC)', () => {
    // Name-addressed toggles flip every same-named plugin across category
    // dirs — pre-contract-v6 rows must never reach the RPC.
    $agentPlugins.set([
      {
        description: 'Returned by a pre-key backend',
        name: 'Legacy plugin',
        source: 'user',
        status: 'disabled',
        version: '0.20.0'
      }
    ])

    renderPlugins({ profile: null })

    const toggle = screen.getByRole('switch', { name: 'Agent: Legacy plugin' })

    expect(toggle.hasAttribute('disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true)

    toggle.click()

    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'toggle' }))
  })

  it('appends the picked subdir for multi-plugin repos without changing its catalog pin', async () => {
    const entry = {
      ...weatherEntry,
      name: 'nested-plugin',
      repo: 'https://github.com/example/plugins-monorepo',
      subdir: 'packages/nested-plugin'
    }
    renderPlugins({ profile: null })

    await pickCatalogEntry(entry)

    await waitFor(() =>
      expect($pluginInstallRequest.get()).toMatchObject({
        catalogName: entry.name,
        repo: `${entry.repo}#${entry.subdir}`,
        profile: null,
        sha: entry.sha
      })
    )
  })
})

describe('PluginsTab catalog UX', () => {
  beforeEach(() => {
    $agentPlugins.set([])
    $agentPluginsStatus.set('ready')
    closePluginInstallRequest()
    $notifications.set([])
    requestGateway.mockReset()
    requestGateway.mockImplementation(async () => ({ plugins: $agentPlugins.get() }))
    $pluginRecords.set({})
  })

  it('ignores catalog picks from a foreign origin and picks without a repo', async () => {
    renderPlugins({ profile: 'workbot' })

    // Foreign origin: the picker iframe is cross-origin; only the trusted
    // catalog origin's messages may open an install dialog.
    fireEvent(
      window,
      new MessageEvent('message', {
        data: { type: 'hermes-plugin-pick', name: 'evil', repo: 'https://evil.example.com/x.git' },
        origin: 'https://evil.example.com'
      })
    )
    // Trusted origin but incomplete payload (no repo): dropped.
    fireEvent(
      window,
      new MessageEvent('message', {
        data: { type: 'hermes-plugin-pick', name: 'no-repo' },
        origin: 'https://hermes-agent.nousresearch.com'
      })
    )
    expect($pluginInstallRequest.get()).toBeNull()
  })

  it('shows an Update chip when the catalog pin moved past the installed SHA', () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        catalog_sha: 'b'.repeat(40),
        catalog_tier: 'community',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: true,
        version: '1.0.0'
      }
    ])

    renderPlugins({ profile: null })

    expect(screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` })).toBeTruthy()
  })

  it('re-pins through plugins.manage update when the chip is clicked', async () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        catalog_sha: 'b'.repeat(40),
        catalog_tier: 'community',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: true,
        version: '1.0.0'
      }
    ])
    requestGateway.mockResolvedValue({ ok: true, unchanged: false, plugins: [] } as never)

    renderPlugins({ profile: 'workbot' })

    screen.getByRole('button', { name: `Update to ${'b'.repeat(8)}` }).click()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'update', name: 'demo-weather', profile: 'workbot' })
      )
    )
  })

  it('uninstalls through plugins.manage remove only after the confirm dialog is accepted', async () => {
    $agentPlugins.set([
      {
        description: '',
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        version: '1.0.0'
      }
    ])
    requestGateway.mockResolvedValue({ ok: true, name: 'demo-weather', plugins: [] } as never)

    render(<PluginsTab profile="workbot" />)

    screen.getByRole('button', { name: 'Uninstall: demo-weather' }).click()

    // The click only asks; nothing is deleted until the destructive confirm is answered.
    await waitFor(() => expect($confirmRequest.get()?.title).toBe('Uninstall demo-weather?'))
    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'remove' }))

    settleConfirm(true)

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'remove', name: 'demo-weather', profile: 'workbot' })
      )
    )
    await waitFor(() => expect(screen.queryByText('demo-weather')).toBeNull())
  })

  it('uninstalls a standalone desktop plugin through Electron only after the confirm dialog is accepted', async () => {
    $pluginRecords.set({
      clock: { id: 'clock', name: 'Clock', kind: 'disk', status: 'loaded', file: '/h/desktop-plugins/clock/plugin.js' }
    })
    uninstallDiskPlugin.mockClear()

    render(<PluginsTab profile={null} />)

    screen.getByRole('button', { name: 'Uninstall: Clock' }).click()

    await waitFor(() => expect($confirmRequest.get()?.title).toBe('Uninstall Clock?'))
    expect(uninstallDiskPlugin).not.toHaveBeenCalled()

    settleConfirm(true)

    await waitFor(() => expect(uninstallDiskPlugin).toHaveBeenCalledWith('clock'))
    // Nothing goes over the gateway: this half lives in this app, not the profile.
    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'remove' }))
  })

  it('offers no desktop Uninstall for a bundled plugin or a unified package half', () => {
    $pluginRecords.set({
      bots: { id: 'bots', name: 'Bot Mode', kind: 'bundled', status: 'loaded' },
      media: { id: 'media', name: 'Media Studio', kind: 'disk', status: 'loaded', packageName: 'hermes-media-studio' }
    })

    render(<PluginsTab profile={null} />)

    expect(screen.getByText('Bot Mode')).toBeTruthy()
    expect(screen.getByText('Media Studio')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^Uninstall:/ })).toBeNull()
  })

  it('offers no Uninstall for a pip-installed (entrypoint) agent plugin', () => {
    $agentPlugins.set([
      { description: '', key: 'demo-tool', name: 'demo-tool', source: 'entrypoint', status: 'enabled', version: '' }
    ])

    render(<PluginsTab profile={null} />)

    expect(screen.getByText('demo-tool')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Uninstall: demo-tool' })).toBeNull()
  })

  it('refuses a catalog pick that is already installed and current', async () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: false,
        version: '1.0.0'
      }
    ])

    await act(async () => { renderPlugins({ profile: null }) })
    // The already-installed short-circuit lives in the shared helper: a pick
    // whose installed row is current resolves to a success toast, no dialog.
    await pickCatalogEntry({ name: 'demo-weather', repo: weatherEntry.repo, sha: weatherEntry.sha })
    expect($pluginInstallRequest.get()).toBeNull()
    expect($notifications.get().some(n => n.kind === 'success')).toBe(true)
  })

  it('offers catalog installation for an installed entry when an update is available', async () => {
    $agentPlugins.set([
      {
        catalog_name: 'demo-weather',
        description: '',
        installed_sha: 'a'.repeat(40),
        key: 'demo-weather',
        name: 'demo-weather',
        source: 'git',
        status: 'enabled',
        update_available: true,
        version: '1.0.0'
      }
    ])

    const entry = { ...weatherEntry, name: 'demo-weather', sha: 'b'.repeat(40) }
    await act(async () => { renderPlugins({ profile: null }) })
    await pickCatalogEntry(entry)

    await waitFor(() =>
      expect($pluginInstallRequest.get()).toMatchObject({
        catalogName: entry.name,
        repo: entry.repo,
        profile: null,
        sha: entry.sha
      })
    )
  })
})
