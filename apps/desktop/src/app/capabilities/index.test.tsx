// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import type * as ReactRouterDom from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type * as HubActions from '@/store/hub-actions'

import { parseCatalog } from './plugins/catalog-data'
import { SkillCatalog } from './plugins/skill-catalog'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const setSkillEnabled = vi.fn()
const setToolsetEnabled = vi.fn()
const getToolsetConfig = vi.fn()
const selectToolsetProvider = vi.fn()
const getUsageAnalytics = vi.fn()
const getProfiles = vi.fn()
const getSkillContent = vi.fn()

// Partial mock: keep the real module (CapabilitiesView pulls in @/store/profile,
// whose import-time subscription calls setApiRequestProfile) and stub only the
// calls we assert on. Args are forwarded so the per-profile scope arg is
// observable.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkills: (profile?: null | string) => getSkills(profile),
  getToolsets: (profile?: null | string) => getToolsets(profile),
  setSkillEnabled: (name: string, enabled: boolean, profile?: null | string) => setSkillEnabled(name, enabled, profile),
  setToolsetEnabled: (name: string, enabled: boolean, profile?: null | string) =>
    setToolsetEnabled(name, enabled, profile),
  getToolsetConfig: (name: string, profile?: null | string) => getToolsetConfig(name, profile),
  selectToolsetProvider: (toolset: string, provider: string) => selectToolsetProvider(toolset, provider),
  getUsageAnalytics: (days: number, profile?: null | string) => getUsageAnalytics(days, profile),
  getProfiles: () => getProfiles(),
  getSkillContent: (name: string, profile?: null | string) => getSkillContent(name, profile)
}))

// Notifications hit nanostores/timers we don't care about here.
vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

// The catalog Install button routes through the hub action pipeline — stub the
// action entrypoint (real module kept: CapabilitiesView reads $hubActions and the
// query keys from it).
vi.mock('@/store/hub-actions', async importOriginal => ({
  ...(await importOriginal<typeof HubActions>()),
  installHubSkill: vi.fn().mockResolvedValue(undefined)
}))

// The vision detail navigates to Settings → Models via useNavigate; spy on it
// so the deep-link target is assertable.
const navigateSpy = vi.fn()

vi.mock('react-router', async importOriginal => ({
  ...(await importOriginal<typeof ReactRouterDom>()),
  useNavigate: () => navigateSpy
}))

// Import at module scope (after the hoisted vi.mock calls) so the heavy
// component-tree transform is paid during collection, not billed against the
// first test's testTimeout — same flake class as messaging/index.test.tsx.
const { CapabilitiesView } = await import('./index')

function toolset(overrides: Record<string, unknown> = {}) {
  return {
    name: 'web',
    label: 'Web Search',
    description: 'web_search, web_extract',
    enabled: true,
    available: true,
    configured: true,
    tools: ['web_search', 'web_extract'],
    ...overrides
  }
}

async function renderSkills() {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // CapabilitiesView reads skills/toolsets via useQuery, so it needs a provider.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/capabilities?tab=toolsets']}>
          <CapabilitiesView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  getSkills.mockResolvedValue([])
  getToolsets.mockResolvedValue([toolset()])
  setToolsetEnabled.mockResolvedValue({ ok: true, name: 'web', enabled: false })
  getToolsetConfig.mockResolvedValue({ has_category: true, active_provider: null, providers: [] })
  getUsageAnalytics.mockResolvedValue({ tools: [] })
  getSkillContent.mockResolvedValue({
    name: 'web-research',
    path: '/skills/web-research/SKILL.md',
    content: '---\nname: web-research\nversion: 1.2.0\nauthor: Nous\n---\n\n# Web Research\n\nDeep research steps.'
  })
  // Single profile by default → the scope selector stays hidden (>1 gate),
  // so existing tests see unchanged single-profile behavior.
  getProfiles.mockResolvedValue({ profiles: [{ name: 'default', is_default: true }] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  // Shared singleton client — drop cached skills/toolsets so each test refetches.
  queryClient.clear()
})

// CapabilitiesView is a heavy module (import cost now paid at module scope above,
// during collection) but the file still legitimately runs ~14s on CI runners —
// right against the global 15s per-test budget, so slow runners cascade-fail
// all 11 tests (2× in a row on PR #93612, plus a main run the same hour).
// Give this file headroom; the tests are not slow individually.
describe('CapabilitiesView toolset management', { timeout: 60_000 }, () => {
  it('renders a switch for each toolset and toggles it off', async () => {
    await renderSkills()

    // The switch names the action, so an enabled toolset offers to turn it off.
    const sw = await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
    expect(sw.getAttribute('aria-checked')).toBe('true')

    await act(async () => {
      fireEvent.click(sw)
    })

    await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalled())
    expect(setToolsetEnabled.mock.calls[0].slice(0, 2)).toEqual(['web', false])
  })

  it('renders toolset titles without leading emoji', async () => {
    getToolsets.mockResolvedValue([toolset({ name: 'cronjob', label: '⏰ Cron Jobs', description: 'cron tools' })])

    await renderSkills()

    // The label renders in both the row and the auto-selected detail header, so
    // assert via the switch's (emoji-stripped) accessible name and the absence
    // of the emoji rather than a single-match text lookup.
    await screen.findByRole('switch', { name: 'Turn Cron Jobs toolset off' })
    expect(screen.queryByText(/⏰/)).toBeNull()
  })

  it('renders the provider config panel inline for the selected toolset', async () => {
    // The master-detail UI dropped the resting "Configured" pill and the
    // "Configure" expander: the detail column auto-selects the first toolset
    // and renders its config panel directly, which fetches on mount.
    await renderSkills()

    await screen.findByRole('switch', { name: 'Turn Web Search toolset off' })
    await waitFor(() => expect(getToolsetConfig).toHaveBeenCalled())
    expect(getToolsetConfig.mock.calls[0][0]).toBe('web')
  })

  it('scopes Tools config to the profile chosen in the selector', async () => {
    // Two profiles → the "Configuring:" selector renders. Picking a non-active
    // profile must re-fetch toolsets scoped to THAT profile.
    // jsdom's scrollIntoView is missing/non-functional; Radix Select calls it
    // on open. Force a stub so the dropdown can render in the test env.
    Element.prototype.scrollIntoView = vi.fn()
    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'researcher', is_default: false }
      ]
    })

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=toolsets']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // The selector appears with >1 profile.
    const trigger = await screen.findByRole('combobox')
    await act(async () => {
      fireEvent.click(trigger)
    })
    const option = await screen.findByRole('option', { name: 'researcher' })
    await act(async () => {
      fireEvent.click(option)
    })

    // Toolsets refetch scoped to the picked profile.
    await waitFor(() => expect(getToolsets).toHaveBeenCalledWith('researcher'))
  })

  it('scopes the Skills tab (and skill toggles) to the profile chosen in the selector', async () => {
    // The selector is Capabilities-WIDE: picking a profile on the Skills tab
    // must refetch the skill list scoped to it, and route toggles there too.
    Element.prototype.scrollIntoView = vi.fn()
    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'researcher', is_default: false }
      ]
    })
    getSkills.mockResolvedValue([
      {
        name: 'web-research',
        description: 'Research the web',
        category: 'research',
        enabled: true,
        usage: 3,
        provenance: 'bundled'
      }
    ])

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // The selector renders on the Skills tab too (Capabilities-wide).
    const trigger = await screen.findByRole('combobox')
    await act(async () => {
      fireEvent.click(trigger)
    })
    const option = await screen.findByRole('option', { name: 'researcher' })
    await act(async () => {
      fireEvent.click(option)
    })

    // Skills refetch scoped to the picked profile...
    await waitFor(() => expect(getSkills).toHaveBeenCalledWith('researcher'))

    // ...and a toggle routes its write to that profile as well.
    const sw = await screen.findByRole('switch', { name: 'web-research' })
    await act(async () => {
      fireEvent.click(sw)
    })
    await waitFor(() => expect(setSkillEnabled).toHaveBeenCalledWith('web-research', false, 'researcher'))
  })

  it('shows the FULL skill in the detail pane — frontmatter metadata + body', async () => {
    getSkills.mockResolvedValue([
      {
        name: 'web-research',
        description: 'Research the web',
        category: 'research',
        enabled: true,
        usage: 3,
        provenance: 'bundled'
      }
    ])

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
            <CapabilitiesView />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    // Frontmatter renders as metadata rows, the body as full text — not just
    // the one-line description.
    await waitFor(() => expect(getSkillContent).toHaveBeenCalled())
    expect(getSkillContent.mock.calls[0][0]).toBe('web-research')
    expect(await screen.findByText('version')).toBeTruthy()
    expect(await screen.findByText('1.2.0')).toBeTruthy()
    expect(await screen.findByText(/Deep research steps/)).toBeTruthy()
  })

  it.each(['web-research', 'official/research/web-research'])(
    'disables catalog installation when the installed list contains %s',
    async installedName => {
      const { installHubSkill } = await import('@/store/hub-actions')
      queryClient.setQueryData(
        ['public-catalog', 'skills'],
        parseCatalog('skills', [
          {
            name: 'web-research',
            identifier: 'official/research/web-research',
            source: 'official',
            category: 'research',
            description: 'Research the web'
          }
        ])
      )

      render(
        <QueryClientProvider client={queryClient}>
          <SkillCatalog installedNames={new Set([installedName])} profile="researcher" />
        </QueryClientProvider>
      )

      fireEvent.click(screen.getByRole('button', { name: /^web-research/ }))
      const installed = screen.getByRole('button', { name: 'Installed' })
      expect((installed as HTMLButtonElement).disabled).toBe(true)
      fireEvent.click(installed)
      expect(installHubSkill).not.toHaveBeenCalled()
    }
  )

  it.each([
    {
      source: 'github',
      identifier: 'github:example/skills/research/community-research',
      expectedIdentifier: 'github:example/skills/research/community-research'
    },
    {
      source: 'clawhub',
      identifier: 'community-research',
      expectedIdentifier: 'clawhub/community-research'
    },
    {
      source: 'clawhub',
      identifier: 'clawhub/community-research',
      expectedIdentifier: 'clawhub/community-research'
    }
  ])(
    'installs $identifier with its source-qualified target in the pinned connection and profile',
    async ({ source, identifier, expectedIdentifier }) => {
      const { installHubSkill } = await import('@/store/hub-actions')

      const entry = {
        name: 'community-research',
        identifier,
        source,
        category: 'research',
        description: 'Community research workflow'
      }

      queryClient.setQueryData(
        ['public-catalog', 'skills'],
        parseCatalog('skills', [
          { ...entry, name: 'other-skill', source: 'github', identifier: 'github:example/skills/other-skill' },
          entry
        ])
      )

      // The picker iframe owns discovery now; the install contract lives in
      // SkillCatalog, so exercise it directly with the pinned scope object.
      render(
        <QueryClientProvider client={queryClient}>
          <SkillCatalog installedNames={new Set()} profile={{ connectionId: 'homelab', profile: 'researcher' }} />
        </QueryClientProvider>
      )

      fireEvent.click(await screen.findByRole('button', { name: /^community-research/ }))
      expect(screen.getByRole('heading', { name: entry.name })).toBeTruthy()
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: 'Install' }))
      })

      expect(installHubSkill).toHaveBeenCalledExactlyOnceWith(expectedIdentifier, {
        connectionId: 'homelab',
        profile: 'researcher'
      })
    }
  )

  it('keeps a pending install tied to its entry and scope when the pinned target changes', async () => {
    const { installHubSkill } = await import('@/store/hub-actions')
    const identifier = 'clawhub/community-research'
    queryClient.setQueryData(
      ['public-catalog', 'skills'],
      parseCatalog('skills', [
        { name: 'community-research', identifier: 'community-research', source: 'clawhub' },
        { name: 'other-skill', identifier: 'official/research/other-skill', source: 'optional' }
      ])
    )
    let finishFirst!: () => void
    let finishSecond!: () => void
    const firstInstall = new Promise<void>(resolve => {
      finishFirst = resolve
    })
    const secondInstall = new Promise<void>(resolve => {
      finishSecond = resolve
    })
    vi.mocked(installHubSkill).mockReturnValueOnce(firstInstall).mockReturnValueOnce(secondInstall)

    const scopedView = (connectionId: string, profile: string) => (
      <QueryClientProvider client={queryClient}>
        <SkillCatalog
          installedNames={new Set()}
          key={`${connectionId}:${profile}`}
          profile={{ connectionId, profile }}
        />
      </QueryClientProvider>
    )

    const view = render(scopedView('homelab', 'researcher'))
    fireEvent.click(await screen.findByRole('button', { name: /^community-research/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    const pending = screen.getByRole<HTMLButtonElement>('button', { name: 'Installing...' })
    expect(pending.disabled).toBe(true)
    expect(pending.querySelector('svg.animate-spin')).not.toBeNull()
    fireEvent.click(pending)
    expect(installHubSkill).toHaveBeenCalledExactlyOnceWith(identifier, {
      connectionId: 'homelab',
      profile: 'researcher'
    })

    fireEvent.click(screen.getByRole('button', { name: /^other-skill/ }))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Install' }).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: /^community-research/ }))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Installing...' }).disabled).toBe(true)

    view.rerender(scopedView('other-gateway', 'writer'))
    fireEvent.click(await screen.findByRole('button', { name: /^community-research/ }))
    const nextInstall = screen.getByRole<HTMLButtonElement>('button', { name: 'Install' })
    expect(nextInstall.disabled).toBe(false)
    fireEvent.click(nextInstall)
    expect(installHubSkill).toHaveBeenNthCalledWith(2, identifier, {
      connectionId: 'other-gateway',
      profile: 'writer'
    })

    await act(async () => {
      finishFirst()
      await firstInstall
    })
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Installing...' }).disabled).toBe(true)
    await act(async () => {
      finishSecond()
      await secondInstall
    })
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Install' }).disabled).toBe(false)
    expect(installHubSkill).toHaveBeenCalledTimes(2)
  })

  it('loads the public catalog only when the browse view is enabled, and one fetch serves repeated views', async () => {
    // The picker iframe owns discovery in the Capabilities view now; the lazy-
    // load contract lives in useCatalog(kind, enabled): the feed is fetched
    // only while the browse view is mounted, and the query cache means one
    // fetch serves every remount (tab switches) for the same kind.
    const fetchCatalog = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        {
          name: 'catalog-research',
          identifier: 'official/research/catalog-research',
          source: 'official',
          category: 'research',
          description: 'Research from the public snapshot'
        }
      ]
    })

    vi.stubGlobal('fetch', fetchCatalog)

    queryClient.clear()

    const { renderHook } = await import('@testing-library/react')
    const { useCatalog } = await import('./plugins/catalog-data')

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )

    const view1 = renderHook(() => useCatalog('skills', true), { wrapper })
    await waitFor(() => expect(view1.result.current.isSuccess).toBe(true))
    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    expect(fetchCatalog.mock.calls[0][0]).toMatch(/\/skills\.json$/)

    view1.unmount()

    // Same kind remounts (the tab-bounce shape): served from cache, no second fetch.
    const view2 = renderHook(() => useCatalog('skills', true), { wrapper })
    await waitFor(() => expect(view2.result.current.isSuccess).toBe(true))
    expect(fetchCatalog).toHaveBeenCalledTimes(1)

    // Disabled (installed view) never fetches: a fresh kind with enabled=false
    // stays pending forever (enabled gates the queryFn), and the cached skills
    // kind gains no new fetch either.
    const view3 = renderHook(() => useCatalog('plugins', false), { wrapper })
    expect(view3.result.current.isPending).toBe(true)
    expect(fetchCatalog).toHaveBeenCalledTimes(1)
    view3.unmount()
  })

  it('shows a vision explainer that deep-links to Settings → Models', async () => {
    // Vision has no TOOL_CATEGORIES provider matrix — its model lives in the
    // auxiliary model config, so the detail pane must point there instead of
    // rendering an empty panel.
    getToolsets.mockResolvedValue([
      toolset({
        name: 'vision',
        label: 'Vision / Image Analysis',
        description: 'vision_analyze',
        tools: ['vision_analyze']
      })
    ])
    getToolsetConfig.mockResolvedValue({ has_category: false, active_provider: null, providers: [] })

    await renderSkills()

    expect(await screen.findByText(/auxiliary model configuration/)).toBeTruthy()
    const link = screen.getByRole('button', { name: /Choose vision model in Settings/ })

    await act(async () => {
      fireEvent.click(link)
    })

    // Internal route change into the Models section with the aux slot target —
    // consumed by ModelSettings' deep-link highlight. Never an external URL.
    await waitFor(() => expect(navigateSpy).toHaveBeenCalledWith('/settings?tab=config:model&aux=vision'))
  })

  it('fixedConnection pins every read to the target connection', async () => {
    // Bot Mode's remote-target door: a bot on another registered gateway gets
    // the live surface pointed at ITS backend — the reads must carry the
    // (connection, profile) pin, not a bare profile name that would resolve
    // against the ACTIVE gateway (the wrong-machine bug).
    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/capabilities']}>
            <CapabilitiesView embedded fixedConnection="homelab" fixedProfile="inbox-bot" />
          </MemoryRouter>
        </QueryClientProvider>
      )
    })

    await waitFor(() => expect(getSkills).toHaveBeenCalled())
    expect(getSkills.mock.calls[0][0]).toEqual({ connectionId: 'homelab', profile: 'inbox-bot' })
    expect(getToolsets.mock.calls[0][0]).toEqual({ connectionId: 'homelab', profile: 'inbox-bot' })
    // Pinned scope → no roster/profiles fetch, selector hidden.
    expect(getProfiles).not.toHaveBeenCalled()
  })

  it('offers (connection, profile) scope rows on multi-connection desktops', async () => {
    // With a v2 registry holding >1 connection, the scope selector lists the
    // union agent roster — profile + owning device — instead of the local
    // profiles list, so a selection identifies WHICH gateway's capabilities
    // are being configured.
    const connections = {
      list: vi.fn().mockResolvedValue({
        version: 2,
        primary: 'local',
        secureTokenStorage: true,
        connections: [
          { id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null },
          { id: 'homelab', kind: 'remote', label: 'Homelab', tokenSet: true, tokenPreview: '…' }
        ]
      })
    }

    const getAgentRoster = vi.fn().mockResolvedValue({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local',
          connectionLabel: 'This device',
          profile: 'default',
          handle: 'default'
        },
        {
          connectionId: 'homelab',
          connectionKind: 'remote',
          connectionLabel: 'Homelab',
          profile: 'inbox-bot',
          handle: 'inbox-bot-homelab'
        }
      ],
      sources: []
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections, getAgentRoster }

    try {
      await renderSkills()

      await waitFor(() => expect(getAgentRoster).toHaveBeenCalled())
      // The selector paints roster rows labeled profile — device.
      expect(await screen.findByText('default — This device (current)')).toBeTruthy()
    } finally {
      delete (window as { hermesDesktop?: unknown }).hermesDesktop
    }
  })

  it('routes hub picker installs through the scoped profile and refuses picks already installed', async () => {
    // The Skills tab embeds the real hub as an iframe; a card click posts
    // { type: 'hermes-skill-pick', name, identifier } to the parent. The
    // picker must install through installHubSkill with the scoped profile and
    // refuse (toast, no call) a pick whose name/identifier is already
    // installed in that scope.
    const { installHubSkill } = await import('@/store/hub-actions')
    const { EmbeddedHubPicker } = await import('./skills/embedded-hub-picker')

    const installedNames = new Set(['web-research'])

    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddedHubPicker installedNames={installedNames} profile="researcher" />
      </QueryClientProvider>
    )

    const pick = (identifier: string, name: string) => {
      fireEvent(
        window,
        new MessageEvent('message', {
          data: { type: 'hermes-skill-pick', identifier, name },
          origin: 'https://hermes-agent.nousresearch.com'
        })
      )
    }

    // Installed pick → refused, no install call.
    pick('official/research/web-research', 'web-research')
    expect(installHubSkill).not.toHaveBeenCalled()

    // New pick → installed with the scoped profile.
    pick('official/gifs/gif-search', 'gif-search')
    expect(installHubSkill).toHaveBeenCalledExactlyOnceWith('official/gifs/gif-search', 'researcher')

    // Foreign-origin messages never reach the install path.
    vi.mocked(installHubSkill).mockClear()
    fireEvent(
      window,
      new MessageEvent('message', {
        data: { type: 'hermes-skill-pick', identifier: 'evil/evil', name: 'evil' },
        origin: 'https://evil.example.com'
      })
    )
    expect(installHubSkill).not.toHaveBeenCalled()
  })
})
