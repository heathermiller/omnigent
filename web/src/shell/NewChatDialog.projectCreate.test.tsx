import type * as UseConversationsModule from "@/hooks/useConversations";
import type * as AgentLabelsModule from "@/lib/agentLabels";
import type * as ToastModule from "@/components/ui/toast";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { authenticatedFetch } from "@/lib/identity";
import type { Host } from "@/hooks/useHosts";
import { useHosts } from "@/hooks/useHosts";
import type { AvailableAgent } from "@/hooks/useAvailableAgents";
import { useAvailableAgents } from "@/hooks/useAvailableAgents";
import { moveConversationToProject, useProjectConfig, useProjects } from "@/hooks/useConversations";
import type { ProjectConfig } from "@/lib/projectsApi";
import { showToast } from "@/components/ui/toast";
import { useHostWorktrees } from "@/hooks/useHostWorktrees";
import type { HostWorktree } from "@/hooks/useHostWorktrees";
import { NewChatLandingScreen, resetLandingDraft } from "./NewChatDialog";

// A project-driven visit (`?project=` resolved to a first-class project id)
// creates the session WITH `project_id`: the server files it atomically and
// default-fills config-seeded fields the composer omits. These tests pin the
// new request shape, the field-omission semantics, the skipped follow-up
// move, and the untouched legacy paths (label-only folders, plain visits).
const navigateMock = vi.fn();

const RECENT_KEY = "omnigent:recent-workspaces";
const RECENT_WORKSPACE = "/Users/corey/universe/src/foo";
const REPO = "/Users/corey/projects/alpha";

// Mutable so a test can choose a project-driven or plain visit.
let searchParams = new URLSearchParams("project=Alpha");
vi.mock("@/lib/routing", () => ({
  useNavigate: () => navigateMock,
  useSearchParams: () => [searchParams, vi.fn()],
}));

vi.mock("@/store/chatStore", () => ({
  setPendingInitialPrompt: vi.fn(),
}));

vi.mock("@/lib/identity", () => ({ authenticatedFetch: vi.fn() }));
vi.mock("@/components/ui/toast", async (importOriginal) => ({
  ...(await importOriginal<typeof ToastModule>()),
  showToast: vi.fn(),
}));
vi.mock("@/hooks/useHosts", () => ({
  useHosts: vi.fn(),
  useHostModelOptions: vi.fn(() => ({ data: [] })),
  useInstallHarness: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  useInstallingHarnesses: vi.fn(() => new Set<string>()),
}));
vi.mock("@/hooks/useAvailableAgents", () => ({
  useAvailableAgents: vi.fn(),
  prefetchAvailableAgentDetails: vi.fn(),
}));
vi.mock("@/hooks/useHostFilesystem", () => ({
  useHostFilesystem: () => ({ data: undefined }),
  useCreateHostDirectory: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));
vi.mock("@/hooks/useHostWorktrees", () => ({
  useHostWorktrees: vi.fn(),
}));
vi.mock("@/hooks/useDirectorySessions", () => ({
  useDirectorySessions: () => ({ data: [] }),
}));
vi.mock("@/hooks/RunnerHealthProvider", () => ({
  useRunnerHealthRegistration: () => new Map<string, boolean>(),
}));
// The projects list + config drive `project_id` resolution; the move helper is
// mocked so the tests can assert it is (not) called without HTTP plumbing.
vi.mock("@/hooks/useConversations", async (importOriginal) => ({
  ...(await importOriginal<typeof UseConversationsModule>()),
  useProjects: vi.fn(),
  useProjectConfig: vi.fn(),
  moveConversationToProject: vi.fn(),
  // The landing reads useConversations to decide hasNoSessions (the empty-state
  // import affordance); stub it so it doesn't fire an authenticatedFetch that
  // lands at mock.calls[0] and skews these create-POST call assertions.
  useConversations: () => ({ data: undefined }),
}));
vi.mock("@/lib/agentLabels", async (importOriginal) => ({
  ...(await importOriginal<typeof AgentLabelsModule>()),
  useBrainHarnessLabels: () => ({}),
  useHarnessSetupSteps: () => ({}),
}));

function host(overrides: Partial<Host> = {}): Host {
  return {
    host_id: "host_1",
    name: "corey-laptop",
    owner: "corey",
    status: "online",
    ...overrides,
  };
}

function agent(overrides: Partial<AvailableAgent> = {}): AvailableAgent {
  return {
    id: "ag_hello",
    name: "hello_world",
    display_name: "Hello World",
    description: null,
    harness: null,
    skills: [],
    ...overrides,
  };
}

function setProjectConfig(config: ProjectConfig | undefined, isLoading = false): void {
  vi.mocked(useProjectConfig).mockReturnValue({ data: config, isLoading } as ReturnType<
    typeof useProjectConfig
  >);
}

function setProjects(
  data: { id: string | null; name: string }[] | undefined,
  isLoading = false,
): void {
  vi.mocked(useProjects).mockReturnValue({ data, isLoading } as ReturnType<typeof useProjects>);
}

/** Serve a git repo (has an is_main worktree) at REPO; [] elsewhere. */
function setRepoIsGit(): void {
  vi.mocked(useHostWorktrees).mockImplementation((hostId, path) => {
    const known = hostId === "host_1" && path === REPO;
    return {
      data: known
        ? ([{ path: REPO, branch: "main", is_main: true, detached: false }] as HostWorktree[])
        : ([] as HostWorktree[]),
      isError: false,
    } as ReturnType<typeof useHostWorktrees>;
  });
}

function renderLanding(): { rerender: (ui: ReactNode) => void; unmount: () => void } {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  const { rerender, unmount } = render(<NewChatLandingScreen />, { wrapper: Wrapper });
  return { rerender, unmount };
}

/** Open the picker and commit (select + close) an agent by clicking its row. */
function selectAgent(agentId: string): void {
  fireEvent.pointerDown(screen.getByTestId("new-chat-landing-agent-select"), { button: 0 });
  if (screen.queryByTestId(`new-chat-landing-agent-${agentId}`) == null) {
    fireEvent.click(screen.getByTestId("new-chat-landing-custom-agents"));
  }
  fireEvent.click(screen.getByTestId(`new-chat-landing-agent-${agentId}`));
}

async function submitAndReadBody(
  response: Record<string, unknown> = { id: "conv_new" },
): Promise<Record<string, unknown>> {
  vi.mocked(authenticatedFetch).mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(response),
  } as Response);
  fireEvent.change(screen.getByTestId("new-chat-landing-input"), {
    target: { value: "hello" },
  });
  fireEvent.click(screen.getByTestId("new-chat-landing-submit"));
  await waitFor(() => expect(vi.mocked(authenticatedFetch)).toHaveBeenCalled());
  // Let the post-create bookkeeping (move / invalidations / navigation) run
  // before reading the body, so move-call assertions can't race the submit.
  await waitFor(() => expect(navigateMock).toHaveBeenCalled());
  const [, init] = vi.mocked(authenticatedFetch).mock.calls[0] as [string, RequestInit];
  return JSON.parse(init.body as string) as Record<string, unknown>;
}

beforeEach(() => {
  navigateMock.mockReset();
  vi.mocked(authenticatedFetch).mockReset();
  vi.mocked(moveConversationToProject).mockReset();
  vi.mocked(moveConversationToProject).mockResolvedValue({} as never);
  vi.mocked(showToast).mockReset();
  searchParams = new URLSearchParams("project=Alpha");
  resetLandingDraft();
  localStorage.clear();
  localStorage.setItem(RECENT_KEY, JSON.stringify({ host_1: [RECENT_WORKSPACE] }));
  vi.mocked(useHosts).mockReturnValue({ data: [host()] } as ReturnType<typeof useHosts>);
  vi.mocked(useAvailableAgents).mockReturnValue({
    data: [agent(), agent({ id: "ag_other", name: "other", display_name: "Other" })],
  } as ReturnType<typeof useAvailableAgents>);
  setRepoIsGit();
  setProjects([{ id: "proj_alpha", name: "Alpha" }]);
  setProjectConfig({});
});

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe("NewChatLandingScreen project-aware create (first-class project_id)", () => {
  it("sends project_id and omits config-seeded agent_id + workspace when nothing was overridden", async () => {
    setProjectConfig({ host_id: "host_1", workspace: REPO, agent_id: "ag_other" });
    renderLanding();
    await waitFor(() =>
      expect(screen.getByTestId("new-chat-landing-workspace-chip").textContent).toContain("alpha"),
    );

    const body = await submitAndReadBody();
    expect(body.project_id).toBe("proj_alpha");
    // Config-seeded and untouched: the server default-fills these from the
    // project config, so the request omits them entirely.
    expect("agent_id" in body).toBe(false);
    expect("workspace" in body).toBe(false);
    // The host is not part of the omission contract — still sent explicitly.
    expect(body.host_id).toBe("host_1");
    // Born filed via first-class project_id — no legacy omni_project label.
    expect((body.labels as Record<string, string> | undefined)?.omni_project).toBeUndefined();
  });

  it("sends an explicit agent_id alongside project_id when the user picks a different agent", async () => {
    setProjectConfig({ host_id: "host_1", workspace: REPO, agent_id: "ag_other" });
    renderLanding();
    await waitFor(() =>
      expect(screen.getByTestId("new-chat-landing-agent-select").textContent).toContain("Other"),
    );
    selectAgent("ag_hello");

    const body = await submitAndReadBody();
    expect(body.project_id).toBe("proj_alpha");
    // The explicit pick is authoritative — sent so the server can warn on a
    // mismatch instead of silently default-filling the config agent.
    expect(body.agent_id).toBe("ag_hello");
    // The workspace stayed config-seeded, so it remains omitted.
    expect("workspace" in body).toBe(false);
  });

  it("keeps a non-project visit's body unchanged: no project_id, all fields explicit", async () => {
    searchParams = new URLSearchParams("");
    localStorage.setItem("omnigent:last-agent-id", "ag_hello");
    renderLanding();
    await waitFor(() =>
      expect(screen.getByTestId("new-chat-landing-workspace-chip").textContent).toContain("foo"),
    );

    const body = await submitAndReadBody();
    // Byte-identical to the pre-project_id shape: exactly the explicit
    // fields, nothing new riding along.
    expect(Object.keys(body).sort()).toEqual(["agent_id", "host_id", "workspace"]);
    expect(body.agent_id).toBe("ag_hello");
    expect(body.host_id).toBe("host_1");
    expect(body.workspace).toBe(RECENT_WORKSPACE);
  });

  it("skips the post-create move when project_id was sent (atomic server-side filing)", async () => {
    setProjectConfig({ host_id: "host_1", workspace: REPO, agent_id: "ag_other" });
    renderLanding();
    await waitFor(() =>
      expect(screen.getByTestId("new-chat-landing-workspace-chip").textContent).toContain("alpha"),
    );

    const body = await submitAndReadBody();
    expect(body.project_id).toBe("proj_alpha");
    expect(vi.mocked(moveConversationToProject)).not.toHaveBeenCalled();
  });

  it("keeps the label + follow-up move for a label-only folder (no first-class id)", async () => {
    // The folder exists only as a legacy label: no project row to address by
    // id, so the create cannot send project_id — it stamps the born-filed
    // label and the follow-up move creates the row on demand.
    setProjects([{ id: null, name: "Alpha" }]);
    setProjectConfig(undefined);
    renderLanding();
    await waitFor(() =>
      expect(screen.getByTestId("new-chat-landing-workspace-chip").textContent).toContain("foo"),
    );

    const body = await submitAndReadBody();
    expect("project_id" in body).toBe(false);
    expect((body.labels as Record<string, string>).omni_project).toBe("Alpha");
    expect(vi.mocked(moveConversationToProject)).toHaveBeenCalledWith("conv_new", "Alpha");
  });

  it("surfaces server mismatch warnings from the create response as toasts", async () => {
    setProjectConfig({ host_id: "host_1", workspace: REPO, agent_id: "ag_other" });
    renderLanding();
    await waitFor(() =>
      expect(screen.getByTestId("new-chat-landing-agent-select").textContent).toContain("Other"),
    );
    selectAgent("ag_hello");

    await submitAndReadBody({
      id: "conv_new",
      warnings: [
        {
          code: "project_agent_mismatch",
          message: "Explicit builtin agent differs from the project's custom agent hint",
        },
      ],
    });
    await waitFor(() =>
      expect(vi.mocked(showToast)).toHaveBeenCalledWith(
        "Explicit builtin agent differs from the project's custom agent hint",
      ),
    );
  });
});
