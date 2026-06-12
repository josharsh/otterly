import { execFileSync, spawn } from "child_process";
import type {
  AgentEvent,
  AgentResult,
  EngineOptions,
  ToolExecution,
} from "./types.js";
import { normalizeEvents, createEventContext } from "./events.js";
import { classifyError, AgentError } from "./errors.js";
import { wrapPermissionHandler } from "./permissions.js";
import { Session } from "./session.js";

type QueryFn = (args: { prompt: unknown; options: Record<string, unknown> }) => AsyncIterable<Record<string, unknown>>;

let cachedQueryFn: QueryFn | null = null;
let resolvedMode: "sdk" | "cli" | null = null;

/**
 * Find the `claude` CLI binary. Returns the name or null.
 */
function findClaudeCLI(): string | null {
  for (const bin of ["claude", "claude-code"]) {
    try {
      execFileSync("which", [bin], { stdio: "pipe" });
      execFileSync(bin, ["--version"], { stdio: "pipe", timeout: 5000 });
      return bin;
    } catch {
      // Try next
    }
  }
  return null;
}

/**
 * Build a shell command string for `claude -p`.
 * The prompt is NOT embedded in the command — it is passed via stdin by the
 * caller (see createCLIQueryFn). This avoids hitting ARG_MAX (E2BIG) for large
 * prompts such as those produced by IDEs/agents that include repo maps and
 * file context.
 */
function buildCLICommand(cliBin: string, prompt: string, opts: Record<string, unknown>): string {
  // Prompt is fed via stdin; do not place it on the command line.
  void prompt;
  const parts = [cliBin, "-p", "--output-format", "stream-json", "--verbose"];

  if (opts.model) parts.push("--model", String(opts.model));
  if (opts.systemPrompt) {
    const safe = String(opts.systemPrompt).replace(/'/g, "'\\''");
    parts.push("--system-prompt", `'${safe}'`);
  }
  if (opts.resume) parts.push("--resume", String(opts.resume));
  if (opts.permissionMode) parts.push("--permission-mode", String(opts.permissionMode));
  if (opts.allowedTools) {
    for (const tool of opts.allowedTools as string[]) parts.push("--allowedTools", tool);
  }
  if (opts.disallowedTools) {
    for (const tool of opts.disallowedTools as string[]) parts.push("--disallowedTools", tool);
  }

  return parts.join(" ");
}

/**
 * Spawn a shell command and return its stdout. Cancellable via AbortController.
 *
 * Hard rule: the returned promise must not resolve or reject until the child
 * process has actually exited. The whole point is that the upstream request
 * queue uses the promise lifecycle to track a worker slot — if we return early
 * while the child is still alive, the slot leaks and we end up with a stuck
 * queue (see https://github.com/josharsh/otterly/issues/4).
 *
 * On abort or timeout:
 *   1. Send SIGTERM to the child's process group.
 *   2. Wait `killGraceMs` for it to exit cleanly.
 *   3. If still alive, send SIGKILL to the group.
 *   4. Resolve only when the child's `exit` event fires.
 *
 * Spawned with `detached: true` so it owns its own process group — a Bun-
 * compiled binary that re-execs or spawns helpers can be reaped in one shot
 * via the negative-PID kill (`process.kill(-pid, signal)`).
 *
 * `shell: true` is used because the legacy execSync path worked through a
 * shell and the Bun-compiled `claude` binary historically had stdout-piping
 * quirks with direct `spawn`. Using a shell preserves the previous behavior
 * while making the call cancellable.
 *
 * The prompt (if any) is written to the child's stdin via the `input` option
 * rather than being placed on the command line, so that large prompts do not
 * exceed the OS ARG_MAX limit (E2BIG).
 */
export interface SpawnWithAbortOptions {
  cwd?: string;
  abortController?: AbortController;
  /** Hard timeout in ms. Triggers the same kill ladder as abort. */
  timeoutMs: number;
  /** Grace period between SIGTERM and SIGKILL. Default 5s. */
  killGraceMs?: number;
  /** Data to write to the child's stdin (e.g. the prompt). */
  input?: string;
  /** Test-only injection hook. */
  spawnFn?: typeof spawn;
}

export function spawnWithAbort(cmd: string, opts: SpawnWithAbortOptions): Promise<string> {
  const killGraceMs = opts.killGraceMs ?? 5000;
  const doSpawn = opts.spawnFn ?? spawn;

  return new Promise<string>((resolve, reject) => {
    const child = doSpawn(cmd, [], {
      shell: true,
      cwd: opts.cwd,
      env: process.env,
      detached: process.platform !== "win32",
      stdio: [opts.input != null ? "pipe" : "ignore", "pipe", "pipe"],
    });

    // Feed the prompt to the child via stdin. Guard against EPIPE in case the
    // child exits before consuming all input.
    if (opts.input != null && child.stdin) {
      child.stdin.on("error", () => {
        // Child closed stdin early — ignore (e.g. EPIPE).
      });
      child.stdin.write(opts.input);
      child.stdin.end();
    }

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    child.stdout?.on("data", (c: Buffer) => stdoutChunks.push(c));
    child.stderr?.on("data", (c: Buffer) => stderrChunks.push(c));

    let aborted = false;
    let timedOut = false;
    let sigkillTimer: NodeJS.Timeout | null = null;

    const killGroup = (signal: NodeJS.Signals) => {
      const pid = child.pid;
      if (!pid) return;
      try {
        if (process.platform === "win32") {
          // No process groups on Windows; kill the child directly.
          child.kill(signal);
        } else {
          // Negative PID = whole process group (works because detached: true).
          process.kill(-pid, signal);
        }
      } catch {
        // Already gone — that is what we wanted.
      }
    };

    const beginKillLadder = () => {
      killGroup("SIGTERM");
      sigkillTimer = setTimeout(() => killGroup("SIGKILL"), killGraceMs);
    };

    const hardTimer = setTimeout(() => {
      timedOut = true;
      beginKillLadder();
    }, opts.timeoutMs);

    const onAbort = () => {
      aborted = true;
      beginKillLadder();
    };
    opts.abortController?.signal.addEventListener("abort", onAbort, { once: true });

    const cleanup = () => {
      clearTimeout(hardTimer);
      if (sigkillTimer) clearTimeout(sigkillTimer);
      opts.abortController?.signal.removeEventListener("abort", onAbort);
    };

    child.on("error", (err) => {
      cleanup();
      reject(err);
    });

    // Resolve/reject only after the child has truly exited. This is the
    // contract that prevents queue-slot leaks.
    child.on("exit", () => {
      cleanup();
      if (aborted) {
        reject(new Error("Aborted"));
      } else if (timedOut) {
        const stderr = Buffer.concat(stderrChunks).toString("utf-8").trim();
        reject(new Error(`CLI timed out after ${opts.timeoutMs}ms${stderr ? `: ${stderr.slice(-500)}` : ""}`));
      } else {
        resolve(Buffer.concat(stdoutChunks).toString("utf-8"));
      }
    });
  });
}

/**
 * Build a QueryFn that runs `claude -p` via a cancellable subprocess.
 *
 * Historical context: an earlier version of this file ran `execSync` inside a
 * worker thread because the Bun-compiled `claude` binary appeared not to pipe
 * stdout to async `spawn`/`exec`. That implementation leaked worker slots —
 * calling `worker.terminate()` on abort killed the worker but orphaned the
 * underlying `claude` subprocess, which kept holding the request queue slot
 * until it exited on its own (sometimes never, for hung interactive prompts).
 *
 * This implementation uses `spawn` through `sh -c '<cmd>'`, which preserves
 * the shell-piping path that worked for execSync while making the child
 * killable on abort. See `spawnWithAbort` for the signal handling.
 *
 * The prompt is fed to claude via stdin (not argv) so that large prompts do
 * not exceed the OS ARG_MAX limit (E2BIG).
 */
function createCLIQueryFn(cliBin: string): QueryFn {
  return function cliQuery(args: { prompt: unknown; options: Record<string, unknown> }): AsyncIterable<Record<string, unknown>> {
    const opts = args.options || {};
    const prompt = typeof args.prompt === "string" ? args.prompt : JSON.stringify(args.prompt);
    const cmd = buildCLICommand(cliBin, prompt, opts);

    return {
      async *[Symbol.asyncIterator]() {
        const ac = opts.abortController as AbortController | undefined;
        const timeoutMs = ac
          ? 10 * 60 * 1000   // 10 min for streaming
          : 5 * 60 * 1000;   // 5 min for one-shot

        const stdout = await spawnWithAbort(cmd, {
          cwd: opts.cwd ? String(opts.cwd) : undefined,
          abortController: ac,
          timeoutMs,
          input: prompt,
        });

        // Parse the NDJSON output line by line and yield each event
        const lines = stdout.trim().split("\n");
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            yield JSON.parse(line);
          } catch {
            // Skip non-JSON lines
          }
        }
      },
    };
  };
}

async function resolveSDK(): Promise<QueryFn> {
  if (cachedQueryFn) return cachedQueryFn;

  // 1. Try the npm SDK packages
  for (const pkg of ["@anthropic-ai/claude-code", "@anthropic-ai/claude-agent-sdk"]) {
    try {
      const mod = await import(pkg);
      const fn = mod.query || mod.default?.query;
      if (typeof fn === "function") {
        cachedQueryFn = fn as QueryFn;
        resolvedMode = "sdk";
        return cachedQueryFn;
      }
    } catch {
      // Try next
    }
  }

  // 2. Fall back to the CLI binary (works regardless of install method)
  const cliBin = await findClaudeCLI();
  if (cliBin) {
    cachedQueryFn = createCLIQueryFn(cliBin);
    resolvedMode = "cli";
    return cachedQueryFn;
  }

  throw new AgentError(
    "SDK_NOT_FOUND",
    "Could not find Claude Code. Install it from https://docs.anthropic.com/en/docs/claude-code or:\n  npm install -g @anthropic-ai/claude-code"
  );
}

export class ClaudeEngine {
  private defaults: Partial<EngineOptions>;

  constructor(defaults?: Partial<EngineOptions>) {
    this.defaults = defaults || {};
  }

  private mergeOptions(options?: EngineOptions): EngineOptions {
    return { ...this.defaults, ...options };
  }

  /**
   * Run a prompt and get the final result. Blocks until complete.
   *
   * ```ts
   * const result = await claude.run("Fix the login bug", { cwd: "./app" });
   * console.log(result.text, result.cost);
   * ```
   */
  async run(prompt: string, options?: EngineOptions): Promise<AgentResult> {
    const tools: ToolExecution[] = [];
    let resultText = "";
    let cost = 0;
    let duration = 0;
    let sessionId = "";
    let usage = { input_tokens: 0, output_tokens: 0 };
    const pendingToolUses = new Map<string, { tool: string; input: Record<string, unknown> }>();

    for await (const event of this.stream(prompt, options)) {
      switch (event.type) {
        case "text":
          resultText = event.text;
          break;
        case "tool_use":
          pendingToolUses.set(event.id, { tool: event.tool, input: event.input });
          break;
        case "tool_result": {
          const pending = pendingToolUses.get(event.toolUseId);
          tools.push({
            tool: pending?.tool || event.tool,
            input: pending?.input || {},
            output: event.output,
            isError: event.isError,
          });
          pendingToolUses.delete(event.toolUseId);
          break;
        }
        case "result":
          resultText = event.text || resultText;
          cost = event.cost;
          duration = event.duration;
          sessionId = event.sessionId;
          usage = event.usage;
          break;
        case "error":
          throw event.error instanceof AgentError
            ? event.error
            : classifyError(event.error);
      }
    }

    return { text: resultText, cost, duration, sessionId, usage, tools };
  }

  /**
   * Stream events from a prompt in real-time.
   *
   * ```ts
   * for await (const event of claude.stream("Refactor auth", { cwd: "." })) {
   *   if (event.type === "text_delta") process.stdout.write(event.delta);
   *   if (event.type === "tool_use") console.log(`Using ${event.tool}...`);
   * }
   * ```
   */
  async *stream(prompt: string, options?: EngineOptions): AsyncGenerator<AgentEvent> {
    const opts = this.mergeOptions(options);
    const queryFn = await resolveSDK();

    const abortController = new AbortController();
    if (opts.signal) {
      opts.signal.addEventListener("abort", () => abortController.abort());
    }

    const queryOptions: Record<string, unknown> = {
      abortController,
      cwd: opts.cwd || process.cwd(),
      permissionMode: opts.permissionMode || "bypassPermissions",
      includePartialMessages: true,
    };

    if (opts.model) queryOptions.model = opts.model;
    if (opts.maxTurns) queryOptions.maxTurns = opts.maxTurns;
    if (opts.allowedTools) queryOptions.allowedTools = opts.allowedTools;
    if (opts.disallowedTools) queryOptions.disallowedTools = opts.disallowedTools;
    if (opts.mcpServers) queryOptions.mcpServers = opts.mcpServers;
    if (opts.effort) queryOptions.effort = opts.effort;
    if (opts.resume) queryOptions.resume = opts.resume;
    if (opts.systemPrompt) queryOptions.systemPrompt = opts.systemPrompt;

    if (opts.onPermission) {
      queryOptions.permissionMode = "default";
      queryOptions.canUseTool = wrapPermissionHandler(opts.onPermission);
    }

    const ctx = createEventContext();

    try {
      const queryInstance = queryFn({ prompt, options: queryOptions });

      for await (const raw of queryInstance) {
        if (abortController.signal.aborted) break;

        const events = normalizeEvents(raw as Record<string, unknown>, ctx);
        for (const event of events) {
          yield event;
        }
      }
    } catch (err) {
      if (
        err instanceof Error &&
        (err.name === "AbortError" || abortController.signal.aborted)
      ) {
        return;
      }
      throw classifyError(err);
    }
  }

  /**
   * Create a multi-turn session. Context persists across send() calls.
   *
   * ```ts
   * const session = claude.session({ cwd: "./my-project" });
   * await session.send("Create a REST API");
   * await session.send("Now add auth to it");
   * session.close();
   * ```
   */
  session(options?: EngineOptions): Session {
    const opts = this.mergeOptions(options);
    // Session needs the query function — resolve it lazily on first send()
    // We wrap it in a lazy resolver to avoid top-level await
    let resolvedFn: QueryFn | null = null;

    const lazyQueryFn: QueryFn = async function* (args) {
      if (!resolvedFn) {
        resolvedFn = await resolveSDK();
      }
      yield* resolvedFn(args);
    };

    return new Session(lazyQueryFn, opts);
  }
}