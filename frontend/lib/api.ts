/**
 * InterVu API client
 * Typed wrapper around all backend endpoints.
 *
 * Every call except login sends `Authorization: Bearer <access_token>`.
 * A 401 response clears the stored session and sends the user to /login.
 */

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

const STORAGE_KEY = "intervu_user";

/** Default request timeout. */
const DEFAULT_TIMEOUT_MS = 30_000;
/**
 * Calls that run LLM work (answer evaluation, report generation). Sized for
 * the backend defaults (LLM_TIMEOUT_SECONDS=40, LLM_MAX_ATTEMPTS=2, so about
 * 81 s worst case per LLM call). If the client still gives up first, a
 * resubmitted answer gets 409 while the first one is being processed.
 */
const LLM_TIMEOUT_MS = 180_000;
/** Starting an interview runs the full research pipeline (search, scraping, extraction). */
const START_TIMEOUT_MS = 420_000;

// ── Errors ────────────────────────────────────────────────────────────────────

/**
 * Error thrown for every failed request.
 * `status` is the HTTP status, or 0 for network failures and timeouts
 * (`timedOut` tells them apart).
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly timedOut: boolean;

  constructor(status: number, detail: string, timedOut = false) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.timedOut = timedOut;
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError;
}

/** A human friendly message for any thrown value. */
export function errorMessage(e: unknown, fallback = "Something went wrong. Please try again."): string {
  if (isApiError(e)) {
    if (e.timedOut) return "The server took too long to respond. Please try again.";
    if (e.status === 0) return "Cannot reach the InterVu server. Check that the backend is running.";
    return e.detail || fallback;
  }
  return fallback;
}

/** Extract FastAPI's `detail` (string or validation error list) from a body. */
function parseDetail(text: string, status: number): string {
  if (!text) return `Request failed (${status})`;
  try {
    const data: unknown = JSON.parse(text);
    if (data && typeof data === "object" && "detail" in data) {
      const detail = (data as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        const messages = detail
          .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : ""))
          .filter(Boolean);
        if (messages.length > 0) return messages.join("; ");
      }
    }
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return text.length > 300 ? `${text.slice(0, 300)}...` : text;
}

// ── Core request ──────────────────────────────────────────────────────────────

type RequestOptions = {
  method?: "GET" | "POST";
  body?: unknown;
  /** Send the bearer token (default true). */
  authenticated?: boolean;
  timeoutMs?: number;
  signal?: AbortSignal;
};

function handleUnauthorized() {
  auth.clear();
  if (typeof window !== "undefined" && window.location.pathname !== "/login") {
    window.location.assign("/login?expired=1");
  }
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const {
    method = "GET",
    body,
    authenticated = true,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    signal,
  } = options;

  const headers: Record<string, string> = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";

  if (authenticated) {
    const session = auth.load();
    if (!session) {
      handleUnauthorized();
      throw new ApiError(401, "Please sign in again.");
    }
    headers.Authorization = `Bearer ${session.access_token}`;
  }

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const forwardAbort = () => controller.abort();
  signal?.addEventListener("abort", forwardAbort);

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (e) {
    if (timedOut) throw new ApiError(0, "Request timed out", true);
    if (signal?.aborted) throw e;
    throw new ApiError(0, "Network error");
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forwardAbort);
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    if (res.status === 401 && authenticated) handleUnauthorized();
    throw new ApiError(res.status, parseDetail(text, res.status));
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export type Difficulty = "Easy" | "Medium" | "Hard";

export type BlueprintSection = {
  name: string;
  type: "coding" | "behavioral" | "technical" | "system_design" | "screening";
  questions: number;
  focus_areas: string[];
};

export type Blueprint = {
  sections: BlueprintSection[];
};

export type StartInterviewResponse = {
  interview_id: number;
  first_question: string;
  blueprint: Blueprint;
  difficulty: string;
};

export type Evaluation = {
  score: number;
  correctness: number;
  depth: number;
  communication: number;
  problem_solving: number;
  strengths: string[];
  weaknesses: string[];
  brief_feedback: string;
  /** True when the LLM failed and a placeholder evaluation was stored. */
  fallback?: boolean;
};

export type HistoryEntry = {
  question: string;
  question_type: string;
  section: string;
  answer: string | null;
  evaluation: Evaluation | null;
};

export type InterviewStatus = "in_progress" | "completed" | "scheduled";

export type InterviewDetail = {
  id: number;
  company: string;
  role: string;
  difficulty: string;
  status: InterviewStatus;
  blueprint: Blueprint | null;
  created_at: string;
  current_question: string | null;
  current_question_type: string | null;
  current_section: string | null;
  history: HistoryEntry[];
};

export type SubmitAnswerResponse = {
  evaluation: Evaluation;
  next_question: string | null;
  next_section: string | null;
  interview_complete: boolean;
  next_acknowledgment: string | null;
  /** True when the report was generated together with the final answer. */
  report_ready: boolean;
};

export type LoginResponse = {
  user_id: number;
  name: string;
  email: string;
  is_new: boolean;
  access_token: string;
  token_type: string;
  /** Token lifetime in seconds. */
  expires_in: number;
};

/** What is kept in localStorage: the login response plus an absolute expiry. */
export type StoredSession = LoginResponse & {
  /** Epoch milliseconds after which the token is considered expired. */
  expires_at: number;
};

export type MeResponse = {
  user_id: number;
  name: string;
  email: string;
};

export type EndInterviewResponse = {
  message: string;
  interview_id: number;
  questions_answered: number;
  /** Null when only a placeholder (fallback) report could be produced. */
  overall_score: number | null;
};

export type CompleteInterviewResponse = {
  message: string;
  interview_id: number;
  /** Null when only a placeholder (fallback) report could be produced. */
  overall_score: number | null;
  already_completed: boolean;
};

export type ReportBody = {
  overall_score?: number;
  strong_topics: string[];
  weak_topics: string[];
  section_scores: Record<string, number>;
  recommendations: string[];
  summary: string;
  /** True when the LLM failed and a placeholder report was stored. */
  fallback?: boolean;
};

export type ReportResponse = {
  interview_id: number;
  company: string;
  role: string;
  overall_score: number | null;
  report: ReportBody;
  created_at: string;
};

export type GenerateReportResponse = {
  message: string;
  interview_id: number;
  /** Null when only a placeholder (fallback) report could be produced. */
  overall_score: number | null;
  report: ReportBody;
};

export type InterviewSummary = {
  id: number;
  company: string;
  role: string;
  difficulty: string | null;
  date: string;
  /** Overall report score (0-100), null until a report exists. */
  score: number | null;
  status: InterviewStatus;
  skills: string[];
};

// ── API Methods ───────────────────────────────────────────────────────────────

export const api = {
  interviews: {
    start: (body: {
      user_id?: number;
      username: string;
      company: string;
      role: string;
      /** Omit (or "auto") to use the researched difficulty. */
      difficulty?: Difficulty | "auto";
    }) =>
      request<StartInterviewResponse>("/api/interviews/start", {
        method: "POST",
        body,
        timeoutMs: START_TIMEOUT_MS,
      }),

    get: (id: number, signal?: AbortSignal) =>
      request<InterviewDetail>(`/api/interviews/${id}`, { signal }),

    list: (params: { limit?: number; offset?: number } = {}) => {
      const query = new URLSearchParams();
      if (params.limit !== undefined) query.set("limit", String(params.limit));
      if (params.offset !== undefined) query.set("offset", String(params.offset));
      const qs = query.toString();
      return request<InterviewSummary[]>(`/api/interviews${qs ? `?${qs}` : ""}`);
    },

    submitAnswer: (id: number, answer: string) =>
      request<SubmitAnswerResponse>(`/api/interviews/${id}/answer`, {
        method: "POST",
        body: { answer },
        timeoutMs: LLM_TIMEOUT_MS,
      }),

    complete: (id: number) =>
      request<CompleteInterviewResponse>(`/api/interviews/${id}/complete`, {
        method: "POST",
        timeoutMs: LLM_TIMEOUT_MS,
      }),

    endEarly: (id: number) =>
      request<EndInterviewResponse>(`/api/interviews/${id}/end`, {
        method: "POST",
        timeoutMs: LLM_TIMEOUT_MS,
      }),
  },

  reports: {
    /** May generate the report on the fly for completed interviews. */
    get: (interviewId: number, signal?: AbortSignal) =>
      request<ReportResponse>(`/api/reports/${interviewId}`, {
        timeoutMs: LLM_TIMEOUT_MS,
        signal,
      }),

    generate: (interviewId: number) =>
      request<GenerateReportResponse>(`/api/reports/${interviewId}`, {
        method: "POST",
        timeoutMs: LLM_TIMEOUT_MS,
      }),
  },

  auth: {
    login: (body: { name: string; email: string }) =>
      request<LoginResponse>("/api/auth/login", {
        method: "POST",
        body,
        authenticated: false,
      }),

    me: () => request<MeResponse>("/api/auth/me"),
  },
};

// ── Session storage ───────────────────────────────────────────────────────────

function isStoredSession(value: unknown): value is StoredSession {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.user_id === "number" &&
    typeof v.name === "string" &&
    typeof v.email === "string" &&
    typeof v.access_token === "string" &&
    v.access_token.length > 0 &&
    typeof v.expires_at === "number"
  );
}

export const auth = {
  save: (data: LoginResponse) => {
    const session: StoredSession = {
      ...data,
      expires_at: Date.now() + Math.max(0, data.expires_in) * 1000,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  },

  /**
   * The stored session, or null when there is none, it is malformed, it
   * predates token auth (no access_token) or it has expired. Invalid entries
   * are removed so the user is sent to /login.
   */
  load: (): StoredSession | null => {
    if (typeof window === "undefined") return null;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    if (!isStoredSession(parsed) || parsed.expires_at <= Date.now()) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return parsed;
  },

  clear: () => {
    if (typeof window === "undefined") return;
    localStorage.removeItem(STORAGE_KEY);
  },

  isLoggedIn: (): boolean => {
    return !!auth.load();
  },

  /**
   * The raw stored JSON when it holds a valid, unexpired session, else null
   * (invalid entries are removed, as in `load`). Returning the string keeps
   * the value referentially stable, which `useSyncExternalStore` requires.
   */
  loadRaw: (): string | null => {
    if (!auth.load()) return null;
    return localStorage.getItem(STORAGE_KEY);
  },

  /** Subscribe to session changes made in other tabs. Returns an unsubscribe function. */
  subscribe: (onChange: () => void): (() => void) => {
    if (typeof window === "undefined") return () => {};
    const handler = (event: StorageEvent) => {
      if (event.key === null || event.key === STORAGE_KEY) onChange();
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  },
};
