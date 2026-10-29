# InterVu

An AI-powered mock interview platform. Pick a company and a role: InterVu researches real interview experiences on the web, builds an interview plan, runs a conversational interview with you turn by turn, scores every answer and finishes with a detailed feedback report.

<img width="6838" height="4304" alt="InterVu architecture sketch" src="https://github.com/user-attachments/assets/a11082f2-0db6-4d18-b936-51be2f68eba1" />

## Contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [API reference](#api-reference)
- [Authentication model](#authentication-model)
- [Testing and CI](#testing-and-ci)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## How it works

1. **Research.** Tavily runs several concurrent web searches for interview experiences at the company/role, Firecrawl scrapes the pages and Anakin scrapes Reddit threads. An extractor agent turns the content into structured knowledge (skills, topics, rounds, DSA questions, difficulty). The result is cached per company/role in the `research_profiles` table, so repeat interviews skip this step.
2. **Knowledge base.** Scraped content is embedded with Google `gemini-embedding-001` and stored in a local ChromaDB collection used for retrieval during the interview.
3. **Blueprint.** A blueprint agent plans the interview sections (screening, coding, technical, system design, behavioral) and how many questions each gets, at the difficulty you picked (or the researched one).
4. **Interview.** A conductor agent running inside a LangGraph state machine asks one question at a time. It sees the conversation so far, the blueprint and per-section coverage, and pulls context from ChromaDB.
5. **Evaluation.** Every answer is scored 0-10 on correctness, depth, communication and problem solving, with strengths, weaknesses and brief feedback. Difficulty adapts to how you are doing.
6. **Report.** When the interview ends (naturally or early), a feedback agent produces an overall score (0-100), per-section scores, strong and weak topics, recommendations and a summary. You can download it as JSON or print it to PDF.

---

## Architecture

```text
                         Browser (Next.js 16, React 19)
                                     |
                 Authorization: Bearer <token>  (JSON over HTTP)
                                     v
+------------------------------------------------------------------------+
| FastAPI backend (backend/main.py)                                      |
|   /api/auth  /api/interviews  /api/reports  /health  /ready            |
|                                                                        |
|   ResearchPipeline --> Tavily (search), Firecrawl + Anakin (scrape)    |
|          |                                                             |
|          v                                                             |
|   Extractor, Blueprint, Conductor, Evaluator, Feedback agents          |
|   (all openai/gpt-oss-120b on Groq, configured in backend/config.py)   |
|          |                                                             |
|   LangGraph interview graph  <-->  AsyncPostgresSaver checkpoints      |
|   ChromaDB (local, ./chroma_db) <-- Google gemini-embedding-001        |
+------------------------------------------------------------------------+
                  |                                   ^
                  v                                   | read-only
          PostgreSQL 17 (users, interviews,           |
          questions, responses, reports,     intervu-admin (Django 5.1 +
          research_profiles, graph checkpoints)  DRF): admin UI + REST API
```

| Agent | Model | Provider |
|-------|-------|----------|
| Extractor | `openai/gpt-oss-120b` | Groq |
| Blueprint generator | `openai/gpt-oss-120b` | Groq |
| Conductor (interviewer) | `openai/gpt-oss-120b` | Groq |
| Evaluator | `openai/gpt-oss-120b` | Groq |
| Feedback | `openai/gpt-oss-120b` | Groq |
| Embeddings | `gemini-embedding-001` | Google AI |

Model names live in `backend/config.py`. LLM calls go through `backend/utils/llm_json.py`, which retries with exponential backoff, applies a per-attempt timeout and robustly extracts JSON from the model output. When an evaluation or report cannot be produced, a placeholder is stored with `"fallback": true` and the UI says so.

### Tech stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Recharts |
| Backend | Python 3.11, FastAPI, LangGraph, SQLAlchemy 2 (async, asyncpg) |
| LLMs | Groq (`openai/gpt-oss-120b`) |
| Embeddings / vector DB | Google `gemini-embedding-001`, ChromaDB (local persistent) |
| Graph state | LangGraph `AsyncPostgresSaver` (psycopg) |
| Database | PostgreSQL 17 (docker compose locally; any Postgres such as Neon works) |
| Web research | Tavily, Firecrawl, Anakin |
| Admin panel | Django 5.1, Django REST Framework, django-filter, SimpleJWT |
| Tooling | pytest, ESLint, GitHub Actions, Docker, Makefile |

---

## Quick start

Prerequisites: Python 3.11, Node.js 20, Docker (for the local database), and API keys for Groq, Google AI, Tavily, Firecrawl and Anakin.

A `Makefile` wraps every command below (`make help` lists them). Commands are shown for macOS/Linux; on Windows activate virtualenvs with `.venv\Scripts\activate`.

### 1. Database

```bash
# From the repository root
cp .env.example .env          # then set POSTGRES_PASSWORD (URL-safe characters only)
docker compose up -d postgres # or: make db-up
```

Postgres is published on `127.0.0.1:5432` only and has a `pg_isready` healthcheck.

### 2. Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
# Fill in API keys, put POSTGRES_PASSWORD into DATABASE_URL and
# DATABASE_URL_PSYCOPG, keep ENV=development and set AUTH_SECRET:
#   python3 -c "import secrets;print(secrets.token_urlsafe(48))"
# (`make setup-backend` or `make backend-env` creates backend/.env and
# generates AUTH_SECRET for you when it is blank.)

uvicorn main:app --reload --host 127.0.0.1 --port 8000   # or: make backend
```

**`ENV` fails closed.** Only `development`, `dev`, `test` and `local` count as development. With `ENV` unset (or any other value) the backend behaves as production: it refuses to start without `AUTH_SECRET`, ignores `AUTH_DISABLED` and does not add the localhost CORS origins. Without a fixed `AUTH_SECRET` in development, every `--reload` signs everybody out.

Tables and the LangGraph checkpoint tables are created on startup. Check `http://127.0.0.1:8000/ready`; it returns 200 once the database and graph are ready.

**Upgrading an existing `backend/.env`.** A `.env` created before `ENV` fails closed has no `ENV` line, so the backend now starts in production mode. Add `ENV=development` and a non-empty `AUTH_SECRET` to it (`make backend` runs `make backend-env`, which generates a blank `AUTH_SECRET` and warns when `ENV` is missing).

#### Upgrading an existing database

`create_all` only creates missing tables; it does not add indexes or change foreign keys on tables that already exist. If your database predates this version, run this once (for example with `docker compose exec postgres psql -U postgres -d intervu`):

```sql
-- Indexes added for the per-user and per-interview queries.
CREATE INDEX IF NOT EXISTS ix_interviews_user_id ON interviews (user_id);
CREATE INDEX IF NOT EXISTS ix_questions_interview_id ON questions (interview_id);
CREATE INDEX IF NOT EXISTS ix_responses_question_id ON responses (question_id);

-- ON DELETE CASCADE on the child foreign keys (the ORM relies on it).
-- Constraint names below are Postgres' defaults; check yours with \d <table>.
ALTER TABLE interviews DROP CONSTRAINT IF EXISTS interviews_user_id_fkey,
  ADD CONSTRAINT interviews_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE;
ALTER TABLE questions DROP CONSTRAINT IF EXISTS questions_interview_id_fkey,
  ADD CONSTRAINT questions_interview_id_fkey FOREIGN KEY (interview_id) REFERENCES interviews (id) ON DELETE CASCADE;
ALTER TABLE responses DROP CONSTRAINT IF EXISTS responses_question_id_fkey,
  ADD CONSTRAINT responses_question_id_fkey FOREIGN KEY (question_id) REFERENCES questions (id) ON DELETE CASCADE;
ALTER TABLE reports DROP CONSTRAINT IF EXISTS reports_interview_id_fkey,
  ADD CONSTRAINT reports_interview_id_fkey FOREIGN KEY (interview_id) REFERENCES interviews (id) ON DELETE CASCADE;

-- Placeholder (fallback) reports no longer count as a score of 0.
UPDATE reports SET overall_score = NULL WHERE report_json->>'fallback' = 'true';

-- Optional: drop research profiles cached from a failed (empty) extraction.
DELETE FROM research_profiles
WHERE skills::text = '[]' AND topics::text = '[]' AND rounds::text = '[]';
```

The repository has an `alembic/` environment but no revisions; the SQL above is the supported upgrade path for now.

### 3. Frontend

```bash
cd frontend
npm ci
cp .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                  # or: make frontend
```

Open `http://localhost:3000`. The `dev` and `start` scripts bind to `127.0.0.1`.

### 4. Admin panel (optional)

A separate, read-only Django service over the same database. See [intervu-admin/README.md](intervu-admin/README.md) for details.

```bash
cd intervu-admin
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # keep DJANGO_DEBUG=True for local dev
python manage.py migrate          # Django's own tables only, never InterVu's
python manage.py createsuperuser
python manage.py runserver 127.0.0.1:8002   # or: make admin
```

### Everything in containers (optional)

```bash
cp .env.example .env                 # POSTGRES_PASSWORD
cp backend/.env.example backend/.env # API keys, AUTH_SECRET
docker compose --profile app up -d --build   # or: make up
```

This builds `backend/Dockerfile` and `frontend/Dockerfile` and starts Postgres, the backend (`127.0.0.1:8000`) and the frontend (`127.0.0.1:3000`). Compose points the backend at the `postgres` service, disables TLS for that internal connection and keeps ChromaDB data in a named volume. All ports are published on loopback only; to reach the stack from another machine use an SSH tunnel rather than publishing wider.

Under compose, `ENV` comes from the root `.env` (default `development`), not from `backend/.env`: the value set in `docker-compose.yaml` takes precedence over the `env_file`, so `ENV=production` in `backend/.env` is ignored there.

---

## Environment variables

### Root `.env` (docker compose only)

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_PASSWORD` | yes | Password for the local Postgres container. URL-safe characters only. |
| `ENV` | no | `ENV` passed to the compose backend (default `development`). Set it here, not in `backend/.env`, when using compose. |
| `FRONTEND_URL` | no | CORS origin passed to the compose backend (default `http://localhost:3000`). |
| `NEXT_PUBLIC_API_URL` | no | Backend URL baked into the compose frontend build (default `http://localhost:8000`). |

### `backend/.env`

See `backend/.env.example` for the full, commented list.

| Variable | Description |
|----------|-------------|
| `ENV` | Fails closed: `development`, `dev`, `test` or `local` enable development behaviour. Unset or anything else (for example `production`) is production: `AUTH_SECRET` required, `AUTH_DISABLED` ignored, no localhost CORS defaults. `.env.example` sets `development`. |
| `FRONTEND_URL` | Comma-separated browser origins allowed by CORS. |
| `LOG_LEVEL` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. |
| `AUTH_SECRET` | Key that signs access tokens. Required outside development; `make backend-env` generates one. Without it in development a random per-process key is used (tokens break on every restart or reload). |
| `AUTH_TOKEN_TTL_HOURS` | Token lifetime, default `168` (one week). |
| `AUTH_DISABLED` | `true` skips token checks for local debugging. Ignored unless `ENV` is a development value. |
| `RATE_LIMIT_LOGIN_PER_MIN`, `RATE_LIMIT_START_PER_HOUR` | In-memory sliding-window limits: sign-ins per client IP per minute (default 10) and interview starts per user per hour (default 20). `0` disables a limit. |
| `RATE_LIMIT_ENABLED` | `true`/`false` forces rate limiting on or off. Default: on, except `ENV=test`. |
| `DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql+asyncpg://postgres:<password>@localhost:5432/intervu`. |
| `DATABASE_URL_PSYCOPG` | Same database for the LangGraph checkpointer, e.g. `postgresql://postgres:<password>@localhost:5432/intervu`. |
| `DB_SSL` | `require` (encrypt, no certificate check), `verify-ca` (verify the certificate chain), `verify-full` (chain and host name) or `disable`. Unset: `require` for every host except localhost and the compose `postgres` service. Unknown values fail at startup. Applies to `DATABASE_URL`; put `sslmode=...` in `DATABASE_URL_PSYCOPG` for the checkpointer. |
| `DB_SSL_ROOT_CERT` | Optional CA bundle for `verify-ca` / `verify-full` (default: system trust store). |
| `GROQ_API_KEY`, `GROQ_API_KEY_1`, `GROQ_API_KEY_2` | Groq keys. The extractor rotates over the numbered keys; other agents use `GROQ_API_KEY_2`; `GROQ_API_KEY` is the fallback. |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_ATTEMPTS` | Per-attempt LLM timeout (default 40) and attempts per call (default 2). Raising them may exceed the frontend timeouts (180 s for answers and reports, 420 s for starting an interview). |
| `GOOGLE_API_KEY`, `GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`, ... | Google AI keys for embeddings (numbered keys are rotated). |
| `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, `ANAKIN_API_KEY` | Web search and scraping. |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | ChromaDB directory (default `./chroma_db`) and collection (default `intervu_kb_v2`). |

### `frontend/.env.local`

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend base URL (default `http://localhost:8000`). Inlined into the browser bundle, so never put secrets here. |

### `intervu-admin/.env`

| Variable | Description |
|----------|-------------|
| `DJANGO_DEBUG` | Defaults to `False`. Use `True` for local development. |
| `DJANGO_SECRET_KEY` | Required when `DJANGO_DEBUG=False` (startup fails without it). |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated hosts, default `localhost,127.0.0.1`. |
| `DJANGO_CORS_ALLOWED_ORIGINS` | Comma-separated origins, default `http://localhost:3000,http://127.0.0.1:3000`. |
| `DJANGO_SECURE_COOKIES` | HTTPS-only cookies; defaults to `True` when debug is off. |
| `INTERVU_DB_ENGINE`, `INTERVU_DB_NAME`, `INTERVU_DB_USER`, `INTERVU_DB_PASSWORD`, `INTERVU_DB_HOST`, `INTERVU_DB_PORT` | Connection to the InterVu database (use a read-only role). Unset: a local SQLite fixture DB. |

---

## API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`. Every endpoint except `/api/auth/login`, `/health` and `/ready` needs `Authorization: Bearer <access_token>`. Requests for interviews owned by another user return 404.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness: the process is up. |
| `GET` | `/ready` | Readiness: database answers and the interview graph is initialised (503 otherwise). |
| `POST` | `/api/auth/login` | `{name, email}`. Creates or finds the user and returns `user_id, name, email, is_new, access_token, token_type, expires_in`. Rate limited per client IP (429 with `Retry-After`). |
| `GET` | `/api/auth/me` | The user behind the token. |
| `GET` | `/api/interviews?limit=&offset=` | Your interviews, newest first, with `difficulty`, `status` and report `score` (null until a real report exists; placeholder reports have no score). `limit` 1-100, default 50. |
| `POST` | `/api/interviews/start` | `{username, company, role, difficulty?}`. Runs research (or uses the cache), builds the blueprint and returns the first question. `difficulty` is `Easy`, `Medium`, `Hard`, or omitted/`auto` for the researched value. Rate limited per user (429). |
| `GET` | `/api/interviews/{id}` | Interview state and full Q&A history (used to resume). |
| `POST` | `/api/interviews/{id}/answer` | `{answer}`. Returns the evaluation and the next question, or `interview_complete: true` and `report_ready` (false when the report failed or is only a placeholder). 409 while another answer for the same interview is still being processed. |
| `POST` | `/api/interviews/{id}/complete` | Mark completed and generate the report. Idempotent once a real report exists. 400 when nothing was answered. |
| `POST` | `/api/interviews/{id}/end` | End early and generate a report from the answers so far. Idempotent once completed with a real report. 400 when nothing was answered. |
| `GET` | `/api/reports/{id}` | The feedback report. Generated on the fly for completed interviews that have no report yet or only a placeholder (`report.fallback: true`, `overall_score: null`). |
| `POST` | `/api/reports/{id}` | Regenerate the report of a completed interview. |

Error bodies are JSON with a `detail` field. Unhandled errors return 500 with a `request_id`; every response carries an `X-Request-ID` header that also appears in the backend logs.

---

## Authentication model

Sign-in is **passwordless**: you enter a name and an email and the backend issues a self-managed, HMAC-SHA256 signed bearer token (`backend/utils/auth.py`) that expires after `AUTH_TOKEN_TTL_HOURS`. The frontend keeps it in `localStorage`, sends it on every request and returns to the login page on any 401 or when the token has expired.

Limitations you should know about:

- There is **no password and no email verification**. Anyone who types an email address can sign in as that user and see their interviews. The token only proves "signed in with this email", not ownership of the mailbox. The login response's `is_new` field also tells any caller whether an email is already registered.
- Sign-ins (per client IP) and interview starts (per user) are rate limited in memory (`RATE_LIMIT_LOGIN_PER_MIN`, `RATE_LIMIT_START_PER_HOUR`). The counters are per process and the client IP is the direct peer address (`X-Forwarded-For` is not trusted), so this slows down casual abuse of your API keys but is not a substitute for a gateway-level limit in a shared deployment.
- Tokens cannot be revoked individually; rotating `AUTH_SECRET` invalidates all of them.
- `localStorage` tokens are readable by any script running on the page.

This is fine for local, single-user practice. **Before any public deployment, replace it with a real identity provider using OIDC/OAuth 2.0** (for example Auth0, Clerk, Cognito or Google sign-in) and validate the provider's tokens in the backend instead of minting your own.

The admin panel uses Django's own authentication (session cookies plus SimpleJWT) with staff-only permissions.

---

## Testing and CI

```bash
cd backend && python -m pytest -q            # or: make test
cd frontend && npm run lint && npm run build && npx tsc --noEmit
cd intervu-admin && DJANGO_DEBUG=True python manage.py check
make lint                                    # all of the static checks above
```

The backend suite (`backend/tests`) mocks every LLM, search API, the LangGraph graph and the database, so it needs no API keys or running services (`tests/conftest.py` sets `ENV=test`). It covers token signing and expiry, `ENV` fail-closed behaviour, JSON extraction, difficulty handling, graph helpers, ownership checks, natural completion creating a report, lazy report generation, fallback-report regeneration, idempotent completion, the empty-history guards, rate limiting, CORS origins and CORS headers on 500 responses, `DB_SSL` modes, research-profile caching and `/ready`.

GitHub Actions (`.github/workflows/ci.yml`, read-only `contents` permission) runs three jobs on every push to `main` and every pull request:

- **backend**: Python 3.11, install requirements, byte-compile, `pytest`.
- **frontend**: Node 20, `npm ci`, lint, build, `tsc --noEmit`.
- **admin**: install requirements, `manage.py check` with `DJANGO_DEBUG=False`.

---

## Project structure

```text
InterVu/
├── backend/                  FastAPI + LangGraph service
│   ├── main.py               App, CORS, request-id logging, /health, /ready
│   ├── config.py             Model names, Groq key helpers, Chroma settings
│   ├── api/                  Routers: auth.py, interview.py, report.py
│   ├── agents/               conductor, evaluator, feedback, extractor, blueprint, registry
│   ├── graph/                interview_graph.py (LangGraph state machine + checkpointer)
│   ├── services/             reports.py (build, store and load reports)
│   ├── rag/                  ChromaDB vector store and retriever
│   ├── search/               tavily.py, firecrawl.py, anakin.py
│   ├── normalization/        Round name canonicalisation and de-duplication
│   ├── pipeline.py           End-to-end async research pipeline
│   ├── research.py           Research profile cache
│   ├── db/postgres.py        SQLAlchemy models and session
│   ├── schemas/, prompts/    Pydantic schemas and LLM prompts
│   ├── utils/                auth, llm_json, difficulty, logging, locks, rate_limit, embedding key pool
│   ├── tests/                pytest suite (fully mocked)
│   ├── Dockerfile
│   ├── requirements.txt      Runtime dependencies (authoritative)
│   └── requirements-dev.txt  Test dependencies
├── frontend/                 Next.js 16 app
│   ├── app/login/            Sign-in page
│   ├── app/(protected)/      Auth-guarded pages: dashboard, interview/new, interview/[id], report/[id]
│   ├── components/           Layout and UI components
│   ├── lib/api.ts            Typed API client (bearer token, timeouts, ApiError)
│   ├── hooks/useAuth.ts      Session guard
│   └── Dockerfile
├── intervu-admin/            Read-only Django admin + REST API (separate venv)
├── .github/workflows/ci.yml  CI pipeline
├── docker-compose.yaml       Postgres (+ optional backend/frontend profile "app")
├── Makefile                  Developer commands
├── .env.example              Root env for docker compose
└── requirements.txt          Points at backend/requirements.txt
```

---

## Troubleshooting

**"This interview's session state was lost" (HTTP 410).** The LangGraph checkpoint for that interview is missing, for example after the checkpoint tables were reset or the backend pointed at a different database. Your answered questions are still stored: the interview page offers "Get report for answered questions" (which calls `/end`) or a link to start a new interview.

**`DATABASE_URL_PSYCOPG environment variable is not set.`** The backend refuses to start without it because the interview graph checkpoints to Postgres. Set it in `backend/.env` (same database as `DATABASE_URL`, without `+asyncpg`).

**SSL errors against a local Postgres.** TLS is required for every non-local host. Local hosts (`localhost`, `127.0.0.1`, `::1`, `postgres`) skip it automatically; for any other non-TLS server set `DB_SSL=disable`. For hosted Postgres such as Neon keep TLS on (`DB_SSL=require` or unset).

**`docker compose` says `Set POSTGRES_PASSWORD in .env`.** Copy `.env.example` to `.env` in the repository root and set a password.

**Redirected to the login page right after signing in.** The backend rejected the token (401). Usually `AUTH_SECRET` is unset and the backend restarted or reloaded, which invalidates every token. Set a fixed `AUTH_SECRET` in `backend/.env` (`make backend-env` generates one).

**`RuntimeError: AUTH_SECRET must be set unless ENV is development...` at startup.** `ENV` is unset or not a development value, so the backend runs as production. For local work set `ENV=development` in `backend/.env`; for a real deployment set `AUTH_SECRET`.

**HTTP 429 "Too many ... requests".** A rate limit was hit; wait for the `Retry-After` seconds or raise `RATE_LIMIT_LOGIN_PER_MIN` / `RATE_LIMIT_START_PER_HOUR`.

**"Your answer is still being processed" (HTTP 409).** The previous submission of an answer is still running (for example after a client timeout). Wait a moment and reload the interview page.

**A company/role always gets a generic interview plan.** Older versions could cache an empty research profile after an LLM or scraper outage. Delete that row from `research_profiles` (see [Upgrading an existing database](#upgrading-an-existing-database)); empty extractions are no longer cached.

**CORS errors in the browser console.** Add the exact frontend origin (scheme, host and port) to `FRONTEND_URL` in `backend/.env`.

**A report says it is a placeholder.** The feedback LLM call failed after retries. Use "Regenerate report" on the report page once the provider is reachable.

---

## Roadmap

- Replace passwordless sign-in with an OIDC identity provider.
- Alembic migrations for schema changes (tables are currently created with `create_all`, which does not alter existing tables).
- Stream interviewer questions and research progress to the browser instead of a simulated progress animation.
- Voice answers (speech to text) and timed coding questions.
- Resume analysis and settings pages (shown as "Soon" in the sidebar).
- Frontend unit and end-to-end tests.
