"""
Interview API Router
--------------------
All endpoints require ``Authorization: Bearer <token>`` (see api/auth.py) and
only ever expose interviews owned by the authenticated user. Interviews owned
by someone else are reported as 404 so ids cannot be probed.

Endpoints:
  GET  /api/interviews                – list the caller's interviews (paginated)
  POST /api/interviews/start          – create session + run research pipeline
  GET  /api/interviews/{id}           – get interview state
  POST /api/interviews/{id}/answer    – submit an answer, get next question
                                         (generates the report on natural completion)
  POST /api/interviews/{id}/complete  – mark interview done (idempotent)
  POST /api/interviews/{id}/end       – end early, generate report based on answers so far

Persistence: every mutating handler calls ``db.commit()`` itself before
returning. ``get_db`` also commits, but on recent FastAPI versions the exit
code of a yield dependency runs after the response has been sent, so a
failure there would be invisible to the client and an immediate follow-up
request (for example the report page opened right after the final answer)
could miss the data. The second commit in ``get_db`` is then a harmless no-op.

Concurrency: work that advances an interview's graph or writes its report
runs under a per-interview in-process lock (utils/locks.py). A second answer
submitted while one is in flight gets 409.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.registry import blueprint_generator
from db.postgres import Interview, Question, Report, Response, get_db
from graph.interview_graph import InterviewState, get_compiled_graph, graph_config
from pipeline import ResearchPipeline
from research import (
    get_research_profile,
    is_cacheable,
    is_stale,
    save_research_profile,
    update_research_profile,
)
from schemas.interiew import (
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
)
from services.reports import (
    build_and_store_report,
    collect_qa_history,
    get_report_row,
    is_usable_report,
    isoformat,
)
from utils.auth import CurrentUser, get_current_user
from utils.difficulty import normalize_difficulty
from utils.locks import interview_locks
from utils.rate_limit import limit_start

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/interviews", tags=["interviews"])

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100

# ── Helpers ────────────────────────────────────────────────────────────────────


async def get_owned_interview_or_404(
    interview_id: int,
    db: AsyncSession,
    user: CurrentUser,
) -> Interview:
    """Fetch an interview owned by ``user`` or raise 404 (also for other users' ids)."""
    interview = await db.get(Interview, interview_id)
    if interview is None or not user.owns(interview.user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    return interview


def resolve_owner_id(requested_user_id: int | None, user: CurrentUser) -> int:
    """
    The user id to act as. With a token this is always the token's user and a
    conflicting ``requested_user_id`` is rejected with 403. With
    AUTH_DISABLED and no token, ``requested_user_id`` is required.
    """
    if user.id is not None:
        if requested_user_id is not None and requested_user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="user_id does not match the authenticated user",
            )
        return user.id

    if requested_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required when authentication is disabled",
        )
    return requested_user_id


async def _mark_graph_complete(interview_id: int) -> None:
    """Best-effort: flag the LangGraph thread as complete."""
    try:
        compiled_graph = get_compiled_graph()
        await compiled_graph.aupdate_state(graph_config(interview_id), {"interview_complete": True})
    except Exception as exc:  # noqa: BLE001 - the DB status is the source of truth
        logger.warning("Could not mark graph state complete for interview %s: %s", interview_id, exc)


def _profile_to_extracted(profile: Any) -> dict[str, Any]:
    """The extractor-shaped dict for a stored ResearchProfile."""
    return {
        "skills": profile.skills,
        "technologies": profile.technologies,
        "topics": profile.topics,
        "rounds": profile.rounds,
        "behavioral_patterns": profile.behavioral_patterns,
        "key_insights": profile.key_insights,
        "dsa_questions": profile.dsa_questions,
        "difficulty": profile.difficulty,
    }


async def _run_research(company: str, role: str) -> dict[str, Any] | None:
    """
    Run the research pipeline. Returns the extracted knowledge, or None when
    the pipeline could not even run (for example a missing API key), so the
    caller can degrade instead of returning a 500.
    """
    try:
        pipeline = ResearchPipeline()
        result = await pipeline.run(company, role)
        return result["extracted"]
    except Exception:  # noqa: BLE001 - research is best-effort
        logger.exception("Research pipeline failed for %s / %s.", company, role)
        return None


def _blueprint_skills(blueprint: dict[str, Any] | None, limit: int = 4) -> list[str]:
    skills: list[str] = []
    for section in (blueprint or {}).get("sections", []):
        skills.extend(section.get("focus_areas", [])[:2])
    return skills[:limit]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/start", response_model=StartInterviewResponse, dependencies=[Depends(limit_start)])
async def start_interview(
    body: StartInterviewRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StartInterviewResponse:
    """
    1. Check ResearchProfile cache
    2. Run pipeline if cache miss
    3. Generate blueprint (using the requested difficulty, or the researched one)
    4. Initialize LangGraph state, retrieve context, generate first question
    5. Persist Interview + Question to DB
    """
    owner_id = resolve_owner_id(body.user_id, user)
    company = body.company
    role = body.role

    warnings: list[str] = []
    profile = await get_research_profile(db, company, role)
    extracted: dict[str, Any] | None = None
    research_failed = False

    if profile is not None:
        extracted = _profile_to_extracted(profile)
        if body.refresh_research or is_stale(profile):
            logger.info(
                "Research profile for %s / %s is %s; refreshing.",
                company, role, "being force-refreshed" if body.refresh_research else "stale",
            )
            fresh = await _run_research(company, role)
            if fresh is not None and is_cacheable(fresh):
                profile = await update_research_profile(db, profile, fresh)
                extracted = _profile_to_extracted(profile)
            else:
                # Keep serving the old profile rather than failing the interview.
                warnings.append(
                    f"Saved research for {company} could not be refreshed, so questions use older information."
                )
        else:
            logger.info("Research profile found for %s / %s, skipping pipeline.", company, role)
    else:
        logger.info("No research profile for %s / %s, running pipeline.", company, role)
        fresh = await _run_research(company, role)
        if fresh is not None and is_cacheable(fresh):
            extracted = fresh
            await save_research_profile(db, company, role, extracted)
        else:
            # Never cache an empty or failed extraction: the next start retries research.
            logger.warning(
                "Research for %s / %s produced no skills, topics or rounds; not caching it.",
                company, role,
            )
            research_failed = True
            extracted = fresh or {}
            warnings.append(
                f"Live research on {company} was unavailable, so the questions are general rather than tailored to {company}."
            )

    # The candidate's explicit choice wins over the researched estimate.
    difficulty = body.difficulty or normalize_difficulty(extracted.get("difficulty"))

    blueprint = await blueprint_generator.generate(
        company=company,
        role=role,
        skills=extracted.get("skills", []),
        topics=extracted.get("topics", []),
        rounds=extracted.get("rounds", []),
        dsa_questions=extracted.get("dsa_questions", []),
        difficulty=difficulty,
    )

    blueprint_failed = bool(blueprint.get("fallback"))
    if blueprint_failed:
        warnings.append("The AI interview planner was unavailable, so a standard interview plan is being used.")

    if research_failed and blueprint_failed:
        # Both the research and the LLM are down: an interview now would be a
        # generic script with no AI behind it. Fail fast with a clear message
        # instead of silently starting a degraded session.
        logger.error("Refusing to start an interview for %s / %s: research and planner both failed.", company, role)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The research and AI services are unavailable right now, so a useful interview "
                "cannot be started. Check the API keys and network access, then try again."
            ),
        )

    if warnings:
        # Persisted with the blueprint so a resumed interview still shows the notice.
        blueprint = {**blueprint, "degraded": True, "warnings": list(warnings)}

    interview = Interview(
        user_id=owner_id,
        company=company,
        role=role,
        difficulty=difficulty,
        blueprint=blueprint,
        status="in_progress",
    )

    db.add(interview)
    await db.flush()

    initial_state: InterviewState = {
        "company": company,
        "role": role,
        "difficulty": difficulty,
        "username": body.username,
        "blueprint": blueprint,
        "context": "",
        "current_question": "",
        "current_question_type": "",
        "current_section": "",
        "current_focus_area": "",
        "current_acknowledgment": "",
        "current_answer": "",
        "evaluation": {},
        "previous_questions": [],
        "qa_history": [],
        "llm_suggests_wrap_up": False,
        "interview_complete": False,
    }

    compiled_graph = get_compiled_graph()
    state = await compiled_graph.ainvoke(initial_state, config=graph_config(interview.id))

    first_question = state["current_question"]
    question_type = state.get("current_question_type", "screening")
    section_name = state.get("current_section", "General")

    question = Question(
        interview_id=interview.id,
        question=first_question,
        section=section_name,
        question_type=question_type,
        order_index=0,
    )

    db.add(question)
    await db.flush()
    await db.commit()  # Persist before the response is sent (see "Persistence" above).

    return StartInterviewResponse(
        interview_id=interview.id,
        first_question=first_question,
        blueprint=blueprint,
        difficulty=difficulty,
        degraded=bool(warnings),
        warnings=warnings,
    )


@router.get("/{interview_id}")
async def get_interview(
    interview_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get the current state of an interview session."""
    interview = await get_owned_interview_or_404(interview_id, db, user)

    # One query for every question and its (optional) response.
    result = await db.execute(
        select(Question, Response)
        .outerjoin(Response, Response.question_id == Question.id)
        .where(Question.interview_id == interview.id)
        .order_by(Question.order_index.asc(), Question.id.asc())
    )

    # Keyed by question id: the outer join yields one row per (question, response).
    ordered: dict[int, list[Any]] = {}
    for question, response in result.all():
        entry = ordered.setdefault(question.id, [question, None])
        if response is not None:
            entry[1] = response

    history = []
    for question, response in ordered.values():
        history.append({
            "question": question.question,
            "answer": response.answer if response else None,
            "question_type": question.question_type,
            "section": question.section,
            "evaluation": response.evaluation if response else None,
        })

    questions = [question for question, _ in ordered.values()]
    latest_question = questions[-1] if questions else None

    return {
        "id": interview.id,
        "company": interview.company,
        "role": interview.role,
        "difficulty": interview.difficulty,
        "status": interview.status,
        "blueprint": interview.blueprint,
        "created_at": isoformat(interview.created_at),
        "current_question": latest_question.question if latest_question else None,
        "current_question_type": latest_question.question_type if latest_question else None,
        "current_section": latest_question.section if latest_question else None,
        "history": history,
    }


@router.post("/{interview_id}/answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    interview_id: int,
    body: SubmitAnswerRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubmitAnswerResponse:
    """
    1. Load current graph state via checkpointer
    2. Inject user answer into state
    3. Run evaluate_answer, adjust_difficulty, store_result
    4. If not complete: run retrieve_context, generate_question
    5. Persist Response + next Question to DB
    6. If complete: mark the interview completed and generate the report
    """
    interview = await get_owned_interview_or_404(interview_id, db, user)
    if interview.status == "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview already completed")

    # Checking and taking the lock happen without an await in between, so two
    # overlapping submissions (for example a resubmit after a client timeout)
    # can never both advance the graph.
    if interview_locks.is_locked(interview.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An answer for this interview is still being processed. Please wait for it to finish.",
        )
    async with interview_locks.hold(interview.id):
        return await _advance_with_answer(db, interview, body.answer)


async def _advance_with_answer(db: AsyncSession, interview: Interview, answer: str) -> SubmitAnswerResponse:
    """Body of POST /answer; runs under the interview's lock."""
    config = graph_config(interview.id)
    compiled_graph = get_compiled_graph()
    current_state = await compiled_graph.aget_state(config)
    if not current_state or not current_state.values or "blueprint" not in current_state.values:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This interview's session state was lost (likely due to a server restart). Please start a new interview.",
        )

    await compiled_graph.aupdate_state(config, {"current_answer": answer})

    final_state = await compiled_graph.ainvoke(None, config=config)

    evaluation = final_state.get("evaluation", {}) or {}
    interview_complete = bool(final_state.get("interview_complete", False))

    result = await db.execute(
        select(Question)
        .where(Question.interview_id == interview.id)
        .order_by(Question.order_index.desc(), Question.id.desc())
        .limit(1)
    )

    last_question = result.scalar_one_or_none()
    if last_question:
        response = Response(
            question_id=last_question.id,
            answer=answer,
            score=evaluation.get("score"),
            evaluation=evaluation,
            feedback=evaluation.get("brief_feedback", ""),
        )
        db.add(response)
        await db.flush()

    next_question = None
    next_section = None
    next_acknowledgment = None
    report_ready = False

    if not interview_complete:
        next_q_type = final_state.get("current_question_type", "screening")
        next_q_text = final_state.get("current_question", "N/A")
        section_name = final_state.get("current_section", "General")
        next_acknowledgment = final_state.get("current_acknowledgment", "")
        order_index = len(final_state.get("previous_questions", []))
        next_question_row = Question(
            interview_id=interview.id,
            question=next_q_text,
            section=section_name,
            question_type=next_q_type,
            order_index=order_index,
        )
        db.add(next_question_row)
        await db.flush()
        next_question = next_q_text
        next_section = section_name
    else:
        interview.status = "completed"
        await db.flush()
        # Generate the report now. A failure here must never lose the saved
        # Response: it runs in a SAVEPOINT and GET /api/reports/{id} will
        # lazily generate the report later if needed. A fallback (placeholder)
        # report is stored but not reported as ready, so the report page
        # regenerates it.
        try:
            async with db.begin_nested():
                report = await build_and_store_report(db, interview, list(final_state.get("qa_history", []) or []))
            report_ready = is_usable_report(report)
        except Exception:  # noqa: BLE001
            logger.exception("Report generation failed for interview %s; it will be generated lazily.", interview.id)

    await db.commit()  # Persist before the response is sent (see "Persistence" above).

    return SubmitAnswerResponse(
        evaluation=evaluation,
        next_question=next_question,
        next_section=next_section,
        next_acknowledgment=next_acknowledgment,
        interview_complete=interview_complete,
        report_ready=report_ready,
    )


@router.post("/{interview_id}/complete")
async def complete_interview(
    interview_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Generate the final feedback report and mark the interview completed.
    Idempotent: when a real report already exists it is returned unchanged.
    A stored fallback (placeholder) report counts as "no report yet" and is
    regenerated. Like /end, an interview without any answers is rejected
    with 400 instead of producing a placeholder report.
    """
    interview = await get_owned_interview_or_404(interview_id, db, user)

    async with interview_locks.hold(interview.id):
        # Refresh under the lock: a request that held it earlier may have
        # committed a new status (and report) while this one was waiting.
        await db.refresh(interview)
        existing = await get_report_row(db, interview.id)
        if is_usable_report(existing):
            if interview.status != "completed":
                interview.status = "completed"
                await db.flush()
                await db.commit()
            return {
                "message": "Interview completed",
                "interview_id": interview.id,
                "overall_score": existing.overall_score,
                "already_completed": True,
            }

        qa_history = await collect_qa_history(db, interview)
        if not qa_history:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No questions answered yet, cannot generate report",
            )

        await _mark_graph_complete(interview.id)
        report = await build_and_store_report(db, interview, qa_history)
        interview.status = "completed"
        await db.flush()
        await db.commit()  # Persist before the response is sent (see "Persistence" above).

    return {
        "message": "Interview completed",
        "interview_id": interview.id,
        "overall_score": report.overall_score,
        "already_completed": False,
    }


@router.post("/{interview_id}/end")
async def end_interview_early(
    interview_id: int,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    End an in-progress interview early and generate a report based on
    whatever questions were answered so far. Idempotent for interviews that
    are already completed and have a real (non-fallback) report.
    """
    interview = await get_owned_interview_or_404(interview_id, db, user)

    async with interview_locks.hold(interview.id):
        # Refresh under the lock so the status check below sees state committed
        # by a request that held the lock earlier (for example a final /answer).
        await db.refresh(interview)
        existing = await get_report_row(db, interview.id)
        if interview.status == "completed" and is_usable_report(existing):
            qa_history = await collect_qa_history(db, interview)
            return {
                "message": "Interview already completed",
                "interview_id": interview.id,
                "questions_answered": len(qa_history),
                "overall_score": existing.overall_score,
            }

        qa_history = await collect_qa_history(db, interview)
        if not qa_history:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No questions answered yet, cannot generate report",
            )

        await _mark_graph_complete(interview.id)
        report = await build_and_store_report(db, interview, qa_history)
        interview.status = "completed"
        await db.flush()
        await db.commit()  # Persist before the response is sent (see "Persistence" above).

    return {
        "message": "Interview ended early",
        "interview_id": interview.id,
        "questions_answered": len(qa_history),
        "overall_score": report.overall_score,
    }


@router.get("")
async def list_interviews(
    user_id: int | None = Query(default=None, description="Must match the authenticated user when given"),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List the caller's interviews, most recent first, with their report score."""
    owner_id = resolve_owner_id(user_id, user)

    result = await db.execute(
        select(Interview, Report.overall_score)
        .outerjoin(Report, Report.interview_id == Interview.id)
        .where(Interview.user_id == owner_id)
        .order_by(Interview.created_at.desc(), Interview.id.desc())
        .limit(limit)
        .offset(offset)
    )

    out = []
    for interview, overall_score in result.all():
        out.append({
            "id": interview.id,
            "company": interview.company,
            "role": interview.role,
            "difficulty": interview.difficulty,
            "date": isoformat(interview.created_at),
            "score": overall_score,
            "status": interview.status,
            "skills": _blueprint_skills(interview.blueprint),
        })
    return out
