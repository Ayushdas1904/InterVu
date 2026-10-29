from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from utils.difficulty import parse_difficulty

_AUTO_DIFFICULTY = {"", "auto", "default", "research"}


class StartInterviewRequest(BaseModel):
    # Optional and only used as a consistency check: the interview is always
    # owned by the authenticated user from the bearer token.
    user_id: int | None = None
    username: str = Field(default="", max_length=100)
    company: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=255)
    # "Easy" | "Medium" | "Hard" (any case). None / "auto" uses the
    # difficulty inferred by the research pipeline.
    difficulty: str | None = None

    @field_validator("username", "company", "role", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("difficulty must be a string")
        if value.strip().lower() in _AUTO_DIFFICULTY:
            return None
        parsed = parse_difficulty(value)
        if parsed is None:
            raise ValueError("difficulty must be one of: Easy, Medium, Hard")
        return parsed


class StartInterviewResponse(BaseModel):
    interview_id: int
    first_question: str
    blueprint: dict[str, Any]
    difficulty: str | None = None


class SubmitAnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=20000)

    @field_validator("answer", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class SubmitAnswerResponse(BaseModel):
    evaluation: dict[str, Any]
    next_question: str | None
    next_section: str | None
    interview_complete: bool
    next_acknowledgment: str | None
    report_ready: bool = False
