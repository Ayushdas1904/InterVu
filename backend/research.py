"""
Research profile cache
----------------------
One ResearchProfile row per (company, role) so repeat interviews skip the
research pipeline. Empty or failed extractions are never cached, otherwise an
LLM or scraper outage would pin an empty profile forever.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres import ResearchProfile

logger = logging.getLogger(__name__)

# Columns of ResearchProfile that come from the extractor output.
PROFILE_FIELDS = (
    "skills",
    "technologies",
    "topics",
    "rounds",
    "behavioral_patterns",
    "key_insights",
    "dsa_questions",
    "difficulty",
)


def is_cacheable(extracted: dict[str, Any]) -> bool:
    """
    False when the extraction failed outright or produced no skills, topics
    and rounds at all (nothing worth caching).
    """
    if extracted.get("extraction_failed"):
        return False
    return any(extracted.get(key) for key in ("skills", "topics", "rounds"))


async def get_research_profile(
        db: AsyncSession,
        company: str,
        role: str
):
    result = await db.execute(
        select(ResearchProfile).where(
            ResearchProfile.company == company,
            ResearchProfile.role == role
        )
    )
    return result.scalar_one_or_none()


async def save_research_profile(
        db: AsyncSession,
        company: str,
        role: str,
        extracted: dict
):
    """
    Cache ``extracted`` for (company, role). Returns the profile, or None when
    another request cached the same pair first (unique constraint).
    """
    columns = {key: extracted[key] for key in PROFILE_FIELDS if key in extracted}
    profile = ResearchProfile(company=company, role=role, **columns)
    db.add(profile)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        logger.info("Research profile for %s / %s was cached concurrently; keeping the existing row.", company, role)
        return None
    except Exception:
        await db.rollback()
        raise
    await db.refresh(profile)
    return profile
