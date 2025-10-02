"""Pydantic data models for the collaborative server layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["admin", "user"]


class User(BaseModel):
    """Represents a stored user including hashed password."""

    username: str
    password_hash: str = Field(repr=False)
    role: Role = "user"


class UserPublic(BaseModel):
    """Public information about a user returned by the API."""

    username: str
    role: Role


class UserLogin(BaseModel):
    """Payload used when a user attempts to login."""

    username: str
    password: str


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class RunInfo(BaseModel):
    """Metadata about a saved run."""

    name: str
    created_at: datetime
    description: Optional[str] = None


class JobInfo(BaseModel):
    """Represents a lightweight job entry."""

    id: str
    status: Literal["queued", "running", "finished", "failed"]
    created_at: datetime
    updated_at: datetime
    message: Optional[str] = None


class CompareRequest(BaseModel):
    """Request body for a run comparison."""

    base_run: str
    candidate_run: str
    compare_type: Literal["summary", "full"] = "summary"


class CompareResponse(BaseModel):
    """Simplified response for comparison runs."""

    job_id: str
    status: str


class SimulationBundle(BaseModel):
    """Information describing a simulation bundle registration."""

    name: str
    description: Optional[str] = None
    dataset: Optional[str] = None
    metadata: dict[str, str] | None = None


class SimulationResponse(BaseModel):
    """Response confirming that the simulation bundle was registered."""

    bundle_name: str
    artifact_path: str
