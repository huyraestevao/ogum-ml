"""FastAPI application exposing the collaborative backend."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from . import auth, storage
from .models import (
    CompareRequest,
    CompareResponse,
    JobInfo,
    RunInfo,
    SimulationBundle,
    SimulationResponse,
    Token,
    UserLogin,
    UserPublic,
)

app = FastAPI(title="Ogum-ML Collaborative API", version="0.15.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:  # pragma: no cover - fastapi hook
    storage.ensure_demo_run()


@app.post("/auth/login", response_model=Token)
async def login(payload: UserLogin) -> Token:
    user = auth.authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    return auth.create_jwt(user)


@app.get("/auth/me", response_model=UserPublic)
async def read_current_user(
    current_user: UserPublic = Depends(auth.get_current_user),
) -> UserPublic:
    return current_user


@app.get("/runs/list", response_model=list[RunInfo])
async def list_runs(
    current_user: UserPublic = Depends(auth.get_current_user),
) -> list[RunInfo]:
    return storage.list_runs()


@app.get("/jobs/list", response_model=list[JobInfo])
async def list_jobs(
    current_user: UserPublic = Depends(auth.get_current_user),
) -> list[JobInfo]:
    return storage.list_jobs()


@app.post("/runs/upload", response_model=RunInfo)
async def upload_run(
    file: UploadFile = File(...),
    description: str | None = Form(None),
    current_user: UserPublic = Depends(auth.get_current_user),
) -> RunInfo:
    return storage.save_run_artifact(file.filename, file.file, description)


@app.post("/compare/run", response_model=CompareResponse)
async def trigger_compare(
    request: CompareRequest,
    current_user: UserPublic = Depends(auth.get_current_user),
) -> CompareResponse:
    job_id = f"compare-{uuid.uuid4().hex}"
    job = JobInfo(
        id=job_id,
        status="queued",
        message=f"Comparison queued for {request.base_run} vs {request.candidate_run}",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    storage.upsert_job(job)
    return CompareResponse(job_id=job_id, status="queued")


@app.post("/sim/import", response_model=SimulationResponse)
async def register_simulation(
    bundle: SimulationBundle,
    current_user: UserPublic = Depends(auth.get_current_user),
) -> SimulationResponse:
    path = storage.register_simulation(bundle)
    return SimulationResponse(bundle_name=bundle.name, artifact_path=str(path))


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
