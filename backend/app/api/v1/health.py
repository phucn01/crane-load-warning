"""Process and pipeline readiness endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    service = request.app.state.image_processing_service
    return {"status": "ok", **service.readiness()}


__all__ = ["router"]
