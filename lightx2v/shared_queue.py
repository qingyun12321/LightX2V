"""Shared generation queue for LightX2V examples and APIs."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

ASYNC_GENERATION_SEMAPHORE = asyncio.Semaphore(1)
GENERATION_LOCK = threading.Lock()

_STATUS_LOCK = threading.Lock()
_REQUEST_STATUS: dict[str, str] = {}
_QUEUE_LOCK = threading.Lock()
_PENDING_QUEUE: list[str] = []
_CURRENT_REQUEST: str | None = None


def get_queue_status(request_id: str | None = None) -> dict[str, int | bool | str]:
    with _QUEUE_LOCK:
        pending = len(_PENDING_QUEUE)
        current = _CURRENT_REQUEST
        position = None
        if request_id:
            if request_id == current:
                position = 0
            elif request_id in _PENDING_QUEUE:
                position = _PENDING_QUEUE.index(request_id) + 1
    status = None
    if request_id:
        with _STATUS_LOCK:
            status = _REQUEST_STATUS.get(request_id, "unknown")
    payload: dict[str, int | bool | str] = {
        "processing": current is not None,
        "pending": pending,
    }
    if request_id:
        payload["status"] = status
        payload["position"] = position if position is not None else -1
    return payload


def _set_request_status(request_id: str | None, status: str) -> None:
    if not request_id:
        return
    with _STATUS_LOCK:
        _REQUEST_STATUS[request_id] = status


def _clear_request_status(request_id: str | None) -> None:
    if not request_id:
        return
    with _STATUS_LOCK:
        _REQUEST_STATUS.pop(request_id, None)


@asynccontextmanager
async def generation_slot(request_id: str | None = None):
    global _CURRENT_REQUEST
    _set_request_status(request_id, "pending")
    if request_id:
        with _QUEUE_LOCK:
            if request_id not in _PENDING_QUEUE and request_id != _CURRENT_REQUEST:
                _PENDING_QUEUE.append(request_id)
    try:
        await ASYNC_GENERATION_SEMAPHORE.acquire()
    except Exception:
        if request_id:
            with _QUEUE_LOCK:
                if request_id in _PENDING_QUEUE:
                    _PENDING_QUEUE.remove(request_id)
        _clear_request_status(request_id)
        raise
    try:
        if request_id:
            with _QUEUE_LOCK:
                if request_id in _PENDING_QUEUE:
                    _PENDING_QUEUE.remove(request_id)
                _CURRENT_REQUEST = request_id
        _set_request_status(request_id, "processing")
        yield
    finally:
        if request_id:
            with _QUEUE_LOCK:
                if _CURRENT_REQUEST == request_id:
                    _CURRENT_REQUEST = None
        _clear_request_status(request_id)
        ASYNC_GENERATION_SEMAPHORE.release()
