"""Text-only Chat Completions contract; no retrieval or memory implementation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..common.config import Settings


class RequestError(ValueError):
    def __init__(self, message: str, *, param: str = "request_body",
                 code: str = "validation_error", status: int = 400) -> None:
        super().__init__(message)
        self.param, self.code, self.status = param, code, status


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart]
    name: str | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    user: str | None = Field(default=None, max_length=200)
    compact_k: int = Field(default=5, ge=1)
    # Minutes; the stored expiration is an absolute UTC deadline.
    memory_expiration: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    # Accepted for client compatibility. Server-side generation settings remain
    # authoritative and are not changed by public requests.
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    n: int = Field(default=1, ge=1, le=1)

    def text_messages(self) -> list[dict[str, str]]:
        """Normalize all supplied turns without changing their order or roles."""
        return [
            {"role": message.role, "content": message.text(),
             **({"name": message.name} if message.name is not None else {})}
            for message in self.messages
        ]


def _session(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 200 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise RequestError("Session ID must be 1-200 characters without control characters.", param="user")
    return value


def parse_request(payload: Any, settings: Settings) -> tuple[ChatCompletionRequest, str, str]:
    if not isinstance(payload, dict):
        raise RequestError("The JSON body must be an object.")
    try:
        req = ChatCompletionRequest.model_validate({
            "compact_k": settings.compact_k,
            "memory_expiration": settings.memory_expiration,
            **payload,
        })
    except ValidationError as exc:
        error = exc.errors(include_input=False, include_url=False)[0]
        param = ".".join(str(part) for part in error["loc"]) or "request_body"
        raise RequestError(f"{param}: {error['msg']}", param=param) from None
    try:
        datetime.now(timezone.utc) + timedelta(minutes=req.memory_expiration)
    except OverflowError:
        raise RequestError("memory_expiration exceeds the supported UTC timestamp range.",
                           param="memory_expiration") from None
    if req.stream:
        raise RequestError("Streaming is not supported; use stream=false.",
                           param="stream", code="streaming_not_supported")
    if req.model is not None and req.model != settings.public_model_name:
        raise RequestError("The requested model is not served by this News Agent.",
                           param="model", code="model_not_found", status=404)
    if req.messages[-1].role != "user" or not req.messages[-1].text().strip():
        raise RequestError("messages must end with a non-empty user message.",
                           param="messages", code="missing_user_message")
    # The final user turn remains the question to audit and capture. All
    # messages are separately forwarded as working context. A stored session's
    # latest history takes precedence over a stale caller-supplied prefix.
    query = req.messages[-1].text()
    session_id = _session(req.user if req.user and req.user.strip() else settings.mem0_session_id)
    return req, query, session_id


def authenticated_user(environ: dict[str, Any]) -> str | None:
    """Only WSGI authentication middleware may establish a memory-user identity."""
    user = environ.get("REMOTE_USER")
    if user is None:
        return None
    if (not isinstance(user, str) or not user.strip() or len(user.strip()) > 200
            or any(ord(c) < 32 or ord(c) == 127 for c in user)):
        raise RequestError("The server authentication context has no valid user identity.",
                           code="invalid_authentication_context", status=401)
    return user.strip()


def completion_response(result: dict[str, Any], *, completion_id: str, model: str,
                        session_id: str, created: int) -> dict[str, Any]:
    """Wrap only the agent's final result; do not rewrite its content or citations."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]},
                     "finish_reason": "stop"}],
        # The core agent does not report aggregated model usage. Unknown is not zero.
        "usage": None,
        # Custom response fields survive OpenAI SDK model_dump(). A replacement
        # already includes this turn; clients must not append the answer again.
        **({"messages": result["messages"]} if "messages" in result else {}),
        "version": result.get("version"),
        "compacted": result.get("compacted", False),
        "metadata": {
            "domain": "technology_news",
            "audit_id": result["audit_id"],
            "request_id": result["request_id"],
            "session_id": session_id,
            "status": result["status"],
            "verification_status": result["verification_status"],
            "evidence_count": result["evidence_count"],
            "memory_status": result.get("memory_status", "not_used"),
            "history_version": result.get("version"),
            "history_status": result.get("history_status", "not_used"),
            "history_snapshot_id": result.get("history_snapshot_id"),
            "compacted": result.get("compacted", False),
        },
    }
