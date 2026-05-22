"""
TaskHandle, StepHandle, ToolHandle — sync context managers for the RouteIQ SDK.

Attribute keys match conventions/telemetry.yaml in routeiq-schema.
Enum values match entities.proto (SUCCESS=1, FAILURE=2, TOOL_SUCCESS=1, TOOL_FAILURE=2).
"""

import hashlib
import json
import time
import uuid
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .client import RouteIQ

# ── Enum values (mirror entities.proto) ──────────────────────────────────────

_COMPLETION_SUCCESS = "1"
_COMPLETION_FAILURE = "2"
_TOOL_SUCCESS = "1"
_TOOL_FAILURE = "2"

PERMISSION = {
    "READ_ONLY":  "1",
    "READ_WRITE": "2",
    "PRIVILEGED": "3",
}


# ── ToolHandle ────────────────────────────────────────────────────────────────

class ToolHandle:
    """Context manager for a single tool invocation inside a step."""

    def __init__(
        self,
        step: "StepHandle",
        name: str,
        args: Optional[dict] = None,
        permission: str = "READ_ONLY",
    ):
        self._step = step
        self.name = name
        self._args = args or {}
        self._permission = PERMISSION.get(permission, "1")
        self._start: float = 0.0
        self._span_cm = None
        self._span = None
        self._done = False

    def __enter__(self) -> "ToolHandle":
        riq = self._step._task._riq
        self._start = time.monotonic()
        args_hash = hashlib.sha256(
            json.dumps(self._args, sort_keys=True).encode()
        ).hexdigest()[:16]
        self._span_cm = riq._tracer.start_as_current_span(f"tool:{self.name}")
        self._span = self._span_cm.__enter__()
        self._span.set_attributes({
            "routeiq.event.type": "7",  # TOOL_CALLED
            **riq._envelope(self._step._task, self._step),
            "routeiq.tool.name": self.name,
            "routeiq.tool.arguments_hash": args_hash,
            "routeiq.tool.permission_level": self._permission,
        })
        return self

    def success(self, latency_ms: Optional[float] = None) -> None:
        """Record a successful tool result."""
        self._finish(_TOOL_SUCCESS, latency_ms=latency_ms)

    def fail(self, error_code: str = "", latency_ms: Optional[float] = None) -> None:
        """Record a failed tool result."""
        self._finish(_TOOL_FAILURE, error_code=error_code, latency_ms=latency_ms)

    def _finish(self, status: str, error_code: str = "", latency_ms: Optional[float] = None):
        if self._done:
            return
        self._done = True
        elapsed = (time.monotonic() - self._start) * 1000
        attrs: dict = {
            "routeiq.tool.result_status": status,
            "routeiq.tool.latency_ms": latency_ms if latency_ms is not None else elapsed,
        }
        if error_code:
            attrs["routeiq.tool.error_code"] = error_code
        if self._span:
            self._span.set_attributes(attrs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._done:
            elapsed = (time.monotonic() - self._start) * 1000
            if exc_type is not None:
                self.fail(latency_ms=elapsed)
            else:
                self.success(latency_ms=elapsed)
        self._span_cm.__exit__(exc_type, exc_val, exc_tb)
        return False


# ── StepHandle ────────────────────────────────────────────────────────────────

class StepHandle:
    """Context manager for one reasoning step within a task."""

    def __init__(
        self,
        task: "TaskHandle",
        action: Optional[str] = None,
        rationale: Optional[str] = None,
        index: int = 1,
    ):
        self._task = task
        self.step_id = str(uuid.uuid4())
        self._action = action
        self._rationale = rationale
        self._index = index
        self._span_cm = None
        self._span = None
        self._done = False

    def __enter__(self) -> "StepHandle":
        riq = self._task._riq
        self._span_cm = riq._tracer.start_as_current_span(f"step:{self.step_id}")
        self._span = self._span_cm.__enter__()
        attrs = {
            "routeiq.event.type": "4",  # STEP_STARTED
            **riq._envelope(self._task, self),
        }
        if self._action:
            attrs["routeiq.step.selected_action"] = self._action
        if self._rationale:
            attrs["routeiq.step.action_rationale"] = self._rationale
        attrs["routeiq.step.index"] = self._index
        self._span.set_attributes(attrs)
        return self

    def tool(
        self,
        name: str,
        args: Optional[dict] = None,
        permission: str = "READ_ONLY",
    ) -> ToolHandle:
        """Start a tool call within this step."""
        return ToolHandle(self, name=name, args=args, permission=permission)

    def complete(self) -> None:
        """Mark step as successfully completed."""
        self._finish(_COMPLETION_SUCCESS)

    def fail(self, category: str = "") -> None:
        """Mark step as failed."""
        self._finish(_COMPLETION_FAILURE, failure_category=category)

    def _finish(self, status: str, failure_category: str = ""):
        if self._done:
            return
        self._done = True
        attrs: dict = {"routeiq.step.completion_status": status}
        if failure_category:
            attrs["routeiq.step.failure_category"] = failure_category
        if self._span:
            self._span.set_attributes(attrs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._done:
            if exc_type is not None:
                self.fail()
            else:
                self.complete()
        self._span_cm.__exit__(exc_type, exc_val, exc_tb)
        return False


# ── TaskHandle ────────────────────────────────────────────────────────────────

class TaskHandle:
    """Context manager for a complete agent task."""

    def __init__(
        self,
        riq: "RouteIQ",
        intent: str,
        task_type: Optional[str] = None,
    ):
        self._riq = riq
        self.intent = intent
        self.task_type = task_type
        self.task_id = str(uuid.uuid4())
        self.run_id = str(uuid.uuid4())
        self._span_cm = None
        self._span = None
        self._done = False
        self._step_index = 0

    def __enter__(self) -> "TaskHandle":
        self._span_cm = self._riq._tracer.start_as_current_span(f"task:{self.task_id}")
        self._span = self._span_cm.__enter__()
        attrs = {
            "routeiq.event.type": "1",  # TASK_STARTED
            **self._riq._envelope(self),
            "routeiq.task.input_intent": self.intent[:256],
        }
        if self.task_type:
            attrs["routeiq.task.type"] = self.task_type
        self._span.set_attributes(attrs)
        return self

    def step(
        self,
        action: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> StepHandle:
        """Start a reasoning step within this task."""
        self._step_index += 1
        return StepHandle(self, action=action, rationale=rationale, index=self._step_index)

    def complete(
        self,
        tokens: int = 0,
        cost_usd: Optional[float] = None,
        cohort: Optional[str] = None,
    ) -> None:
        """Mark task as successfully completed."""
        self._finish(_COMPLETION_SUCCESS, tokens=tokens, cost_usd=cost_usd, cohort=cohort)

    def fail(self, category: str = "") -> None:
        """Mark task as failed."""
        self._finish(_COMPLETION_FAILURE, failure_category=category)

    def _finish(
        self,
        status: str,
        tokens: int = 0,
        cost_usd: Optional[float] = None,
        cohort: Optional[str] = None,
        failure_category: str = "",
    ):
        if self._done:
            return
        self._done = True
        attrs: dict = {"routeiq.task.completion_status": status}
        if tokens:
            attrs["routeiq.task.total_tokens"] = tokens
        if cost_usd is not None:
            attrs["routeiq.task.cost_usd"] = cost_usd
        if cohort:
            attrs["routeiq.task.cohort"] = cohort
        if failure_category:
            attrs["routeiq.task.failure_category"] = failure_category
        if self._span:
            self._span.set_attributes(attrs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._done:
            if exc_type is not None:
                self.fail()
            else:
                self.complete()
        self._span_cm.__exit__(exc_type, exc_val, exc_tb)
        return False
