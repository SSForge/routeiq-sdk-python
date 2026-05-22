"""Tests for RouteIQ task/step/tool context managers."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from routeiq import RouteIQ


@pytest.fixture
def riq(monkeypatch):
    """RouteIQ instance wired to an in-memory exporter (no real OTel collector)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = RouteIQ.__new__(RouteIQ)
    import uuid
    client.agent_id = "test-agent"
    client.tenant_id = "test-tenant"
    client.environment = "test"
    client.model = "gpt-4o"
    client.agent_version = "1.0.0"
    client.session_id = str(uuid.uuid4())
    from opentelemetry import trace
    client._provider = provider
    client._tracer = trace.get_tracer("routeiq.sdk", tracer_provider=provider)

    yield client, exporter


def spans(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


# ── TaskHandle ────────────────────────────────────────────────────────────────

def test_task_span_name_starts_with_task(riq):
    client, exp = riq
    with client.task(intent="find Paris"):
        pass
    names = [s.name for s in exp.get_finished_spans()]
    assert any(n.startswith("task:") for n in names)


def test_task_envelope_attrs(riq):
    client, exp = riq
    with client.task(intent="find Paris") as task:
        task_id = task.task_id

    span = next(s for s in exp.get_finished_spans() if s.name.startswith("task:"))
    assert span.attributes["routeiq.agent.id"] == "test-agent"
    assert span.attributes["routeiq.session.id"] == client.session_id
    assert span.attributes["routeiq.task.id"] == task_id
    assert span.attributes["routeiq.task.input_intent"] == "find Paris"
    assert span.attributes["routeiq.version.model.name"] == "gpt-4o"


def test_task_complete_sets_success(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        task.complete(tokens=100, cost_usd=0.001, cohort="test")

    span = next(s for s in exp.get_finished_spans() if s.name.startswith("task:"))
    assert span.attributes["routeiq.task.completion_status"] == "1"
    assert span.attributes["routeiq.task.total_tokens"] == 100
    assert span.attributes["routeiq.task.cost_usd"] == 0.001
    assert span.attributes["routeiq.task.cohort"] == "test"


def test_task_fail_sets_failure(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        task.fail(category="tool_error")

    span = next(s for s in exp.get_finished_spans() if s.name.startswith("task:"))
    assert span.attributes["routeiq.task.completion_status"] == "2"
    assert span.attributes["routeiq.task.failure_category"] == "tool_error"


def test_task_auto_fails_on_exception(riq):
    client, exp = riq
    with pytest.raises(ValueError):
        with client.task(intent="q"):
            raise ValueError("boom")

    span = next(s for s in exp.get_finished_spans() if s.name.startswith("task:"))
    assert span.attributes["routeiq.task.completion_status"] == "2"


def test_task_auto_succeeds_on_clean_exit(riq):
    client, exp = riq
    with client.task(intent="q"):
        pass

    span = next(s for s in exp.get_finished_spans() if s.name.startswith("task:"))
    assert span.attributes["routeiq.task.completion_status"] == "1"


# ── StepHandle ────────────────────────────────────────────────────────────────

def test_step_span_name_starts_with_step(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step(action="tool_call"):
            pass

    names = [s.name for s in exp.get_finished_spans()]
    assert any(n.startswith("step:") for n in names)


def test_step_carries_task_id(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            step_id = step.step_id

    step_span = next(s for s in exp.get_finished_spans() if s.name.startswith("step:"))
    assert step_span.attributes["routeiq.task.id"] == task.task_id
    assert step_span.attributes["routeiq.step.id"] == step_id


def test_step_selected_action(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step(action="tool_call", rationale="need to search"):
            pass

    step_span = next(s for s in exp.get_finished_spans() if s.name.startswith("step:"))
    assert step_span.attributes["routeiq.step.selected_action"] == "tool_call"
    assert step_span.attributes["routeiq.step.action_rationale"] == "need to search"


def test_step_index_increments(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step():
            pass
        with task.step():
            pass

    step_spans = sorted(
        [s for s in exp.get_finished_spans() if s.name.startswith("step:")],
        key=lambda s: s.attributes["routeiq.step.index"],
    )
    assert step_spans[0].attributes["routeiq.step.index"] == 1
    assert step_spans[1].attributes["routeiq.step.index"] == 2


def test_step_auto_fails_on_exception(riq):
    client, exp = riq
    with pytest.raises(RuntimeError):
        with client.task(intent="q"):
            with client.task(intent="q")._riq._tracer.start_as_current_span("dummy"):
                pass
            with client.task(intent="q") as task2:
                with task2.step() as step:
                    raise RuntimeError("step failed")

    step_spans = [s for s in exp.get_finished_spans() if s.name.startswith("step:")]
    assert any(s.attributes.get("routeiq.step.completion_status") == "2" for s in step_spans)


# ── ToolHandle ────────────────────────────────────────────────────────────────

def test_tool_span_name_is_tool_name(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("search", args={"query": "Paris"}):
                pass

    names = [s.name for s in exp.get_finished_spans()]
    assert "tool:search" in names


def test_tool_success(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("search") as tool:
                tool.success(latency_ms=50.0)

    tool_span = next(s for s in exp.get_finished_spans() if s.name == "tool:search")
    assert tool_span.attributes["routeiq.tool.result_status"] == "1"
    assert tool_span.attributes["routeiq.tool.latency_ms"] == 50.0


def test_tool_fail(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("search") as tool:
                tool.fail(error_code="TIMEOUT")

    tool_span = next(s for s in exp.get_finished_spans() if s.name == "tool:search")
    assert tool_span.attributes["routeiq.tool.result_status"] == "2"
    assert tool_span.attributes["routeiq.tool.error_code"] == "TIMEOUT"


def test_tool_auto_fails_on_exception(riq):
    client, exp = riq
    with pytest.raises(ConnectionError):
        with client.task(intent="q") as task:
            with task.step() as step:
                with step.tool("search"):
                    raise ConnectionError("network down")

    tool_span = next(s for s in exp.get_finished_spans() if s.name == "tool:search")
    assert tool_span.attributes["routeiq.tool.result_status"] == "2"


def test_tool_auto_succeeds_on_clean_exit(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("search"):
                pass  # no explicit success/fail call

    tool_span = next(s for s in exp.get_finished_spans() if s.name == "tool:search")
    assert tool_span.attributes["routeiq.tool.result_status"] == "1"


def test_tool_arguments_hash(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("search", args={"query": "Paris"}):
                pass

    tool_span = next(s for s in exp.get_finished_spans() if s.name == "tool:search")
    assert "routeiq.tool.arguments_hash" in tool_span.attributes
    assert len(tool_span.attributes["routeiq.tool.arguments_hash"]) == 16


def test_tool_permission_level(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("write_file", permission="READ_WRITE"):
                pass

    tool_span = next(s for s in exp.get_finished_spans() if s.name == "tool:write_file")
    assert tool_span.attributes["routeiq.tool.permission_level"] == "2"


def test_session_id_same_across_spans(riq):
    client, exp = riq
    with client.task(intent="q") as task:
        with task.step() as step:
            with step.tool("search"):
                pass

    all_spans = exp.get_finished_spans()
    session_ids = {s.attributes.get("routeiq.session.id") for s in all_spans}
    assert len(session_ids) == 1
    assert client.session_id in session_ids
