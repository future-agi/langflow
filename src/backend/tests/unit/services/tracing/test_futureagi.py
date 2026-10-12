from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langflow.services.tracing import futureagi
from langflow.services.tracing.futureagi import FutureAGITracer, collector_endpoint, span_kind
from langflow.services.tracing.service import ComponentTraceContext, component_context_var
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

FLOW = "Simple Agent - 0f6c2a6e-1d2b-4c1e-9a55-7f4bd3a2c001"


@pytest.fixture
def exported(monkeypatch):
    """Route the tracer's provider to an in-memory exporter and record the provider arguments."""
    exporter = InMemorySpanExporter()
    calls = []

    def fake_provider(endpoint, api_key, secret_key, project):
        calls.append((endpoint, api_key, secret_key, project))
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider

    monkeypatch.setattr(futureagi, "_get_provider", fake_provider)
    monkeypatch.setenv("FI_API_KEY", "api-key")
    monkeypatch.setenv("FI_SECRET_KEY", "secret-key")
    monkeypatch.setenv("FI_PROJECT_NAME", "langflow-project")
    monkeypatch.setenv("FI_BASE_URL", "https://collector.example.com/base/")
    monkeypatch.delenv("FI_REDACT", raising=False)
    exporter.calls = calls
    return exporter


def new_tracer():
    return FutureAGITracer(
        trace_name=FLOW,
        trace_type="chain",
        project_name="Langflow",
        trace_id=uuid4(),
        user_id="user-1",
        session_id="session-1",
    )


def spans_by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


@pytest.mark.parametrize("missing", ["FI_API_KEY", "FI_SECRET_KEY", "FI_PROJECT_NAME"])
def test_inactive_without_keys_or_project(exported, monkeypatch, missing):
    monkeypatch.delenv(missing)
    tracer = new_tracer()
    assert tracer.ready is False
    tracer.end({}, {})
    assert exported.get_finished_spans() == ()
    assert exported.calls == []


def test_collector_endpoint():
    assert collector_endpoint("https://api.futureagi.com") == "https://api.futureagi.com/tracer/v1/traces"
    assert (
        collector_endpoint("https://collector.example.com/base/")
        == "https://collector.example.com/base/tracer/v1/traces"
    )
    assert (
        collector_endpoint("https://api.futureagi.com/tracer/v1/traces") == "https://api.futureagi.com/tracer/v1/traces"
    )


def test_span_kinds():
    assert span_kind("agent") == "AGENT"
    assert span_kind("tool") == "TOOL"
    assert span_kind("llm") == "CHAIN"
    assert span_kind("prompt") == "CHAIN"


def test_run_is_one_trace_with_component_and_llm_spans(exported):
    tracer = new_tracer()
    assert exported.calls == [
        ("https://collector.example.com/base/tracer/v1/traces", "api-key", "secret-key", "langflow-project")
    ]
    tracer.add_trace("ChatInput-abc", "Chat Input", "chain", {"input_value": "What is 2 + 2?"})
    tracer.end_trace("ChatInput-abc", "Chat Input", {"message": "What is 2 + 2?"})

    # The component asks for its callback before the tracing worker has run add_trace for it.
    token = component_context_var.set(ComponentTraceContext("Agent-xyz", "Agent", "agent", None, {}))
    try:
        callback = tracer.get_langchain_callback()
        FakeListChatModel(responses=["4"]).invoke("What is 2 + 2?", config={"callbacks": [callback]})
    finally:
        component_context_var.reset(token)
    tracer.add_trace("Agent-xyz", "Agent", "agent", {"input_value": "What is 2 + 2?"})
    tracer.end_trace("Agent-xyz", "Agent", {"response": "4"})

    tracer.add_trace("ChatOutput-def", "Chat Output", "chain", {"input_value": "4"})
    tracer.end_trace("ChatOutput-def", "Chat Output", {"message": "4"})
    tracer.end({}, {})

    spans = spans_by_name(exported)
    assert set(spans) == {"Simple Agent", "Chat Input", "Agent", "FakeListChatModel", "Chat Output"}
    root = spans["Simple Agent"]
    assert root.parent is None
    assert len({span.context.trace_id for span in spans.values()}) == 1
    assert root.attributes["session.id"] == "session-1"
    assert root.attributes["user.id"] == "user-1"
    assert root.attributes["langflow.flow_id"] == "0f6c2a6e-1d2b-4c1e-9a55-7f4bd3a2c001"
    assert root.attributes["input.value"] == "What is 2 + 2?"
    assert root.attributes["output.value"] == "4"
    assert root.status.status_code == StatusCode.OK
    agent = spans["Agent"]
    assert agent.parent.span_id == root.context.span_id
    assert agent.attributes["openinference.span.kind"] == "AGENT"
    llm = spans["FakeListChatModel"]
    assert llm.parent.span_id == agent.context.span_id
    assert llm.attributes["openinference.span.kind"] == "LLM"
    # The root covers the whole run.
    assert root.start_time <= agent.start_time
    assert root.end_time >= spans["Chat Output"].end_time


def test_component_error_marks_component_and_run(exported):
    tracer = new_tracer()
    tracer.add_trace("Agent-xyz", "Agent", "agent", {"input_value": "hi"})
    tracer.end_trace("Agent-xyz", "Agent", None, error=ValueError("model not found"))
    tracer.end({}, {"Chat Input": {"message": "hi"}})

    spans = spans_by_name(exported)
    for name in ("Agent", "Simple Agent"):
        assert spans[name].status.status_code == StatusCode.ERROR
        assert spans[name].status.description == "ValueError: model not found"
    assert spans["Agent"].events[0].name == "exception"
    assert "output.value" not in spans["Simple Agent"].attributes


def test_unfinished_components_end_with_the_run(exported):
    tracer = new_tracer()
    tracer.add_trace("Agent-xyz", "Agent", "agent", {"input_value": "hi"})
    tracer.end({}, {}, error=RuntimeError("cancelled"))

    spans = spans_by_name(exported)
    assert spans["Agent"].status.status_code == StatusCode.ERROR
    assert spans["Simple Agent"].status.status_code == StatusCode.ERROR


def test_redaction_drops_inputs_and_outputs(exported, monkeypatch):
    monkeypatch.setenv("FI_REDACT", "true")
    tracer = new_tracer()
    tracer.add_trace("ChatInput-abc", "Chat Input", "chain", {"input_value": "secret question"})
    tracer.end_trace("ChatInput-abc", "Chat Input", {"message": "secret question"})
    token = component_context_var.set(ComponentTraceContext("Agent-xyz", "Agent", "agent", None, {}))
    try:
        callback = tracer.get_langchain_callback()
        FakeListChatModel(responses=["secret answer"]).invoke("secret question", config={"callbacks": [callback]})
    finally:
        component_context_var.reset(token)
    tracer.end_trace("Agent-xyz", "Agent", {"response": "secret answer"})
    tracer.end({}, {})

    for span in exported.get_finished_spans():
        # The fake model reports its scripted responses as invocation parameters; real models report settings.
        values = [str(value) for key, value in span.attributes.items() if key != "llm.invocation_parameters"]
        assert not any("secret" in value for value in values), span.name


@pytest.mark.usefixtures("exported")
def test_no_callback_outside_a_component():
    assert new_tracer().get_langchain_callback() is None


def test_provider_is_shared_per_credentials_and_project(monkeypatch):
    monkeypatch.setattr(futureagi, "_providers", {})
    first = futureagi._get_provider("https://c.example.com/tracer/v1/traces", "k", "s", "p")
    assert futureagi._get_provider("https://c.example.com/tracer/v1/traces", "k", "s", "p") is first
    assert futureagi._get_provider("https://c.example.com/tracer/v1/traces", "k", "s", "other") is not first
    assert first.resource.attributes["project_name"] == "p"
    assert first.resource.attributes["project_type"] == "observe"
