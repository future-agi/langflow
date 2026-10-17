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


def test_partial_config_warns_once_naming_the_missing_variables(monkeypatch):
    warnings = []
    monkeypatch.setattr(futureagi.logger, "warning", lambda msg, *a, **k: warnings.append(msg))  # noqa: ARG005
    monkeypatch.setattr(futureagi, "_warned_missing", False)
    for name in ("FI_API_KEY", "FI_SECRET_KEY", "FI_PROJECT_NAME"):
        monkeypatch.delenv(name, raising=False)
    assert new_tracer().ready is False
    assert warnings == []  # Not configured at all: stay quiet.
    monkeypatch.setenv("FI_API_KEY", "api-key")
    monkeypatch.setenv("FI_SECRET_KEY", "secret-key")
    assert new_tracer().ready is False
    assert new_tracer().ready is False
    assert len(warnings) == 1
    assert "FI_PROJECT_NAME" in warnings[0]
    assert "api-key" not in warnings[0]


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


@pytest.fixture
def scrubbed(monkeypatch):
    """Like ``exported``, but spans pass through the tracer's own scrubbing exporter."""
    exporter = InMemorySpanExporter()

    def fake_provider(endpoint, api_key, secret_key, project):  # noqa: ARG001
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(futureagi._ScrubbingExporter(exporter)))
        return provider

    monkeypatch.setattr(futureagi, "_get_provider", fake_provider)
    monkeypatch.setenv("FI_API_KEY", "api-key")
    monkeypatch.setenv("FI_SECRET_KEY", "secret-key")
    monkeypatch.setenv("FI_PROJECT_NAME", "langflow-project")
    for name in ("FI_REDACT", "FI_HIDE_INPUTS", "FI_HIDE_OUTPUTS"):
        monkeypatch.delenv(name, raising=False)
    return exporter


class BrokenChatModel(FakeListChatModel):
    """A chat model whose call fails with a local path and an e-mail address in the message."""

    def _call(self, *args, **kwargs):  # noqa: ARG002
        msg = f"cannot read {futureagi._HOME}/models/config.json for ops@example.com"
        raise ValueError(msg)


def _all_strings(span):
    values = [span.status.description or ""]
    values += [str(value) for value in span.attributes.values()]
    for event in span.events:
        values += [str(value) for value in (event.attributes or {}).values()]
    return values


def test_scrub_error_text(monkeypatch):
    monkeypatch.setattr(futureagi, "_LOCAL_NAMES", ["build-host-7", "jdoe"])
    text = (
        "ValueError('bad input from jdoe@example.com on build-host-7')"
        "Traceback (most recent call last):\n"
        '  File "/Users/jdoe/venv/lib/python3.12/site-packages/x.py", line 1, in f\n'
    )
    assert futureagi.scrub_error_text(text) == "ValueError('bad input from <email> on <redacted>')"
    assert futureagi.scrub_error_text("open /home/jdoe/data.csv failed for jdoe") == (
        "open ~/data.csv failed for <redacted>"
    )
    assert futureagi.scrub_error_text(f"missing {futureagi._HOME}/flows/a.json") == "missing ~/flows/a.json"


def test_failed_llm_call_exports_no_stack_trace_or_local_path(scrubbed):
    tracer = new_tracer()
    token = component_context_var.set(ComponentTraceContext("Agent-xyz", "Agent", "agent", None, {}))
    try:
        callback = tracer.get_langchain_callback()
        with pytest.raises(ValueError, match="cannot read"):
            BrokenChatModel(responses=["unused"]).invoke("hi", config={"callbacks": [callback]})
    finally:
        component_context_var.reset(token)
    tracer.add_trace("Agent-xyz", "Agent", "agent", {"input_value": "hi"})
    tracer.end_trace("Agent-xyz", "Agent", None, error=ValueError(f"model failed in {futureagi._HOME}/x"))
    tracer.end({}, {})

    spans = spans_by_name(scrubbed)
    llm = spans["BrokenChatModel"]
    assert llm.status.status_code == StatusCode.ERROR
    assert "cannot read ~/models/config.json for <email>" in llm.status.description
    for span in spans.values():
        for value in _all_strings(span):
            assert "Traceback" not in value, span.name
            assert 'File "' not in value, span.name
            assert futureagi._HOME not in value, span.name
            assert "ops@example.com" not in value, span.name
        for event in span.events:
            assert "exception.stacktrace" not in (event.attributes or {}), span.name
    assert spans["Agent"].status.description == "ValueError: model failed in ~/x"
    assert spans["Simple Agent"].status.status_code == StatusCode.ERROR


def test_home_directory_is_masked_in_content(scrubbed):
    tracer = new_tracer()
    tracer.add_trace("File-abc", "File", "chain", {"path": f"{futureagi._HOME}/.langflow/report.pdf"})
    tracer.end_trace("File-abc", "File", {"text": "ok"})
    tracer.end({}, {})
    span = spans_by_name(scrubbed)["File"]
    assert "~/.langflow/report.pdf" in span.attributes["input.value"]
    assert futureagi._HOME not in span.attributes["input.value"]


@pytest.mark.parametrize(
    ("env", "inputs_hidden", "outputs_hidden"),
    [
        ({"FI_HIDE_INPUTS": "true"}, True, False),
        ({"FI_HIDE_OUTPUTS": "true"}, False, True),
        ({"FI_HIDE_INPUTS": "1", "FI_HIDE_OUTPUTS": "1"}, True, True),
        ({"FI_REDACT": "true"}, True, True),
        ({}, False, False),
    ],
)
def test_hide_switches(scrubbed, monkeypatch, env, inputs_hidden, outputs_hidden):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    tracer = new_tracer()
    tracer.add_trace("ChatInput-abc", "Chat Input", "chain", {"input_value": "question-marker"})
    tracer.end_trace("ChatInput-abc", "Chat Input", {"message": "question-marker"})
    token = component_context_var.set(ComponentTraceContext("Agent-xyz", "Agent", "agent", None, {}))
    try:
        callback = tracer.get_langchain_callback()
        FakeListChatModel(responses=["answer-marker"]).invoke("question-marker", config={"callbacks": [callback]})
    finally:
        component_context_var.reset(token)
    tracer.end_trace("Agent-xyz", "Agent", {"response": "answer-marker"})
    tracer.add_trace("ChatOutput-def", "Chat Output", "chain", {"input_value": "answer-marker"})
    tracer.end_trace("ChatOutput-def", "Chat Output", {"message": "answer-marker"})
    tracer.end({}, {})

    spans = scrubbed.get_finished_spans()
    root = spans_by_name(scrubbed)["Simple Agent"]
    assert ("input.value" in root.attributes) is not inputs_hidden
    assert ("output.value" in root.attributes) is not outputs_hidden
    in_attrs = [
        str(v)
        for s in spans
        for k, v in s.attributes.items()
        if k.startswith(("input.", "llm.input_messages", "llm.prompts"))
    ]
    out_attrs = [
        str(v) for s in spans for k, v in s.attributes.items() if k.startswith(("output.", "llm.output_messages"))
    ]
    assert any("question-marker" in v for v in in_attrs) is not inputs_hidden
    # Chat Input's output repeats the question, so only the answer marker is checked on the output side.
    assert any("answer-marker" in v for v in out_attrs) is not outputs_hidden
    # Hierarchy, model kind and status survive hiding.
    assert {s.name for s in spans} >= {"Simple Agent", "Chat Input", "Agent", "FakeListChatModel", "Chat Output"}


def test_flow_id_tracing_user_and_version(scrubbed):
    tracer = FutureAGITracer(
        trace_name="My Flow - not-the-id",
        trace_type="chain",
        project_name="Langflow",
        trace_id=uuid4(),
        user_id="user-1",
        session_id="session-1",
        flow_id="flow-123",
        tracing_user_id="end-user-9",
    )
    tracer.add_trace("ChatInput-abc", "Chat Input", "chain", {"input_value": "hi"})
    tracer.end_trace("ChatInput-abc", "Chat Input", {"message": "hi"})
    tracer.end({}, {})
    spans = spans_by_name(scrubbed)
    root = spans["My Flow"]
    assert root.attributes["langflow.flow_id"] == "flow-123"
    assert root.attributes["langflow.tracing_user_id"] == "end-user-9"
    assert root.attributes["user.id"] == "user-1"
    assert root.attributes["langflow.version"]
    assert spans["Chat Input"].attributes["session.id"] == "session-1"


def test_export_failures_are_logged_at_most_once_a_minute(monkeypatch):
    from opentelemetry.sdk.trace.export import SpanExportResult

    class FailingExporter:
        def export(self, spans):  # noqa: ARG002
            return SpanExportResult.FAILURE

    warnings = []
    monkeypatch.setattr(futureagi.logger, "warning", lambda msg, *a, **k: warnings.append(msg))  # noqa: ARG005
    exporter = futureagi._ScrubbingExporter(FailingExporter())
    provider = TracerProvider()
    span = provider.get_tracer("t").start_span("s")
    span.end()
    assert exporter.export([span]) == SpanExportResult.FAILURE
    assert exporter.export([span, span]) == SpanExportResult.FAILURE
    assert len(warnings) == 1
    assert "dropped 1 span(s)" in warnings[0]
    assert "Flow runs are not affected" in warnings[0]


def test_exporter_is_uncompressed_even_with_gzip_env(monkeypatch):
    monkeypatch.setattr(futureagi, "_providers", {})
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_COMPRESSION", "gzip")
    provider = futureagi._get_provider("https://c.example.com/tracer/v1/traces", "k", "s", "p")
    processor = provider._active_span_processor._span_processors[0]
    wrapper = processor._batch_processor._exporter
    assert isinstance(wrapper, futureagi._ScrubbingExporter)
    # The SDK stores its own enum type across versions; both spell "no compression" as "none".
    assert wrapper._exporter._compression.value == "none"
    provider.shutdown()
