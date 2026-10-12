"""Future AGI tracer.

Exports OTLP/HTTP with the headers and resource attributes fi-collector
accepts. The Phoenix and Traceloop tracers authenticate differently and are
not retargeted.

A flow run becomes one trace: a root span for the run, a child span per
component, and the LangChain runs inside a component (LLM calls with model and
token counts, tool calls) nested under that component's span.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from lfx.log.logger import logger
from typing_extensions import override

from langflow.services.tracing.base import BaseTracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain_core.callbacks.base import BaseCallbackHandler
    from lfx.graph.vertex.base import Vertex
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span

    from langflow.services.tracing.schema import Log

_DEFAULT_ENDPOINT = "https://api.futureagi.com"
_COLLECTOR_PATH = "/tracer/v1/traces"
_TRACER_NAME = "langflow.futureagi"

# OpenInference span kinds for Langflow component trace types. Model components map to CHAIN:
# the LLM span, with model and token counts, comes from the LangChain callback beneath them.
_SPAN_KINDS = {"agent": "AGENT", "tool": "TOOL", "retriever": "RETRIEVER", "embedding": "EMBEDDING"}

# One provider per collector, credential and project for the whole process.
_providers: dict[tuple[str, str, str, str], TracerProvider] = {}
_providers_lock = threading.Lock()


def _env_ready() -> tuple[bool, str, str, str]:
    """Both keys and a project name are required. A missing value keeps the tracer inactive."""
    api_key = os.getenv("FI_API_KEY", "")
    secret_key = os.getenv("FI_SECRET_KEY", "")
    project = os.getenv("FI_PROJECT_NAME", "")
    return bool(api_key and secret_key and project), api_key, secret_key, project


def collector_endpoint(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/").removesuffix(_COLLECTOR_PATH)
    return f"{parsed.scheme}://{parsed.netloc}{path}{_COLLECTOR_PATH}"


def span_kind(trace_type: str) -> str:
    return _SPAN_KINDS.get(trace_type, "CHAIN")


def _get_provider(endpoint: str, api_key: str, secret_key: str, project: str) -> TracerProvider:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    key = (endpoint, api_key, secret_key, project)
    with _providers_lock:
        provider = _providers.get(key)
        if provider is None:
            provider = TracerProvider(
                resource=Resource.create({"project_name": project, "project_type": "observe", "service.name": project})
            )
            # Batched so exporting never blocks the event loop; the provider flushes on exit.
            exporter = OTLPSpanExporter(endpoint=endpoint, headers={"X-Api-Key": api_key, "X-Secret-Key": secret_key})
            provider.add_span_processor(BatchSpanProcessor(exporter))
            _providers[key] = provider
        return provider


def _now_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


def _to_text(value: Any) -> str:
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class _ParentedTracer:
    """Tracer that parents spans with no explicit context under a fixed span.

    The LangChain callback starts the outermost run of a component without a parent; this puts
    it under the component's span instead of the runtime context, which the tracing worker does
    not share with the component.
    """

    def __init__(self, tracer: Any, parent: Span) -> None:
        self._tracer = tracer
        self._parent = parent

    def start_span(self, name: str, context: Any = None, *args: Any, **kwargs: Any) -> Span:
        from opentelemetry.trace import set_span_in_context

        if context is None:
            context = set_span_in_context(self._parent)
        return self._tracer.start_span(name, context, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)


class FutureAGITracer(BaseTracer):
    """One Future AGI project per server process. Config comes from the environment."""

    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.trace_name = trace_name
        self.trace_type = trace_type
        # project_name is Langflow's tracing project. The Future AGI project is FI_PROJECT_NAME.
        self.langflow_project = project_name
        self.flow_name = trace_name.split(" - ", maxsplit=1)[0]
        self.flow_id = trace_name.rsplit(" - ", maxsplit=1)[-1]
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self._ready, self._api_key, self._secret_key, self._project = _env_ready()
        self._redact = os.getenv("FI_REDACT", "").lower() == "true"
        self._spans: dict[str, Span] = {}
        self._spans_lock = threading.Lock()
        self._chat_input: str | None = None
        self._chat_output: str | None = None
        self._component_error: Exception | None = None
        if not self._ready:
            return
        try:
            from opentelemetry.trace import SpanKind

            endpoint = collector_endpoint(os.getenv("FI_BASE_URL") or _DEFAULT_ENDPOINT)
            provider = _get_provider(endpoint, self._api_key, self._secret_key, self._project)
            self._tracer = provider.get_tracer(_TRACER_NAME)
            self._root = self._tracer.start_span(self.flow_name, kind=SpanKind.INTERNAL, start_time=_now_ns())
            self._root.set_attributes(self._root_attributes())
        except Exception:  # noqa: BLE001
            logger.exception("[Future AGI] Error setting up the tracer")
            self._ready = False

    @property
    @override
    def ready(self) -> bool:
        return self._ready

    def _root_attributes(self) -> dict[str, str]:
        attributes = {
            "openinference.span.kind": "CHAIN",
            "langflow.flow_name": self.flow_name,
            "langflow.flow_id": self.flow_id,
            "langflow.project_name": str(self.langflow_project),
            "langflow.run_id": str(self.trace_id),
        }
        if self.session_id:
            attributes["session.id"] = self.session_id
        if self.user_id:
            attributes["user.id"] = str(self.user_id)
        return attributes

    def _component_span(self, trace_id: str, trace_name: str, trace_type: str) -> Span:
        """Return the component's span, starting it on first use.

        A component can ask for its LangChain callback before the tracing worker has run
        add_trace for it, so whichever happens first starts the span.
        """
        from opentelemetry.trace import set_span_in_context

        with self._spans_lock:
            span = self._spans.get(trace_id)
            if span is None:
                span = self._tracer.start_span(
                    trace_name, context=set_span_in_context(self._root), start_time=_now_ns()
                )
                span.set_attribute("openinference.span.kind", span_kind(trace_type))
                span.set_attribute("langflow.component_type", trace_type)
                self._spans[trace_id] = span
            return span

    @override
    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        vertex: Vertex | None = None,
    ) -> None:
        if not self.ready:
            return
        span = self._component_span(trace_id, trace_name, trace_type)
        if vertex is not None and vertex.id is not None:
            span.set_attribute("langflow.vertex_id", vertex.id)
        component = trace_id.split("-", maxsplit=1)[0]
        if component == "ChatInput" and inputs and "input_value" in inputs:
            self._chat_input = _to_text(inputs["input_value"])
        elif component == "ChatOutput" and inputs and "input_value" in inputs:
            self._chat_output = _to_text(inputs["input_value"])
        if not self._redact and inputs:
            span.set_attribute("input.value", json.dumps(inputs, default=str))
            span.set_attribute("input.mime_type", "application/json")

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        with self._spans_lock:
            span = self._spans.pop(trace_id, None)
        if span is None:
            return
        if not self._redact and outputs:
            span.set_attribute("output.value", json.dumps(outputs, default=str))
            span.set_attribute("output.mime_type", "application/json")
        if error is not None and self._component_error is None:
            self._component_error = error
        self._finish(span, error)

    @override
    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.ready:
            return
        error = error or self._component_error
        with self._spans_lock:
            unfinished = list(self._spans.values())
            self._spans.clear()
        for span in unfinished:
            self._finish(span, error)
        if not self._redact:
            run_input = self._chat_input if self._chat_input is not None else (inputs or None)
            # Without a chat output, fall back to the component outputs only for a successful run.
            run_output = self._chat_output if self._chat_output is not None else (outputs or None)
            if error is not None and self._chat_output is None:
                run_output = None
            if run_input is not None:
                self._root.set_attribute("input.value", _to_text(run_input))
            if run_output is not None:
                self._root.set_attribute("output.value", _to_text(run_output))
        self._finish(self._root, error)

    @staticmethod
    def _finish(span: Span, error: Exception | None) -> None:
        from opentelemetry.trace import Status, StatusCode

        if error is not None:
            span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
            if isinstance(error, BaseException):
                span.record_exception(error)
        else:
            span.set_status(Status(StatusCode.OK))
        span.end(end_time=_now_ns())

    @override
    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        """OpenInference LangChain callback for the current component, nested under its span."""
        if not self.ready:
            return None
        from langflow.services.tracing.service import component_context_var

        component = component_context_var.get()
        if component is None:
            return None
        try:
            from openinference.instrumentation import OITracer, TraceConfig
            from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
        except ImportError:
            logger.debug("[Future AGI] openinference-instrumentation-langchain is not installed")
            return None
        parent = self._component_span(component.trace_id, component.trace_name, component.trace_type)
        config = TraceConfig(hide_inputs=self._redact, hide_outputs=self._redact)
        tracer = OITracer(self._tracer, config=config)
        return OpenInferenceTracer(_ParentedTracer(tracer, parent), separate_trace_from_runtime_context=False)
