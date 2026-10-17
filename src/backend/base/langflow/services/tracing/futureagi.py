"""Future AGI tracer.

Exports OTLP/HTTP with the headers and resource attributes fi-collector
accepts. The Phoenix and Traceloop tracers authenticate differently and are
not retargeted.

A flow run becomes one trace: a root span for the run, a child span per
component, and the LangChain runs inside a component (LLM calls with model and
token counts, tool calls) nested under that component's span.

Content (inputs and outputs) is exported by default, like the other Langflow
tracers and Future AGI's traceAI instrumentors. ``FI_HIDE_INPUTS`` /
``FI_HIDE_OUTPUTS`` (traceAI names) or ``FI_REDACT`` (both) turn it off. Error
text never carries a stack trace, and the local home directory, OS user name,
host name and e-mail addresses are masked in it in either mode.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
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
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.trace import Span

    from langflow.services.tracing.schema import Log

_DEFAULT_ENDPOINT = "https://api.futureagi.com"
_COLLECTOR_PATH = "/tracer/v1/traces"
_TRACER_NAME = "langflow.futureagi"
_TRUE = {"1", "true", "yes", "on"}
# At most one export-failure line per exporter per interval.
_FAILURE_LOG_INTERVAL_S = 60.0

# OpenInference span kinds for Langflow component trace types. Model components map to CHAIN:
# the LLM span, with model and token counts, comes from the LangChain callback beneath them.
_SPAN_KINDS = {"agent": "AGENT", "tool": "TOOL", "retriever": "RETRIEVER", "embedding": "EMBEDDING"}

# One provider per collector, credential and project for the whole process.
_providers: dict[tuple[str, str, str, str], TracerProvider] = {}
_providers_lock = threading.Lock()


_REQUIRED_ENV = ("FI_API_KEY", "FI_SECRET_KEY", "FI_PROJECT_NAME")
_warned_missing = False


def _env_ready() -> tuple[bool, str, str, str]:
    """Both keys and a project name are required. A missing value keeps the tracer inactive.

    When only some of them are set, one warning per process names the missing ones.
    """
    global _warned_missing  # noqa: PLW0603
    api_key, secret_key, project = (os.getenv(name, "") for name in _REQUIRED_ENV)
    ready = bool(api_key and secret_key and project)
    missing = [name for name, value in zip(_REQUIRED_ENV, (api_key, secret_key, project), strict=True) if not value]
    if not ready and len(missing) < len(_REQUIRED_ENV) and not _warned_missing:
        _warned_missing = True
        logger.warning(f"[Future AGI] tracing is off: {', '.join(missing)} not set. Flows run without it.")
    return ready, api_key, secret_key, project


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def hide_settings() -> tuple[bool, bool]:
    """Return (hide_inputs, hide_outputs) from FI_HIDE_INPUTS / FI_HIDE_OUTPUTS / FI_REDACT."""
    redact = _env_flag("FI_REDACT")
    return redact or _env_flag("FI_HIDE_INPUTS"), redact or _env_flag("FI_HIDE_OUTPUTS")


def collector_endpoint(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/").removesuffix(_COLLECTOR_PATH)
    return f"{parsed.scheme}://{parsed.netloc}{path}{_COLLECTOR_PATH}"


def span_kind(trace_type: str) -> str:
    return _SPAN_KINDS.get(trace_type, "CHAIN")


# --- error-text scrubbing ---------------------------------------------------------------------

_TRACEBACK_MARKER = "Traceback (most recent call last)"
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HOME_PATHS = re.compile(r"(/Users/|/home/|[A-Za-z]:\\Users\\)[^/\\\s'\"]+")
_EXCEPTION_DROP = {"exception.stacktrace"}
_EXCEPTION_SCRUB = {"exception.message"}


# Generic account names (containers, CI) identify nobody and would mask ordinary words.
_MIN_NAME_LENGTH = 3
_GENERIC_NAMES = {"root", "app", "user", "admin", "nobody", "ubuntu", "langflow", "runner", "www-data", "localhost"}


def _local_identity() -> tuple[str, list[str]]:
    """The home directory and the OS user / host names to mask in error text."""
    home = str(Path.home()).rstrip("/\\")
    names: list[str] = []
    try:
        names.append(getpass.getuser())
    except Exception:  # noqa: BLE001
        logger.debug("[Future AGI] could not read the OS user name")
    try:
        host = socket.gethostname()
        names.extend({host, host.split(".")[0]})
    except OSError:
        logger.debug("[Future AGI] could not read the host name")
    names = [n for n in names if n and len(n) >= _MIN_NAME_LENGTH and n.lower() not in _GENERIC_NAMES]
    return home, sorted(set(names), key=len, reverse=True)


_HOME, _LOCAL_NAMES = _local_identity()
_HOME_PREFIX = re.compile(re.escape(_HOME) + r"(?=[/\\\s'\"]|$)") if len(_HOME) > 1 else None


def mask_home(text: str) -> str:
    """Replace this machine's home directory with ``~``."""
    return _HOME_PREFIX.sub("~", text) if _HOME_PREFIX else text


def scrub_error_text(text: str) -> str:
    """Error text without a stack trace, home paths, OS user / host names or e-mail addresses."""
    head, marker, _ = text.partition(_TRACEBACK_MARKER)
    text = head.rstrip() if marker else text
    text = _HOME_PATHS.sub("~", mask_home(text))
    text = _EMAIL.sub("<email>", text)
    for name in _LOCAL_NAMES:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<redacted>", text)
    return text


def _scrub_attributes(attributes: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if isinstance(value, str):
            out[key] = mask_home(value)
        elif isinstance(value, (list, tuple)):
            out[key] = tuple(mask_home(v) if isinstance(v, str) else v for v in value)
        else:
            out[key] = value
    return out


def scrub_span(span: ReadableSpan) -> ReadableSpan:
    """Return a copy of ``span`` that is safe to export.

    LangChain stores every failed run's error as ``repr(error)`` plus the formatted traceback, and
    the OpenInference callback copies it into the span status and an ``exception`` event with an
    ``exception.stacktrace``. Both carry absolute paths that include the OS user name.
    """
    from opentelemetry.sdk.trace import Event, ReadableSpan
    from opentelemetry.trace import Status

    status = span.status
    if status.description:
        status = Status(status.status_code, scrub_error_text(status.description))
    events = []
    for event in span.events:
        attributes = {}
        for key, value in (event.attributes or {}).items():
            if key in _EXCEPTION_DROP:
                continue
            if key in _EXCEPTION_SCRUB and isinstance(value, str):
                attributes[key] = scrub_error_text(value)
            elif isinstance(value, str):
                attributes[key] = mask_home(value)
            else:
                attributes[key] = value
        events.append(Event(event.name, attributes, timestamp=event.timestamp))
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_scrub_attributes(span.attributes),
        events=events,
        links=span.links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class _ScrubbingExporter:
    """Wraps the OTLP exporter: scrubs every span and logs failed exports at most once a minute."""

    def __init__(self, exporter: Any) -> None:
        self._exporter = exporter
        self._last_failure_log = 0.0
        self._dropped = 0

    def export(self, spans: Sequence[ReadableSpan]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            result = self._exporter.export([scrub_span(span) for span in spans])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[Future AGI] export raised {type(exc).__name__}")
            result = SpanExportResult.FAILURE
        if result != SpanExportResult.SUCCESS:
            self._dropped += len(spans)
            now = time.monotonic()
            if now - self._last_failure_log >= _FAILURE_LOG_INTERVAL_S:
                self._last_failure_log = now
                logger.warning(
                    f"[Future AGI] dropped {self._dropped} span(s): export to the collector failed. "
                    "Flow runs are not affected. Check FI_API_KEY, FI_SECRET_KEY, FI_BASE_URL and FI_PROJECT_NAME."
                )
                self._dropped = 0
        return result

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


def _get_provider(endpoint: str, api_key: str, secret_key: str, project: str) -> TracerProvider:
    from opentelemetry.exporter.otlp.proto.http import Compression
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
            # fi-collector rejects gzip bodies, so compression is pinned off even if
            # OTEL_EXPORTER_OTLP_COMPRESSION is set for another exporter in the process.
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers={"X-Api-Key": api_key, "X-Secret-Key": secret_key},
                compression=Compression.NoCompression,
            )
            # Batched so exporting never blocks the event loop; the provider flushes on exit.
            provider.add_span_processor(BatchSpanProcessor(_ScrubbingExporter(exporter)))
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


def _langflow_version() -> str | None:
    try:
        from langflow.utils.version import get_version_info

        return get_version_info()["version"]
    except Exception:  # noqa: BLE001
        return None


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
        flow_id: str | None = None,
        tracing_user_id: str | None = None,
    ) -> None:
        self.trace_name = trace_name
        self.trace_type = trace_type
        # project_name is Langflow's tracing project. The Future AGI project is FI_PROJECT_NAME.
        self.langflow_project = project_name
        self.flow_name = trace_name.split(" - ", maxsplit=1)[0]
        self.flow_id = flow_id or trace_name.rsplit(" - ", maxsplit=1)[-1]
        self.trace_id = trace_id
        self.user_id = user_id
        self.tracing_user_id = tracing_user_id
        self.session_id = session_id
        self._ready, self._api_key, self._secret_key, self._project = _env_ready()
        self._hide_inputs, self._hide_outputs = hide_settings()
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
            "langflow.flow_id": str(self.flow_id),
            "langflow.project_name": str(self.langflow_project),
            "langflow.run_id": str(self.trace_id),
        }
        version = _langflow_version()
        if version:
            attributes["langflow.version"] = version
        if self.session_id:
            attributes["session.id"] = self.session_id
        if self.user_id:
            attributes["user.id"] = str(self.user_id)
        if self.tracing_user_id:
            attributes["langflow.tracing_user_id"] = str(self.tracing_user_id)
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
                if self.session_id:
                    span.set_attribute("session.id", self.session_id)
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
        if not self._hide_inputs and inputs:
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
        if not self._hide_outputs and outputs:
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
        run_input = self._chat_input if self._chat_input is not None else (inputs or None)
        # Without a chat output, fall back to the component outputs only for a successful run.
        run_output = self._chat_output if self._chat_output is not None else (outputs or None)
        if error is not None and self._chat_output is None:
            run_output = None
        if run_input is not None and not self._hide_inputs:
            self._root.set_attribute("input.value", _to_text(run_input))
        if run_output is not None and not self._hide_outputs:
            self._root.set_attribute("output.value", _to_text(run_output))
        self._finish(self._root, error)

    @staticmethod
    def _finish(span: Span, error: Exception | None) -> None:
        from opentelemetry.trace import Status, StatusCode

        if error is not None:
            message = scrub_error_text(f"{type(error).__name__}: {error}")
            span.set_status(Status(StatusCode.ERROR, message))
            # Type and message only: a recorded stack trace would export local paths.
            span.add_event(
                "exception",
                {"exception.type": type(error).__name__, "exception.message": scrub_error_text(str(error))},
            )
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
        config = TraceConfig(hide_inputs=self._hide_inputs, hide_outputs=self._hide_outputs)
        tracer = OITracer(self._tracer, config=config)
        return OpenInferenceTracer(_ParentedTracer(tracer, parent), separate_trace_from_runtime_context=False)
