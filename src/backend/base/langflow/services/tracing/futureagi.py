"""Future AGI tracer.

Exports OTLP/HTTP with the headers and resource attributes fi-collector
accepts. The Phoenix and Traceloop tracers authenticate differently and are
not retargeted.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from uuid import UUID

from typing_extensions import override

from langflow.services.tracing.base import BaseTracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks.base import BaseCallbackHandler
    from lfx.graph.vertex.base import Vertex

    from langflow.services.tracing.schema import Log

_DEFAULT_ENDPOINT = "https://api.futureagi.com"
_COLLECTOR_PATH = "/tracer/v1/traces"


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


def span_attributes(
    *,
    session_id: str | None,
    user_id: str | None,
    flow_name: str,
    inputs: dict[str, Any] | None,
    outputs: dict[str, Any] | None,
    redact: bool,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"openinference.span.kind": "CHAIN", "langflow.flow_name": flow_name}
    if session_id:
        attributes["session.id"] = session_id
    if user_id:
        attributes["user.id"] = user_id
    if not redact:
        if inputs is not None:
            attributes["input.value"] = json.dumps(inputs, default=str)
        if outputs is not None:
            attributes["output.value"] = json.dumps(outputs, default=str)
    return attributes


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
        # project_name is the Langflow flow name. The Future AGI project is FI_PROJECT_NAME.
        self.flow_name = project_name
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self._ready, self._api_key, self._secret_key, self._project = _env_ready()
        self._redact = os.getenv("FI_REDACT", "").lower() == "true"
        self._components: dict[str, dict[str, Any]] = {}

    @property
    @override
    def ready(self) -> bool:
        return self._ready

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
        self._components[trace_id] = {
            "name": trace_name,
            "type": trace_type,
            "inputs": None if self._redact else inputs,
        }

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        component = self._components.get(trace_id)
        if component is None:
            return
        component["outputs"] = None if self._redact else outputs
        component["error"] = error

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
        self._export(inputs, outputs, error)

    def _export(self, inputs: dict[str, Any], outputs: dict[str, Any], error: Exception | None) -> None:
        """Build the exporter lazily so a missing optional dependency never breaks startup."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            return
        base_url = os.getenv("FI_BASE_URL", _DEFAULT_ENDPOINT)
        exporter = OTLPSpanExporter(
            endpoint=collector_endpoint(base_url),
            headers={"X-Api-Key": self._api_key, "X-Secret-Key": self._secret_key},
        )
        provider = TracerProvider(
            resource=Resource.create({"project_name": self._project, "project_type": "observe"})
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("langflow-futureagi")
        with tracer.start_as_current_span(self.trace_name) as span:
            for key, value in span_attributes(
                session_id=self.session_id,
                user_id=self.user_id,
                flow_name=self.flow_name,
                inputs=inputs,
                outputs=outputs,
                redact=self._redact,
            ).items():
                span.set_attribute(key, value)
            if error is not None:
                span.record_exception(error)

    @override
    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        # A traceAI callback needs a LangChain version match that is not verified
        # here. No callback is safer than one that double-counts spans.
        return None
