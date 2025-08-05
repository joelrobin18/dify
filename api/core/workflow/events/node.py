from collections.abc import Sequence
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from core.model_runtime.entities.llm_entities import LLMUsage
from core.rag.entities.citation_metadata import RetrievalSourceMetadata

from .base import GraphBaseNodeEvent, NodeEvent, NodeRunResult


class AgentNodeStrategyInit(BaseModel):
    name: str
    icon: str | None = None


class NodeRunStartedEvent(GraphBaseNodeEvent):
    node_title: str
    predecessor_node_id: Optional[str] = None
    parallel_mode_run_id: Optional[str] = None
    agent_strategy: Optional[AgentNodeStrategyInit] = None
    start_at: datetime = Field(..., description="node start time")
    node_run_index: int = Field(..., description="node run index")

    # FIXME(-LAN-): only for ToolNode
    provider_type: str = ""
    provider_id: str = ""


class NodeRunStreamChunkEvent(GraphBaseNodeEvent):
    # Spec-compliant fields
    selector: Sequence[str] = Field(
        ..., description="selector identifying the output location (e.g., ['nodeA', 'text'])"
    )
    chunk: str = Field(..., description="the actual chunk content")
    is_final: bool = Field(default=False, description="indicates if this is the last chunk")

    # Legacy fields for backward compatibility - will be removed later
    chunk_content: str = Field(default="", description="[DEPRECATED] chunk content")
    from_variable_selector: Optional[list[str]] = Field(default=None, description="[DEPRECATED] variable selector")


class NodeRunRetrieverResourceEvent(GraphBaseNodeEvent):
    retriever_resources: Sequence[RetrievalSourceMetadata] = Field(..., description="retriever resources")
    context: str = Field(..., description="context")


class NodeRunSucceededEvent(GraphBaseNodeEvent):
    start_at: datetime = Field(..., description="node start time")


class NodeRunFailedEvent(GraphBaseNodeEvent):
    error: str = Field(..., description="error")
    start_at: datetime = Field(..., description="node start time")


class NodeRunExceptionEvent(GraphBaseNodeEvent):
    error: str = Field(..., description="error")
    start_at: datetime = Field(..., description="node start time")


class NodeInIterationFailedEvent(GraphBaseNodeEvent):
    error: str = Field(..., description="error")
    start_at: datetime = Field(..., description="node start time")


class NodeInLoopFailedEvent(GraphBaseNodeEvent):
    error: str = Field(..., description="error")
    start_at: datetime = Field(..., description="node start time")


class NodeRunRetryEvent(NodeRunStartedEvent):
    error: str = Field(..., description="error")
    retry_index: int = Field(..., description="which retry attempt is about to be performed")


class NodeRunCompletedEvent(NodeEvent):
    run_result: NodeRunResult = Field(..., description="run result")


class RunRetrieverResourceEvent(NodeEvent):
    retriever_resources: Sequence[RetrievalSourceMetadata] = Field(..., description="retriever resources")
    context: str = Field(..., description="context")


class ModelInvokeCompletedEvent(NodeEvent):
    text: str
    usage: LLMUsage
    finish_reason: str | None = None


class RunRetryEvent(NodeEvent):
    error: str = Field(..., description="error")
    retry_index: int = Field(..., description="Retry attempt number")
    start_at: datetime = Field(..., description="Retry start time")
