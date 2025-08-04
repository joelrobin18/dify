"""
ResponseStreamCoordinator - Coordinates streaming output from response nodes

This component manages response streaming sessions and ensures ordered streaming
of responses based on upstream node outputs and constants.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Optional, cast

from core.workflow.enums import NodeType
from core.workflow.events import GraphBaseNodeEvent, GraphEngineEvent, NodeRunStreamChunkEvent, NodeRunSucceededEvent
from core.workflow.graph_engine.output_registry import OutputRegistry
from core.workflow.nodes.base.template import Template, TextSegment, VariableSegment

if TYPE_CHECKING:
    from core.workflow.graph import Graph


@dataclass
class ResponseSession:
    """Represents an active response streaming session."""

    node_id: str
    template: Template  # Template object from the response node
    streamed_segments: set[int] = field(default_factory=set)  # Track which segments have been streamed
    pending_text_segments: set[int] = field(default_factory=set)  # Text segments waiting for node execution ID


class ResponseStreamCoordinator:
    """
    Manages response streaming sessions without relying on global state.

    Ensures ordered streaming of responses based on upstream node outputs and constants.
    """

    def __init__(self, registry: Optional[OutputRegistry] = None, graph: Optional["Graph"] = None) -> None:
        """
        Initialize coordinator with output registry.

        Args:
            registry: OutputRegistry instance for accessing node outputs
            graph: Graph instance for looking up node information
        """
        self.registry = registry
        self.graph = graph
        self.active_session: Optional[ResponseSession] = None
        self.waiting_sessions: deque[ResponseSession] = deque()
        self.lock = RLock()

        # Track response nodes and their dependencies
        self._response_nodes: set[str] = set()
        self._dep_map: dict[str, dict[str, Any]] = {}
        self._edge_to_responses: dict[str, set[str]] = {}

        # Track node execution IDs and types for proper event forwarding
        self._node_execution_ids: dict[str, str] = {}  # node_id -> execution_id
        self._node_types: dict[str, Any] = {}  # node_id -> node_type

    def register(self, response_node_id: str, dependencies: list[str] | None = None) -> None:
        """
        Register a response node with its dependencies.

        Args:
            response_node_id: ID of the response node to register
            dependencies: List of edge IDs this node depends on (optional)
        """
        with self.lock:
            deps = dependencies or []

            # Initialize dependency tracking
            self._dep_map[response_node_id] = {"deps": set(deps), "unresolved": set(deps), "cancelled": False}

            # Add to response nodes set
            self._response_nodes.add(response_node_id)

            # Update reverse index
            for edge_id in deps:
                if edge_id not in self._edge_to_responses:
                    self._edge_to_responses[edge_id] = set()
                self._edge_to_responses[edge_id].add(response_node_id)

    def on_session_start(self, session: ResponseSession) -> None:
        """
        Handle the start of a new response session.

        Args:
            session: The response session to start
        """
        with self.lock:
            if self.active_session is None:
                self.active_session = session

    def on_variable_update(self, selector: str) -> list[GraphEngineEvent]:
        """
        Handle variable updates from node outputs.

        Args:
            selector: The selector that was updated

        Returns:
            List of streaming events to be emitted
        """
        _ = selector  # Mark as intentionally unused
        with self.lock:
            if self.active_session:
                return self.try_flush()
            return []

    def intercept_event(self, event: GraphEngineEvent, response_node_id: str) -> Optional[GraphEngineEvent]:
        """
        Intercept and potentially transform an event for a response node.

        Args:
            event: The event to intercept
            response_node_id: The ID of the response node that will emit this event

        Returns:
            The transformed event if it should be emitted, None if it should be suppressed
        """
        with self.lock:
            # Store node execution IDs and types for tracking
            # Only GraphBaseNodeEvent instances have these attributes
            if isinstance(event, GraphBaseNodeEvent):
                self._node_execution_ids[event.node_id] = event.id
                self._node_types[event.node_id] = event.node_type

            # Only process if we have an active session for this response node
            if not self.active_session or self.active_session.node_id != response_node_id:
                return None

            # Handle NodeRunStreamChunkEvent
            if isinstance(event, NodeRunStreamChunkEvent):
                # Don't forward streaming events directly - they need to follow template order
                # Just store them in the output registry (already done by GraphEngine)
                return None

            # Handle NodeRunSucceededEvent
            elif isinstance(event, NodeRunSucceededEvent):
                # Don't convert to stream chunk - the scalar output will be handled by try_flush
                # when it processes the template in order
                return None

            return None

    def _get_node_type(self, node_id: str) -> Optional[NodeType]:
        """Get node type from cache or graph."""
        if node_type := self._node_types.get(node_id):
            return cast(NodeType, node_type)

        if self.graph and node_id in self.graph.nodes:
            return cast(NodeType, self.graph.nodes[node_id].type_)

        return None

    def _create_stream_chunk_event(
        self,
        node_id: str,
        node_type: NodeType,
        execution_id: str,
        selector: Sequence[str],
        chunk: str,
        is_final: bool = False,
    ) -> NodeRunStreamChunkEvent:
        """Create a stream chunk event with consistent structure."""
        return NodeRunStreamChunkEvent(
            id=execution_id,
            node_id=node_id,
            node_type=node_type,
            selector=selector,
            chunk=chunk,
            is_final=is_final,
            # Legacy fields
            chunk_content=chunk,
            from_variable_selector=list(selector),
        )

    def _process_pending_text_segments(
        self, response_node_id: str, response_node_exec_id: str, template: Template
    ) -> list[GraphEngineEvent]:
        """Process any pending text segments."""
        events: list[GraphEngineEvent] = []

        if not (response_node_exec_id and self.active_session and self.active_session.pending_text_segments):
            return events

        response_node_type = self._get_node_type(response_node_id)
        if not response_node_type:
            return events

        for seg_idx in sorted(self.active_session.pending_text_segments):
            if seg_idx >= len(template.segments) or seg_idx in self.active_session.streamed_segments:
                continue

            segment = template.segments[seg_idx]
            if isinstance(segment, TextSegment):
                events.append(
                    self._create_stream_chunk_event(
                        node_id=response_node_id,
                        node_type=response_node_type,
                        execution_id=response_node_exec_id,
                        selector=[response_node_id, "output"],
                        chunk=segment.text,
                        is_final=False,
                    )
                )
                self.active_session.streamed_segments.add(seg_idx)

        self.active_session.pending_text_segments.clear()
        return events

    def _process_variable_segment(self, segment: VariableSegment, segment_index: int) -> list[GraphEngineEvent]:
        """Process a variable segment."""
        from uuid import uuid4

        events: list[GraphEngineEvent] = []
        source_node_id = segment.selector[0]

        if self.registry and self.registry.has_unread(segment.selector):
            # Stream all available chunks
            source_exec_id = self._node_execution_ids.get(source_node_id, str(uuid4()))
            source_node_type = self._get_node_type(source_node_id)

            if source_node_type:
                while self.registry and self.registry.has_unread(segment.selector):
                    if chunk := self.registry.pop_chunk(segment.selector):
                        events.append(
                            self._create_stream_chunk_event(
                                node_id=source_node_id,
                                node_type=source_node_type,
                                execution_id=source_exec_id,
                                selector=segment.selector,
                                chunk=chunk,
                                is_final=False,
                            )
                        )

            if self.registry and self.registry.stream_closed(segment.selector) and self.active_session:
                self.active_session.streamed_segments.add(segment_index)

        elif self.registry and (value := self.registry.get_scalar(segment.selector)):
            # Process scalar value
            source_exec_id = self._node_execution_ids.get(source_node_id, str(uuid4()))
            source_node_type = self._get_node_type(source_node_id)

            if source_node_type:
                events.append(
                    self._create_stream_chunk_event(
                        node_id=source_node_id,
                        node_type=source_node_type,
                        execution_id=source_exec_id,
                        selector=segment.selector,
                        chunk=str(value),
                        is_final=True,
                    )
                )
            if self.active_session:
                self.active_session.streamed_segments.add(segment_index)

        return events

    def _process_text_segment(
        self, segment: TextSegment, segment_index: int, response_node_id: str, response_node_exec_id: Optional[str]
    ) -> tuple[list[GraphEngineEvent], bool]:
        """Process a text segment. Returns (events, should_continue)."""
        if not response_node_exec_id:
            if self.active_session:
                self.active_session.pending_text_segments.add(segment_index)
            return [], False  # Stop processing to maintain order

        response_node_type = self._get_node_type(response_node_id)
        if response_node_type:
            event = self._create_stream_chunk_event(
                node_id=response_node_id,
                node_type=response_node_type,
                execution_id=response_node_exec_id,
                selector=[response_node_id, "output"],
                chunk=segment.text,
                is_final=False,
            )
            if self.active_session:
                self.active_session.streamed_segments.add(segment_index)
            return [event], True

        if self.active_session:
            self.active_session.streamed_segments.add(segment_index)
        return [], True

    def try_flush(self) -> list[GraphEngineEvent]:
        """Try to flush output from the active session.

        Returns:
            List of events to be emitted
        """
        with self.lock:
            if not (self.active_session and self.registry):
                return []

            template = self.active_session.template
            response_node_id = self.active_session.node_id
            response_node_exec_id = self._node_execution_ids.get(response_node_id)

            # Process pending text segments first
            events: list[GraphEngineEvent] = []
            if response_node_exec_id:
                events = self._process_pending_text_segments(response_node_id, response_node_exec_id, template)

            # Process each segment in order
            for i, segment in enumerate(template.segments):
                if i in self.active_session.streamed_segments:
                    continue

                if isinstance(segment, VariableSegment):
                    events.extend(self._process_variable_segment(segment, i))
                elif isinstance(segment, TextSegment):
                    segment_events, should_continue = self._process_text_segment(
                        segment, i, response_node_id, response_node_exec_id
                    )
                    events.extend(segment_events)
                    if not should_continue:
                        break

            return events

    def end_session(self, node_id: str) -> None:
        """
        End the active session for a response node.

        Args:
            node_id: ID of the response node ending its session
        """
        with self.lock:
            if self.active_session and self.active_session.node_id == node_id:
                self.active_session = None

                # Try to start next waiting session
                if self.waiting_sessions:
                    next_session = self.waiting_sessions.popleft()
                    self.on_session_start(next_session)

    def is_response_node(self, node_id: str) -> bool:
        """
        Check if a node is registered as a response node.

        Args:
            node_id: ID of the node to check

        Returns:
            True if the node is a registered response node
        """
        with self.lock:
            return node_id in self._response_nodes

    def get_active_session(self) -> Optional[ResponseSession]:
        """
        Get the currently active session.

        Returns:
            The active ResponseSession or None
        """
        with self.lock:
            return self.active_session
