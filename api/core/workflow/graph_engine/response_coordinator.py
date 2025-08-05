"""
ResponseStreamCoordinator - Coordinates streaming output from response nodes

This component manages response streaming sessions and ensures ordered streaming
of responses based on upstream node outputs and constants.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional, cast
from uuid import uuid4

from core.workflow.enums import NodeType
from core.workflow.events import GraphBaseNodeEvent, GraphEngineEvent, NodeRunStreamChunkEvent, NodeRunSucceededEvent
from core.workflow.graph import Graph
from core.workflow.graph_engine.output_registry import OutputRegistry
from core.workflow.nodes.base.template import Template, TextSegment, VariableSegment


@dataclass
class ResponseSession:
    """Represents an active response streaming session."""

    node_id: str
    template: Template  # Template object from the response node
    index: int = 0  # Current position in the template segments


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
        with self.lock:
            if not self.active_session:
                return []

            # Check if the updated selector matches the current segment
            template = self.active_session.template
            if self.active_session.index < len(template.segments):
                current_segment = template.segments[self.active_session.index]

                # If current segment is a VariableSegment matching this selector, try to flush
                if isinstance(current_segment, VariableSegment):
                    selector_str = ".".join(current_segment.selector)
                    if selector_str == selector:
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

    def _process_variable_segment(self, segment: VariableSegment) -> tuple[list[GraphEngineEvent], bool]:
        """Process a variable segment. Returns (events, is_complete)."""

        events: list[GraphEngineEvent] = []
        source_node_id = segment.selector[0]
        is_complete = False

        if self.registry and self.registry.has_unread(segment.selector):
            # Stream all available chunks
            source_exec_id = self._node_execution_ids.get(source_node_id, str(uuid4()))
            source_node_type = self._get_node_type(source_node_id)

            if source_node_type:
                # Check if this is the last chunk by looking ahead
                stream_closed = self.registry.stream_closed(segment.selector)

                while self.registry and self.registry.has_unread(segment.selector):
                    if chunk := self.registry.pop_chunk(segment.selector):
                        # Check if this is the final chunk
                        has_more = self.registry.has_unread(segment.selector)
                        is_final_chunk = stream_closed and not has_more

                        events.append(
                            self._create_stream_chunk_event(
                                node_id=source_node_id,
                                node_type=source_node_type,
                                execution_id=source_exec_id,
                                selector=segment.selector,
                                chunk=chunk,
                                is_final=is_final_chunk,
                            )
                        )

            # Check if stream is closed to determine if segment is complete
            if self.registry and self.registry.stream_closed(segment.selector):
                is_complete = True

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
            is_complete = True

        return events, is_complete

    def _process_text_segment(
        self, segment: TextSegment, response_node_id: str, response_node_exec_id: Optional[str]
    ) -> tuple[list[GraphEngineEvent], bool]:
        """Process a text segment. Returns (events, is_complete)."""
        if not response_node_exec_id:
            return [], False  # Cannot process without execution ID

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
            return [event], True  # Text segments are always immediately complete

        return [], True  # Complete even without type info

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

            events: list[GraphEngineEvent] = []

            # Process segments sequentially from current index
            while self.active_session.index < len(template.segments):
                segment = template.segments[self.active_session.index]

                if isinstance(segment, VariableSegment):
                    segment_events, is_complete = self._process_variable_segment(segment)
                    events.extend(segment_events)

                    # Only advance index if this variable segment is complete
                    if is_complete:
                        self.active_session.index += 1
                    else:
                        # Wait for more data
                        break

                elif isinstance(segment, TextSegment):
                    segment_events, is_complete = self._process_text_segment(
                        segment, response_node_id, response_node_exec_id
                    )
                    events.extend(segment_events)

                    # Text segments are always immediately complete
                    if is_complete:
                        self.active_session.index += 1
                    else:
                        # Cannot proceed without execution ID
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
