"""
ResponseStreamCoordinator - Coordinates streaming output from response nodes

This component manages response streaming sessions and ensures ordered streaming
of responses based on upstream node outputs and constants.
"""

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Optional

from core.workflow.events import GraphEngineEvent, NodeRunStreamChunkEvent, NodeRunSucceededEvent
from core.workflow.graph_engine.output_registry import OutputRegistry
from core.workflow.nodes.base.template import Template

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

    def on_variable_update(self, selector: str) -> list[dict[str, Any]]:
        """
        Handle variable updates from node outputs.

        Args:
            selector: The selector that was updated

        Returns:
            List of streaming events to be emitted
        """
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
            if hasattr(event, "node_id") and hasattr(event, "id"):
                self._node_execution_ids[event.node_id] = event.id
            if hasattr(event, "node_id") and hasattr(event, "node_type"):
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

    def try_flush(self) -> list[GraphEngineEvent]:
        """Try to flush output from the active session.

        Returns:
            List of events to be emitted
        """
        with self.lock:
            if not self.active_session or not self.active_session.template or not self.registry:
                return []

            from uuid import uuid4

            from core.workflow.nodes.base.template import VariableSegment

            events = []
            template = self.active_session.template
            response_node_id = self.active_session.node_id

            # Get the response node's execution ID
            response_node_exec_id = self._node_execution_ids.get(response_node_id)

            # First, try to flush any pending text segments if we now have the execution ID
            if response_node_exec_id and self.active_session.pending_text_segments:
                for seg_idx in sorted(self.active_session.pending_text_segments):
                    if seg_idx < len(template.segments) and seg_idx not in self.active_session.streamed_segments:
                        segment = template.segments[seg_idx]
                        if hasattr(segment, "text"):  # TextSegment
                            response_node_type = self._node_types.get(response_node_id)
                            if response_node_type is None and self.graph and response_node_id in self.graph.nodes:
                                response_node_type = self.graph.nodes[response_node_id].type_

                            events.append(
                                NodeRunStreamChunkEvent(
                                    id=response_node_exec_id,
                                    node_id=response_node_id,
                                    node_type=response_node_type,
                                    selector=[response_node_id, "output"],
                                    chunk=segment.text,
                                    is_final=False,
                                    # Legacy fields
                                    chunk_content=segment.text,
                                    from_variable_selector=[response_node_id, "output"],
                                )
                            )
                            self.active_session.streamed_segments.add(seg_idx)
                self.active_session.pending_text_segments.clear()

            # Process each segment in the template IN ORDER
            # We must respect the template order, so if we can't stream a segment,
            # we must stop and wait (not skip to later segments)
            for i, segment in enumerate(template.segments):
                # Skip already streamed segments
                if i in self.active_session.streamed_segments:
                    continue

                if isinstance(segment, VariableSegment):
                    # First check for streaming data
                    has_stream = self.registry.has_unread(segment.selector)

                    if has_stream:
                        # Stream all available chunks
                        source_node_id = segment.selector[0]
                        source_exec_id = self._node_execution_ids.get(source_node_id, str(uuid4()))
                        source_node_type = self._node_types.get(source_node_id)

                        if source_node_type is None and self.graph and source_node_id in self.graph.nodes:
                            source_node_type = self.graph.nodes[source_node_id].type_

                        while self.registry.has_unread(segment.selector):
                            chunk = self.registry.pop_chunk(segment.selector)
                            if chunk:
                                events.append(
                                    NodeRunStreamChunkEvent(
                                        id=source_exec_id,
                                        node_id=source_node_id,
                                        node_type=source_node_type,
                                        selector=segment.selector,
                                        chunk=chunk,
                                        is_final=False,
                                        # Legacy fields
                                        chunk_content=chunk,
                                        from_variable_selector=segment.selector,
                                    )
                                )

                        # If stream is closed, mark segment as processed
                        if self.registry.stream_closed(segment.selector):
                            self.active_session.streamed_segments.add(i)
                    else:
                        # No streaming data, check for scalar value
                        value = self.registry.get_scalar(segment.selector)
                        if value is not None:
                            # Find the original node's execution ID and type
                            source_node_id = segment.selector[0]
                            source_exec_id = self._node_execution_ids.get(source_node_id, str(uuid4()))
                            source_node_type = self._node_types.get(source_node_id)

                            # If we don't have the node type cached, get it from the graph
                            if source_node_type is None and self.graph and source_node_id in self.graph.nodes:
                                source_node_type = self.graph.nodes[source_node_id].type_

                            # Create a stream chunk event with the source node's ID
                            events.append(
                                NodeRunStreamChunkEvent(
                                    id=source_exec_id,
                                    node_id=source_node_id,
                                    node_type=source_node_type,
                                    selector=segment.selector,
                                    chunk=str(value),
                                    is_final=True,
                                    # Legacy fields
                                    chunk_content=str(value),
                                    from_variable_selector=segment.selector,
                                )
                            )
                            self.active_session.streamed_segments.add(i)
                        else:
                            # Variable not ready yet, don't process further variable segments
                            # but we may have text segments to defer
                            pass
                else:
                    # Text segment - this is from the response node itself
                    # We need the response node's execution ID for this
                    if response_node_exec_id is None:
                        # Response node hasn't started yet, mark as pending
                        self.active_session.pending_text_segments.add(i)
                        # STOP processing further segments to maintain order
                        break

                    response_node_type = self._node_types.get(response_node_id)

                    # If we don't have the node type cached, get it from the graph
                    if response_node_type is None and self.graph and response_node_id in self.graph.nodes:
                        response_node_type = self.graph.nodes[response_node_id].type_

                    events.append(
                        NodeRunStreamChunkEvent(
                            id=response_node_exec_id,
                            node_id=response_node_id,
                            node_type=response_node_type,
                            selector=[response_node_id, "output"],
                            chunk=segment.text,
                            is_final=False,
                            # Legacy fields
                            chunk_content=segment.text,
                            from_variable_selector=[response_node_id, "output"],
                        )
                    )
                    self.active_session.streamed_segments.add(i)

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
