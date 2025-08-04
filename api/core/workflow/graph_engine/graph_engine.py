"""
QueueBasedGraphEngine - Main orchestrator for queue-based workflow execution

This engine replaces the thread pool architecture with a queue-based dispatcher + worker model
for improved control and coordination of workflow execution.
"""

import queue
import threading
import time
from collections.abc import Generator, Mapping
from typing import Any, Optional

from core.app.entities.app_invoke_entities import InvokeFrom
from core.workflow.entities import GraphRuntimeState
from core.workflow.enums import NodeState, NodeType
from core.workflow.events import (
    GraphBaseNodeEvent,
    GraphEngineEvent,
    GraphRunFailedEvent,
    GraphRunStartedEvent,
    GraphRunSucceededEvent,
    NodeRunFailedEvent,
    NodeRunStartedEvent,
    NodeRunStreamChunkEvent,
    NodeRunSucceededEvent,
)
from core.workflow.graph import Graph
from core.workflow.graph.edge import Edge
from core.workflow.graph_engine.output_registry import OutputRegistry
from core.workflow.graph_engine.response_coordinator import ResponseStreamCoordinator
from core.workflow.graph_engine.worker import Worker
from models.enums import UserFrom
from models.workflow import WorkflowType


class GraphEngine:
    """
    Queue-based graph execution engine.

    Uses a single dispatcher thread + 10 worker threads with queues for
    communication instead of the traditional thread pool approach.
    """

    def __init__(
        self,
        tenant_id: str,
        app_id: str,
        workflow_type: WorkflowType,
        workflow_id: str,
        user_id: str,
        user_from: UserFrom,
        invoke_from: InvokeFrom,
        call_depth: int,
        graph: Graph,
        graph_config: Mapping[str, Any],
        graph_runtime_state: GraphRuntimeState,
        max_execution_steps: int,
        max_execution_time: int,
        thread_pool_id: Optional[str] = None,  # Unused in queue-based engine
    ) -> None:
        """
        Initialize queue-based graph engine.

        Args:
            tenant_id: Tenant identifier
            app_id: Application identifier
            workflow_type: Type of workflow (WORKFLOW or CHAT)
            workflow_id: Workflow identifier
            user_id: User identifier
            user_from: Source of user (ACCOUNT, etc.)
            invoke_from: Invocation source (WEB_APP, etc.)
            call_depth: Nested call depth
            graph: Graph to execute
            graph_config: Graph configuration
            graph_runtime_state: Runtime state
            max_execution_steps: Maximum execution steps
            max_execution_time: Maximum execution time in seconds
            thread_pool_id: Optional thread pool identifier (unused in queue-based)
        """
        # Store initialization parameters
        self.tenant_id = tenant_id
        self.app_id = app_id
        self.workflow_type = workflow_type
        self.workflow_id = workflow_id
        self.user_id = user_id
        self.user_from = user_from
        self.invoke_from = invoke_from
        self.call_depth = call_depth
        self.graph = graph
        self.graph_config = graph_config
        self.graph_runtime_state = graph_runtime_state
        self.max_execution_steps = max_execution_steps
        self.max_execution_time = max_execution_time
        # thread_pool_id is unused in queue-based engine but kept for compatibility
        _ = thread_pool_id

        # Core queue-based architecture components
        self.ready_queue: queue.Queue[str] = queue.Queue()
        self.event_queue: queue.Queue[GraphEngineEvent] = queue.Queue()
        self.state_lock = threading.RLock()

        # Subsystems
        self.output_registry = OutputRegistry()
        self.response_coordinator = ResponseStreamCoordinator(output_registry=self.output_registry)

        # Worker threads (10 workers as specified)
        self.workers: list[Worker] = []
        for i in range(10):
            worker = Worker(
                ready_queue=self.ready_queue,
                event_queue=self.event_queue,
                graph=self.graph,
                worker_id=i,
            )
            self.workers.append(worker)

        # Dispatcher thread
        self.dispatcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._execution_complete = threading.Event()

        # Execution state
        self._started = False
        self._error: Optional[Exception] = None

        # Event collection for generator
        self._collected_events: list[GraphEngineEvent] = []
        self._event_collector_lock = threading.Lock()

        # Track nodes currently being executed
        self._executing_nodes: set[str] = set()
        self._executing_nodes_lock = threading.Lock()

        # Validate that all nodes share the same GraphRuntimeState instance
        # This is critical for thread-safe execution and consistent state management
        expected_state_id = id(self.graph_runtime_state)
        for node in self.graph.nodes.values():
            if id(node.graph_runtime_state) != expected_state_id:
                raise ValueError(
                    f"GraphRuntimeState consistency violation: Node '{node.id}' has a different "
                    f"GraphRuntimeState instance than the engine. All nodes must share the same "
                    f"GraphRuntimeState instance for proper execution."
                )

    def run(self) -> Generator[GraphEngineEvent, None, None]:
        """
        Execute the graph and yield events as they occur.

        Returns:
            Generator yielding GraphEngineEvent instances during execution
        """
        try:
            # Yield initial start event
            start_event = GraphRunStartedEvent()
            yield start_event

            # Start execution
            self._start_execution()

            # Yield events as they're generated
            yield from self._event_generator()

            # Check for errors
            if self._error:
                raise self._error

            # Yield completion event
            success_event = GraphRunSucceededEvent(
                outputs=self.graph_runtime_state.outputs,
            )
            yield success_event

        except Exception as e:
            # Yield failure event
            failure_event = GraphRunFailedEvent(
                error=str(e),
            )
            yield failure_event
            raise

        finally:
            # Clean up
            self._stop_execution()

    def _start_execution(self) -> None:
        """Start the execution by launching workers and dispatcher."""
        if self._started:
            return

        self._started = True

        # Start all worker threads
        for worker in self.workers:
            worker.start()

        # Find root node and add to ready queue
        if not self.graph.root_node:
            raise ValueError("No root node found in graph")
        root_node = self.graph.root_node
        root_node.state = NodeState.TAKEN
        self.ready_queue.put(root_node.id)

        # Start dispatcher thread
        self.dispatcher_thread = threading.Thread(target=self._dispatcher_loop, name="GraphDispatcher", daemon=True)
        self.dispatcher_thread.start()

    def _stop_execution(self) -> None:
        """Stop execution and clean up threads."""
        # Signal stop
        self._stop_event.set()

        # Stop all workers
        for worker in self.workers:
            worker.stop()

        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)

        # Wait for dispatcher
        if self.dispatcher_thread and self.dispatcher_thread.is_alive():
            self.dispatcher_thread.join(timeout=1.0)

    def _dispatcher_loop(self) -> None:
        """Main dispatcher loop that processes events from workers."""
        start_time = time.time()

        try:
            # Give workers a moment to start
            time.sleep(0.01)

            while not self._stop_event.is_set():
                # Check for timeout
                if time.time() - start_time > self.max_execution_time:
                    raise TimeoutError(f"Execution exceeded maximum time of {self.max_execution_time} seconds")

                try:
                    # Get event from queue with timeout
                    event = self.event_queue.get(timeout=0.1)

                    # Process the event
                    self._process_event(event)

                    # Mark task done
                    self.event_queue.task_done()

                except queue.Empty:
                    # Check if we should exit (no more work) after some empty cycles
                    if self._should_complete_execution():
                        # Wait a bit more to make sure nothing is in flight
                        time.sleep(0.05)
                        if self._should_complete_execution():
                            break
                    continue

            # Mark execution as complete
            self._execution_complete.set()

        except Exception as e:
            self._error = e
            self._execution_complete.set()

    def _process_event(self, event: GraphEngineEvent) -> None:
        """
        Process an event from a worker.

        Args:
            event: Event to process
        """
        with self._event_collector_lock:
            self._collected_events.append(event)

        if isinstance(event, GraphBaseNodeEvent):
            node_id = event.node_id

            # Track node execution state
            if isinstance(event, NodeRunStartedEvent):
                with self._executing_nodes_lock:
                    self._executing_nodes.add(node_id)
            elif isinstance(event, (NodeRunSucceededEvent, NodeRunFailedEvent)):
                with self._executing_nodes_lock:
                    self._executing_nodes.discard(node_id)

            # Handle streaming chunk events
            if isinstance(event, NodeRunStreamChunkEvent):
                # Write chunk to output registry
                self.output_registry.append_chunk(event.selector, event.chunk)
                if event.is_final:
                    self.output_registry.close_stream(event.selector)
                # Notify response coordinator about variable update
                self.response_coordinator.on_variable_update(event.selector)

            # Check if this is a response node
            if self.response_coordinator.is_response_node(node_id):
                # Handle response node events
                self.response_coordinator.on_edge_update(node_id)

            # Handle node completion - add successor nodes to ready queue
            if isinstance(event, NodeRunSucceededEvent):
                # Add node outputs to variable pool
                for output_key, output_value in event.node_run_result.outputs.items():
                    self.graph_runtime_state.variable_pool.add((node_id, output_key), output_value)
                    self.output_registry.set_scalar([node_id, output_key], output_value)

                # Check if this is an End node and collect its outputs
                node = self.graph.get_node(node_id)
                if node and node.type_ == NodeType.END:
                    self._collect_end_node_outputs(event)

                self._enqueue_successor_nodes(event)

    def _collect_end_node_outputs(self, event: NodeRunSucceededEvent) -> None:
        outputs = event.node_run_result.outputs
        self.graph_runtime_state.outputs.update(outputs)

    def _enqueue_successor_nodes(self, event: NodeRunSucceededEvent) -> None:
        """
        Add successor nodes to the ready queue after a node completes.

        This method marks outgoing edges as Taken/Skipped based on conditional logic,
        then enqueues successor nodes that have all their incoming edges resolved.

        Args:
            event: NodeRunSucceededEvent containing the completed node information
        """
        completed_node_id = event.node_id
        edge_source_handle = event.node_run_result.edge_source_handle

        # Get outgoing edges from this node
        outgoing_edge_ids = self.graph.out_edges.get(completed_node_id, [])

        # Process each outgoing edge
        taken_edges: list[Edge] = []
        skipped_edges: list[Edge] = []

        for edge_id in outgoing_edge_ids:
            edge = self.graph.edges.get(edge_id)
            if edge is None:
                continue
            # Check if this is a conditional edge (has non-default source_handle)
            if edge.source_handle != "source":
                # For conditional edges, check if it matches the taken branch
                if edge_source_handle and edge.source_handle == edge_source_handle:
                    edge.state = NodeState.TAKEN
                    taken_edges.append(edge)
                else:
                    edge.state = NodeState.SKIPPED
                    skipped_edges.append(edge)
            else:
                # For non-conditional edges, always mark as taken
                edge.state = NodeState.TAKEN
                taken_edges.append(edge)

        # Propagate skipped status to downstream nodes
        for edge in skipped_edges:
            self._mark_node_and_descendants_skipped(edge.head)

        # Check if successor nodes are ready to execute
        checked_nodes: set[str] = set()  # Avoid checking the same node multiple times

        for edge in taken_edges:
            target_node_id = edge.head

            # Skip if we already checked this node
            if target_node_id in checked_nodes:
                continue
            checked_nodes.add(target_node_id)

            # Check if the node is ready (no unknown edges, at least one taken)
            if self._is_node_ready(target_node_id):
                self.ready_queue.put(target_node_id)

    def _mark_node_and_descendants_skipped(self, node_id: str) -> None:
        """
        Mark a node and all its descendants as skipped if appropriate.

        A node is only skipped if ALL its incoming edges are skipped.
        This prevents skipping nodes that have alternative paths.

        Args:
            node_id: The ID of the node to potentially mark as skipped
        """
        # First, check if this node should be skipped
        incoming_edge_ids = self.graph.in_edges.get(node_id, [])

        # If node has no incoming edges, it shouldn't be skipped
        if not incoming_edge_ids:
            return

        # Check if all incoming edges are skipped
        all_skipped = True
        for edge_id in incoming_edge_ids:
            edge = self.graph.edges.get(edge_id)
            if edge and edge.state != NodeState.SKIPPED:
                all_skipped = False
                break

        # Only if ALL incoming edges are skipped, skip this node
        if all_skipped:
            # Mark the node itself as skipped
            node = self.graph.nodes.get(node_id)
            if node and node.state == NodeState.UNKNOWN:
                node.state = NodeState.SKIPPED

            # Mark all outgoing edges from this node as skipped
            outgoing_edge_ids = self.graph.out_edges.get(node_id, [])
            for edge_id in outgoing_edge_ids:
                edge = self.graph.edges.get(edge_id)
                if edge and edge.state == NodeState.UNKNOWN:
                    edge.state = NodeState.SKIPPED
                    # Recursively check descendants
                    self._mark_node_and_descendants_skipped(edge.head)

    def _is_node_ready(self, node_id: str) -> bool:
        """
        Check if a node is ready to be executed.

        A node is ready when all its incoming edges from taken branches have been satisfied.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node is ready for execution
        """
        # Check if node is already skipped
        node = self.graph.nodes.get(node_id)
        if node and node.state == NodeState.SKIPPED:
            return False

        # Get all incoming edges to this node
        incoming_edge_ids = self.graph.in_edges.get(node_id, [])

        # If no incoming edges, node is always ready
        if not incoming_edge_ids:
            return True

        has_unknown = False
        has_taken = False

        for edge_id in incoming_edge_ids:
            edge = self.graph.edges.get(edge_id)
            if edge:
                if edge.state == NodeState.UNKNOWN:
                    has_unknown = True
                elif edge.state == NodeState.TAKEN:
                    has_taken = True

        # Node is ready if no unknown edges and at least one taken edge
        return not has_unknown and has_taken

    def _should_complete_execution(self) -> bool:
        """
        Check if execution should be considered complete.

        Returns:
            True if execution should complete
        """
        # Complete if:
        # 1. Ready queue is empty (no nodes waiting to be executed)
        # 2. Event queue is empty (no events to process)
        # 3. No nodes are currently being executed
        with self._executing_nodes_lock:
            no_executing_nodes = len(self._executing_nodes) == 0

        return self.ready_queue.empty() and self.event_queue.empty() and no_executing_nodes

    def _event_generator(self) -> Generator[GraphEngineEvent, None, None]:
        """
        Generator that yields events as they're collected.

        Yields:
            GraphEngineEvent instances as they're processed
        """
        yielded_count = 0

        while not self._execution_complete.is_set() or yielded_count < len(self._collected_events):
            with self._event_collector_lock:
                # Yield any new events
                while yielded_count < len(self._collected_events):
                    yield self._collected_events[yielded_count]
                    yielded_count += 1

            # Small sleep to avoid busy waiting
            if not self._execution_complete.is_set():
                time.sleep(0.001)
