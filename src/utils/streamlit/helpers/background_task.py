import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import streamlit as st

# Type variables for generic task input and result
T = TypeVar("T")
R = TypeVar("R")


class TaskStatus(StrEnum):
    """Status of a background task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class BackgroundTask(Generic[T, R]):
    """Represents a background task that can be executed in a separate thread."""

    # Task function and input
    name: str
    func: Callable[[T, threading.Event], R]
    input_data: T
    cancellation_event: threading.Event

    # Task metadata
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Task results
    result: Optional[R] = None
    error: Optional[Exception] = None

    # Thread management
    thread: Optional[threading.Thread] = None

    def start(self) -> "BackgroundTask[T, R]":
        """Start the task in a background thread."""
        if self.status != TaskStatus.PENDING:
            return self

        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

        # Create a more descriptive thread name
        self.thread = threading.Thread(target=self._run_task, daemon=True, name=f"bgt-{self.name}")
        self.thread.start()
        return self

    def _run_task(self) -> None:
        """Internal method to execute the task function and handle results/errors."""
        try:
            if self.cancellation_event.is_set():
                self.status = TaskStatus.CANCELLED
                return

            self.result = self.func(self.input_data, self.cancellation_event)
            self.status = TaskStatus.COMPLETED
        except Exception as e:
            self.error = e
            self.status = TaskStatus.ERROR
        finally:
            if self.cancellation_event.is_set():
                self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()

    def is_done(self) -> bool:
        """Check if the task is no longer running."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR)

    def get_result(self) -> Optional[R]:
        """Get the task result, or None if not available."""
        return self.result

    def get_error(self) -> Optional[Exception]:
        """Get the task error, or None if no error occurred."""
        return self.error

    def get_duration(self) -> Optional[float]:
        """Get the task duration in seconds, or None if not completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def __hash__(self) -> int:
        """Make task hashable by its ID."""
        return hash(self.task_id)


@dataclass
class TasksManager(Generic[T, R]):
    """Manages multiple background tasks."""

    tasks: Dict[str, BackgroundTask[T, R]] = field(default_factory=dict)
    cancellation_event: threading.Event = field(default_factory=threading.Event)

    def create_and_start_task(
        self, name: str, func: Callable[[T, threading.Event], R], input_data: T
    ) -> BackgroundTask[T, R]:
        """Create and immediately start a new task."""
        task = BackgroundTask(name, func, input_data, self.cancellation_event)
        self.tasks[task.task_id] = task
        task.start()
        return task

    def get_task(self, task_id: str) -> Optional[BackgroundTask[T, R]]:
        """Get a task by its ID."""
        return self.tasks.get(task_id)

    def cancel_all_tasks(self) -> None:
        """Cancel all tasks managed by this manager."""
        if not self.cancellation_event.is_set():
            self.cancellation_event.set()

    def get_task_status_summary(self) -> Dict[TaskStatus, int]:
        """Get a summary of task statuses."""
        summary = {status: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status] += 1
        return summary

    def get_total_task_count(self) -> int:
        """Get the total number of tasks."""
        return len(self.tasks)

    def get_progress_percentage(self) -> float:
        """Get overall progress as a percentage (completed/total)."""
        total = len(self.tasks)
        if total == 0:
            return 0.0

        completed = sum(
            1
            for task in self.tasks.values()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR)
        )
        return (completed / total) * 100

    @property
    def status(self) -> TaskStatus:
        """Get the status of the tasks manager."""
        if self.cancellation_event.is_set():
            return TaskStatus.CANCELLED
        if any(task.status == TaskStatus.RUNNING for task in self.tasks.values()):
            return TaskStatus.RUNNING
        return TaskStatus.COMPLETED


def show_task_status(task: BackgroundTask, show_progress: bool = True) -> None:
    """Display a task's status in the Streamlit UI."""
    status_colors = {
        TaskStatus.PENDING: "blue",
        TaskStatus.RUNNING: "blue",
        TaskStatus.COMPLETED: "green",
        TaskStatus.CANCELLED: "orange",
        TaskStatus.ERROR: "red",
    }

    # Show task name and status
    task_name = task.name or task.task_id
    st.write(f"Task: **{task_name}** - :{status_colors[task.status]}[{task.status}]")

    # Show progress bar for running tasks
    if show_progress and task.status == TaskStatus.RUNNING:
        st.progress(0.5)

    # Show duration for completed tasks
    if task.is_done() and task.get_duration() is not None:
        st.write(f"Duration: {task.get_duration():.2f} seconds")

    # Show error if any
    if task.status == TaskStatus.ERROR and task.error:
        st.error(f"Error: {str(task.error)}")


TManager = TypeVar("TManager", bound=TasksManager)


def show_tasks_manager_summary(
    get_manager: Callable[[], TManager],
    auto_start: bool = False,
    run_every: float = 4,
    on_cancel_click: Optional[Callable[[], Any]] = None,
    on_start_click: Optional[Callable[[], Any]] = None,
    get_additional_metrics: Callable[[TManager], tuple[Dict[str, str], Optional[float]]] = lambda _: ({}, None),
    button_keys_prefix: str = "",
) -> None:
    # Define the main display function

    def display_summary_and_controls(with_rerun=True):
        manager = get_manager()
        summary = manager.get_task_status_summary()
        total = manager.get_total_task_count()

        if total == 0:
            st.info("No tasks created yet.")
            return
        # Use custom progress if provided, otherwise get from manager
        additional_metrics, progress = get_additional_metrics(manager)
        progress = progress if progress is not None else manager.get_progress_percentage() / 100
        # Create summary text
        summary_text = [f"Total tasks: {total} | {progress * 100:.1f}% complete"]

        for status, count in summary.items():
            if count > 0:
                summary_text.append(f"{status}: {count}")

        # Add any additional metrics
        for name, value in additional_metrics.items():
            summary_text.append(f"{name}: {value}")

        # Show summary text
        text = " | ".join(summary_text)

        # Show progress bar and buttons in the same row
        cols = st.columns([7, 1])  # One large column for progress bar, three small columns for buttons

        # Display progress in the first column
        cols[0].progress(progress, text)

        status = manager.status
        if status == TaskStatus.RUNNING:
            # Show Cancel button when running
            # Default to manager.cancel_all_tasks if not provided
            cancel_handler = on_cancel_click if on_cancel_click is not None else manager.cancel_all_tasks

            if cols[1].button("🛑 Stop", key=f"{button_keys_prefix}cancel_tasks"):
                cancel_handler()
        else:
            if status == TaskStatus.PENDING and auto_start:
                assert on_start_click is not None
                on_start_click()
            # Show Start/Resume button if auto_start is True
            if on_start_click is not None:
                button_label = "▶️ Resume" if status == TaskStatus.CANCELLED else "▶️ Start"
                button_key = f"{button_keys_prefix}{'resume' if status == TaskStatus.CANCELLED else 'start'}_tasks"

                if cols[1].button(button_label, key=button_key):
                    on_start_click()
        if status == TaskStatus.COMPLETED:
            st.success("All tasks completed successfully.")
            if with_rerun:
                st.rerun()

    manager = get_manager()

    func = display_summary_and_controls
    if manager.status == TaskStatus.RUNNING:
        func = st.fragment(run_every=run_every)(func)
    func(with_rerun=False)
