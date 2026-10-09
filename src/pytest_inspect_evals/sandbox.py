"""Utilities for testing custom tools that run inside Docker sandboxes.

This module provides reusable helpers for the sandbox tool testing pattern.
The pattern uses a custom @solver to exercise
tools inside a real Docker container, then checks results via eval().

Usage:
    from pytest_inspect_evals.sandbox import (
        MockExecResult,
        create_sandbox_tool_task,
        assert_sandbox_test_passed,
    )

Example (sandbox tool test):
    @solver
    def _my_tool_test_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            tool = my_tool()
            result = await tool("input")
            assert "expected" in result
            state.metadata["test_passed"] = True
            return state
        return solve

    @pytest.mark.docker
    @pytest.mark.slow(30)
    def test_my_tool_in_sandbox() -> None:
        compose_path = Path(__file__).parent.parent.parent / (
            "src/inspect_evals/my_eval/data/compose.yaml"
        )
        task = create_sandbox_tool_task(_my_tool_test_solver(), compose_path)
        [log] = eval(task, model=get_model("mockllm/model"))
        assert_sandbox_test_passed(log)

Example (mocking sandbox exec):
    mock_result = MockExecResult(success=True, stdout="file content")
    with patch(
        "inspect_evals.my_eval.my_module.execute_command",
        new=AsyncMock(return_value=mock_result),
    ):
        tool = my_tool()
        result = await tool("input")
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import Solver
from inspect_ai.util import SandboxEnvironmentSpec


class MockExecResult:
    """Mock for sandbox exec() results.

    Use this when mocking sandbox().exec() calls in unit tests
    that don't need a real Docker container.

    Args:
        success: Whether the command succeeded.
        stdout: Standard output from the command.
        stderr: Standard error from the command.
        returncode: Exit code (defaults to 0 if success, 1 if not).
    """

    def __init__(
        self,
        success: bool,
        stdout: str,
        stderr: str = "",
        returncode: int | None = None,
    ) -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode if returncode is not None else (0 if success else 1)


def create_sandbox_tool_task(
    test_solver: Solver,
    compose_path: str | Path,
    *,
    sample_id: str = "tool_test",
    sample_input: str = "Test tools",
) -> Task:
    """Create a minimal Task for testing tools in a real Docker sandbox.

    This creates a Task with a single sample that uses the specified
    compose.yaml for its sandbox environment. The test_solver should
    exercise the tool(s) being tested and set
    ``state.metadata["test_passed"] = True`` on success.

    Args:
        test_solver: A @solver that exercises the tool and sets
            state.metadata["test_passed"] = True on success.
        compose_path: Path to the eval's compose.yaml file.
        sample_id: ID for the test sample (default: "tool_test").
        sample_input: Input text for the test sample (default: "Test tools").

    Returns:
        A Task ready to be passed to eval().
    """
    return Task(
        dataset=[
            Sample(
                id=sample_id,
                input=sample_input,
                sandbox=SandboxEnvironmentSpec(
                    type="docker",
                    config=str(compose_path),
                ),
            )
        ],
        solver=[test_solver],
        scorer=None,
    )


def assert_sandbox_test_passed(log: EvalLog) -> None:
    """Assert that a sandbox tool test completed successfully.

    Checks both that the eval task succeeded and that the test solver
    set the "test_passed" metadata flag.

    Args:
        log: The EvalLog from running the sandbox tool test.

    Raises:
        AssertionError: If the task failed or test assertions failed.
    """
    assert log.status == "success", f"Task failed: {log.error}"
    assert log.samples, "No samples in eval log"
    assert log.samples[0].metadata.get("test_passed"), "Test assertions failed"


@pytest.fixture
def mock_docker_sandbox() -> Iterator[None]:
    """Mock DockerSandboxEnvironment lifecycle methods to prevent Docker operations.

    The eval framework calls task_init / sample_init before the scorer runs,
    which tries to build and start the Docker container. This fixture
    short-circuits the lifecycle classmethods so the eval can proceed without
    any real Docker infrastructure.
    """
    # no public API available
    from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

    sandbox_env = MagicMock()
    # sandbox_connections() awaits env.connection(); MagicMock is not awaitable so
    # we use AsyncMock with NotImplementedError, which sandbox_connections catches.
    sandbox_env.connection = AsyncMock(side_effect=NotImplementedError)

    with (
        patch.object(DockerSandboxEnvironment, "task_init", AsyncMock(return_value=None)),
        patch.object(DockerSandboxEnvironment, "task_init_environment", AsyncMock(return_value={})),
        patch.object(
            DockerSandboxEnvironment,
            "sample_init",
            AsyncMock(return_value={"default": sandbox_env}),
        ),
        patch.object(DockerSandboxEnvironment, "sample_cleanup", AsyncMock(return_value=None)),
        patch.object(DockerSandboxEnvironment, "task_cleanup", AsyncMock(return_value=None)),
    ):
        yield
