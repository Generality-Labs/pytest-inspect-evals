"""Helpers and fixtures for testing tools and scorers that use a sandbox."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
