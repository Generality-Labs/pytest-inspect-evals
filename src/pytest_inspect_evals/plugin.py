"""pytest entry point: test gates, Hugging Face handling, Windows skips and shared fixtures."""

from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from inspect_ai.model import Model
from inspect_ai.model._model import init_model_roles  # no public API available

from pytest_inspect_evals._hf import (
    hf_apply_collection_markers,
    hf_configure_logging,
    hf_convert_gated_failure_to_skip,
    hf_disable_tokenizer_parallelism,
)
from pytest_inspect_evals._windows import windows_skip_unsupported_tests
from pytest_inspect_evals.gates import GATES, skip_if_marker_present
from pytest_inspect_evals.sandbox import mock_docker_sandbox

__all__ = ["mock_docker_sandbox", "model_roles", "set_model_roles"]

FIXED_MARKERS: tuple[tuple[str, str], ...] = (
    (
        "huggingface",
        "marks tests that use the Hugging Face Hub; skipped without HF_TOKEN, otherwise retried twice",
    ),
    ("docker", "marks tests that pull or build docker images; skipped on Windows"),
    ("posix_only", "marks tests that require POSIX systems (Linux/macOS); skipped on Windows"),
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("inspect_evals", "Inspect evals test gates")
    for gate in GATES:
        group.addoption(
            gate.cli_flag,
            action="store_true",
            default=False,
            help=f"run tests marked {gate.marker} (or set {gate.env_var}=1)",
        )
        parser.addini(
            gate.ini_option,
            type="bool",
            default=False,
            help=f"run tests marked {gate.marker} by default in this repo",
        )


def pytest_configure(config: pytest.Config) -> None:
    for gate in GATES:
        config.addinivalue_line("markers", f"{gate.marker}: {gate.description}")
    for name, description in FIXED_MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")
    hf_disable_tokenizer_parallelism()
    hf_configure_logging()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for gate in GATES:
        skip_if_marker_present(
            config,
            items,
            marker=gate.marker,
            cli_flag=gate.cli_flag,
            env_var=gate.env_var,
            reason=gate.reason,
            ini_option=gate.ini_option,
        )
    windows_skip_unsupported_tests(items)
    hf_apply_collection_markers(items)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    hf_convert_gated_failure_to_skip(item, call, report)
    return report


@contextmanager
def model_roles(**roles: Model) -> Iterator[None]:
    """Set model roles for the duration of the block.

    Usage::

        with model_roles(grader=my_mock_model):
            score = await scorer_fn(state, target)
    """
    # init_model_roles takes dict[str, Model | list[Model]] and dict is
    # invariant, so the **roles dict[str, Model] needs widening to match.
    widened: dict[str, Model | list[Model]] = dict(roles)
    init_model_roles(widened)
    try:
        yield
    finally:
        init_model_roles({})


@pytest.fixture
def set_model_roles() -> Callable[..., AbstractContextManager[None]]:
    """Return a context manager that sets model roles.

    Usage::

        def test_something(set_model_roles):
            with set_model_roles(grader=my_model):
                score = scorer_fn(state, target)
    """
    return model_roles
