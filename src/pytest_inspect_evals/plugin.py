"""pytest entry point: test gates, Hugging Face handling, Windows skips and shared fixtures."""

from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from inspect_ai.model import Model
from inspect_ai.model._model import init_model_roles  # no public API available
from inspect_ai.util import Store, store
from inspect_ai.util._store import init_subtask_store  # no public API available

from pytest_inspect_evals._hf import (
    HF_ACCESS_DENIED_KEY,
    HF_NO_TOKEN_KEY,
    HF_TERMS_URL,
    REQUIRE_HF_ACCESS_FLAG,
    REQUIRE_HF_ACCESS_INI,
    hf_access_required,
    hf_apply_collection_markers,
    hf_configure_logging,
    hf_convert_gated_failure_to_skip,
    hf_disable_tokenizer_parallelism,
    hf_fail_if_token_missing,
    hf_tag_run_without_token,
)
from pytest_inspect_evals._windows import windows_skip_unsupported_tests
from pytest_inspect_evals.gates import GATES, skip_if_marker_present
from pytest_inspect_evals.sandbox import mock_docker_sandbox

__all__ = [
    "hf_access_required",
    "isolated_inspect_store",
    "mock_docker_sandbox",
    "model_roles",
    "set_model_roles",
]

FIXED_MARKERS: tuple[tuple[str, str], ...] = (
    (
        "huggingface",
        "marks tests that use the Hugging Face Hub; retried twice; without HF_TOKEN they run "
        "with a notice in the summary, or fail in strict mode (--require-hf-access)",
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
    group.addoption(
        REQUIRE_HF_ACCESS_FLAG,
        action="store_true",
        default=False,
        help="fail on gated-dataset errors and error when HF_TOKEN is missing "
        "(or set REQUIRE_HF_ACCESS=1)",
    )
    parser.addini(
        REQUIRE_HF_ACCESS_INI,
        type="bool",
        default=False,
        help="require Hugging Face access for huggingface tests by default in this repo",
    )


def pytest_configure(config: pytest.Config) -> None:
    for gate in GATES:
        config.addinivalue_line("markers", f"{gate.marker}: {gate.description}")
    for name, description in FIXED_MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")
    config.pluginmanager.register(_HfSummary(), "inspect_evals_hf_summary")
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
    hf_apply_collection_markers(config, items)


def pytest_runtest_setup(item: pytest.Item) -> None:
    hf_fail_if_token_missing(item)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    # Tag first: a gated-dataset error in a fixture turns the setup report into a skip.
    hf_tag_run_without_token(item, report)
    hf_convert_gated_failure_to_skip(item, call, report)
    return report


class _HfSummary:
    """Print the Hugging Face sections of the terminal summary.

    Reports are seen in pytest_runtest_logreport, which also runs on the xdist
    controller, so tests that ran on workers are included.
    """

    def __init__(self) -> None:
        self.denied: dict[str, str] = {}
        self.ran_without_token: set[str] = set()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        for key, value in report.user_properties:
            if key == HF_ACCESS_DENIED_KEY:
                self.denied[report.nodeid] = str(value)
            elif key == HF_NO_TOKEN_KEY:
                self.ran_without_token.add(report.nodeid)

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if self.ran_without_token:
            terminalreporter.section("Hugging Face token", yellow=True)
            terminalreporter.write_line(
                f"HF_TOKEN is not set: {len(self.ran_without_token)} huggingface test(s) ran "
                "without authentication. A gated-dataset error is reported as a skip, a test "
                "that needs a private dataset fails, and the Hub applies lower rate limits to "
                "requests without a token.",
                yellow=True,
            )
        if not self.denied:
            return
        terminalreporter.section("Hugging Face access denied")
        lines = {
            HF_TERMS_URL.format(repo_id=repo_id) if repo_id else f"{node_id} (dataset unknown)"
            for node_id, repo_id in self.denied.items()
        }
        for line in sorted(lines):
            terminalreporter.write_line(line)


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


@pytest.fixture(autouse=True)
def isolated_inspect_store() -> Iterator[None]:
    """Give every test its own inspect_ai `Store`.

    Outside of `eval()`, `store()` and `store_as()` resolve to a single
    process-global default `Store`, so anything a test writes there (for example
    a `StoreModel`-backed tool database) is still present for every later test in
    the same pytest process. inspect_ai snapshots and diffs the active store on
    every solver step, so leaked data also slows down every later test that runs
    a solver chain or agent directly.
    """
    previous = store()
    init_subtask_store(Store())
    try:
        yield
    finally:
        init_subtask_store(previous)
