"""pytest entry point: test gates, Hugging Face handling, Windows skips and shared fixtures."""

import pytest

from pytest_inspect_evals.gates import GATES, skip_if_marker_present

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
