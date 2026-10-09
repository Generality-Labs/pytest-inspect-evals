"""Opt-in test gates: markers whose tests are skipped unless switched on."""

import os
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

TRUTHY = frozenset({"1", "true", "yes", "on"})


def truthy(value: str | None) -> bool:
    """Return whether an env var value switches a gate on."""
    return (value or "").strip().lower() in TRUTHY


@dataclass(frozen=True)
class Gate:
    """A marker that is skipped unless an env var, CLI flag or ini option enables it."""

    marker: str
    cli_flag: str
    env_var: str
    ini_option: str
    description: str

    @property
    def reason(self) -> str:
        return (
            f"{self.marker} tests disabled (set {self.env_var}=1, pass {self.cli_flag}, "
            f"or set {self.ini_option} = true in the pytest config)"
        )


GATES: tuple[Gate, ...] = (
    Gate(
        "slow",
        "--runslow",
        "RUN_SLOW_TESTS",
        "inspect_evals_slow",
        "marks tests that are slow to run",
    ),
    Gate(
        "dataset_download",
        "--dataset-download",
        "RUN_DATASET_DOWNLOAD_TESTS",
        "inspect_evals_dataset_download",
        "marks tests that download datasets",
    ),
    Gate(
        "k8s",
        "--runk8s",
        "RUN_K8S_TESTS",
        "inspect_evals_k8s",
        "marks tests requiring k8s sandbox support",
    ),
    Gate(
        "gpu",
        "--rungpu",
        "RUN_GPU_TESTS",
        "inspect_evals_gpu",
        "marks tests that need an NVIDIA GPU in the sandbox host (Docker or k8s)",
    ),
)


def skip_if_marker_present(
    config: pytest.Config,
    items: Iterable[pytest.Item],
    marker: str,
    cli_flag: str,
    env_var: str,
    reason: str,
    *,
    default_enabled: bool = False,
    ini_option: str | None = None,
) -> None:
    """Skip tests carrying `marker` unless the gate is switched on.

    Priority: env var, then CLI flag, then ini option (when given), then
    `default_enabled`. An empty env var counts as unset.
    """
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        enabled = truthy(env_val)
    elif config.getoption(cli_flag):
        enabled = True
    elif ini_option is not None:
        enabled = bool(config.getini(ini_option))
    else:
        enabled = default_enabled
    if enabled:
        return
    skip_mark = pytest.mark.skip(reason=reason)
    for item in items:
        if marker in item.keywords:
            item.add_marker(skip_mark)
