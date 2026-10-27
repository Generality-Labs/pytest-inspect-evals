"""Hugging Face handling for tests marked huggingface."""

import os
import re
from importlib.util import find_spec

import pytest

from pytest_inspect_evals.gates import truthy

REQUIRE_HF_ACCESS_FLAG = "--require-hf-access"
REQUIRE_HF_ACCESS_ENV = "REQUIRE_HF_ACCESS"
REQUIRE_HF_ACCESS_INI = "require_hf_access"

HF_ACCESS_DENIED_KEY = "hf_access_denied"
HF_TERMS_URL = "https://huggingface.co/datasets/{repo_id}"
_GATED_DATASET_MESSAGE = re.compile(r"Dataset '([^']+)' is a gated dataset", re.IGNORECASE)

# pytest-rerunfailures applies re.search to "<ExceptionType>: <message>".
GATED_RERUN_EXCEPT = ["GatedRepoError", "(?is)DatasetNotFoundError: .*gated dataset"]


def hf_disable_tokenizer_parallelism() -> None:
    """Disable HF tokenizers parallelism to avoid segfaults."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def hf_configure_logging() -> None:
    """Raise datasets and huggingface_hub logging to INFO when datasets is installed."""
    if find_spec("datasets") is None:
        return
    import datasets
    import huggingface_hub

    datasets.logging.set_verbosity_info()
    huggingface_hub.logging.set_verbosity_info()


def hf_access_required(config: pytest.Config) -> bool:
    """Return whether strict Hugging Face mode is on.

    Priority: env var (when non-empty), then CLI flag, then ini option.
    """
    env_val = os.environ.get(REQUIRE_HF_ACCESS_ENV, "").strip()
    if env_val:
        return truthy(env_val)
    if config.getoption("require_hf_access"):
        return True
    return bool(config.getini(REQUIRE_HF_ACCESS_INI))


def hf_apply_collection_markers(config: pytest.Config, items: list[pytest.Item]) -> None:
    hf_items = [item for item in items if item.get_closest_marker("huggingface") is not None]

    if not hf_items:
        return

    if not os.environ.get("HF_TOKEN", "").strip():
        if hf_access_required(config):
            raise pytest.UsageError(
                f"{REQUIRE_HF_ACCESS_ENV} is on and {len(hf_items)} huggingface test(s) "
                "were collected, but HF_TOKEN is not set. Set HF_TOKEN, or turn strict "
                "mode off."
            )
        for item in hf_items:
            item.add_marker(pytest.mark.skip(reason=f"HF_TOKEN not set ({item.name})"))

    else:
        for item in hf_items:
            item.add_marker(
                pytest.mark.flaky(reruns=2, reruns_delay=60.0, rerun_except=GATED_RERUN_EXCEPT)
            )


def _is_hf_gated_dataset_failure(
    item: pytest.Item, call: pytest.CallInfo, report: pytest.TestReport
) -> bool:
    return (
        report.when == "call"
        and report.failed
        and item.get_closest_marker("huggingface") is not None
        and bool(call.excinfo)
        and is_gated_dataset_exception(call.excinfo.value)  # type: ignore
    )


def hf_convert_gated_failure_to_skip(
    item: pytest.Item, call: pytest.CallInfo, report: pytest.TestReport
) -> None:
    if not _is_hf_gated_dataset_failure(item, call, report):
        return
    if hf_access_required(item.config):
        repo_id = gated_repo_id(call.excinfo.value) or ""  # type: ignore
        report.user_properties.append((HF_ACCESS_DENIED_KEY, repo_id))
        what = f"dataset {repo_id}" if repo_id else f"a dataset ({item.name})"
        hint = (
            f"accept the terms at {HF_TERMS_URL.format(repo_id=repo_id)}"
            if repo_id
            else ("accept the terms on the dataset page")
        )
        report.sections.append(
            ("Hugging Face access", f"Hugging Face access denied for {what}: {hint}")
        )
        return
    report.outcome = "skipped"
    line_number = item.reportinfo()[1] or 0
    report.longrepr = (
        str(item.fspath),
        line_number,
        f"Skipped: Gated dataset ({item.name})",
    )


def gated_repo_id(exc: BaseException) -> str | None:
    """Return the repo id of a gated dataset error, or None when it cannot be found."""
    repo_id = getattr(exc, "repo_id", None)
    if isinstance(repo_id, str) and repo_id:
        return repo_id
    match = _GATED_DATASET_MESSAGE.search(str(exc))
    return match.group(1) if match else None


def is_gated_dataset_exception(exc: BaseException) -> bool:
    exc_name = type(exc).__name__
    return exc_name == "GatedRepoError" or (
        exc_name == "DatasetNotFoundError" and "gated dataset" in str(exc).lower()
    )
