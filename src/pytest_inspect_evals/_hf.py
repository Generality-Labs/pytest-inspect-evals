"""Hugging Face handling for tests marked huggingface."""

import os
from importlib.util import find_spec

import pytest


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


def hf_apply_collection_markers(items: list[pytest.Item]) -> None:
    hf_items = [item for item in items if item.get_closest_marker("huggingface") is not None]

    if not hf_items:
        return

    if not os.environ.get("HF_TOKEN", "").strip():
        for item in hf_items:
            item.add_marker(pytest.mark.skip(reason=f"HF_TOKEN not set ({item.name})"))

    else:
        for item in hf_items:
            item.add_marker(pytest.mark.flaky(reruns=2, reruns_delay=60.0))


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
    if _is_hf_gated_dataset_failure(item, call, report):
        report.outcome = "skipped"
        line_number = item.reportinfo()[1] or 0
        report.longrepr = (
            str(item.fspath),
            line_number,
            f"Skipped: Gated dataset ({item.name})",
        )


def is_gated_dataset_exception(exc: BaseException) -> bool:
    exc_name = type(exc).__name__
    return exc_name == "GatedRepoError" or (
        exc_name == "DatasetNotFoundError" and "gated dataset" in str(exc).lower()
    )
