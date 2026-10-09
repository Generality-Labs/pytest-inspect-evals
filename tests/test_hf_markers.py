import pytest

from pytest_inspect_evals import _hf

HF_TEST = """
import pytest

@pytest.mark.huggingface
def test_hf():
    pass
"""

GATED_FAILURE = """
import pytest

class GatedRepoError(Exception):
    pass

class DatasetNotFoundError(Exception):
    pass

@pytest.mark.huggingface
def test_gated_repo():
    raise GatedRepoError("401")

@pytest.mark.huggingface
def test_gated_dataset():
    raise DatasetNotFoundError("Dataset 'x' is a gated dataset on the Hub.")

@pytest.mark.huggingface
def test_real_failure():
    raise ValueError("boom")

def test_unmarked_gated_error_still_fails():
    raise GatedRepoError("401")
"""


@pytest.mark.parametrize("token", [None, "", "   "])
def test_skipped_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest("-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*HF_TOKEN not set*"])


def test_marked_flaky_with_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    pytester.makepyfile(HF_TEST)
    items, _ = pytester.inline_genitems()
    [flaky] = list(items[0].iter_markers("flaky"))
    assert flaky.kwargs == {"reruns": 2, "reruns_delay": 60.0}


def test_gated_failures_become_skips(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    pytester.makepyfile(GATED_FAILURE)
    result = pytester.runpytest("-p", "no:rerunfailures", "-rs")
    result.assert_outcomes(skipped=2, failed=2)
    result.stdout.fnmatch_lines(
        ["*Gated dataset (test_gated_repo)*", "*Gated dataset (test_gated_dataset)*"]
    )


def test_hf_logging_is_a_no_op_without_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_hf, "find_spec", lambda name: None)
    _hf.hf_configure_logging()


def test_plugin_loads_without_datasets(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_hf, "find_spec", lambda name: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pytester.makepyfile(HF_TEST)
    pytester.runpytest().assert_outcomes(skipped=1)
