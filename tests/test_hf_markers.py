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
    assert flaky.kwargs == {
        "reruns": 2,
        "reruns_delay": 60.0,
        "rerun_except": _hf.GATED_RERUN_EXCEPT,
    }


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


def test_strict_without_token_errors(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest("--require-hf-access")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*REQUIRE_HF_ACCESS*HF_TOKEN*"])


def test_strict_env_var_without_token_errors(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("REQUIRE_HF_ACCESS", "1")
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*REQUIRE_HF_ACCESS*HF_TOKEN*"])


def test_strict_ini_without_token_errors(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makeini("[pytest]\nrequire_hf_access = true\n")
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*REQUIRE_HF_ACCESS*HF_TOKEN*"])


def test_strict_env_zero_overrides_flag(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("REQUIRE_HF_ACCESS", "0")
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest("--require-hf-access")
    result.assert_outcomes(skipped=1)


def test_strict_without_token_and_no_hf_tests_runs(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile("def test_plain():\n    pass\n")
    result = pytester.runpytest("--require-hf-access")
    result.assert_outcomes(passed=1)


class _FakeError(Exception):
    def __init__(self, msg: str, repo_id: object = None) -> None:
        super().__init__(msg)
        if repo_id is not None:
            self.repo_id = repo_id


def test_gated_repo_id_from_attribute() -> None:
    assert _hf.gated_repo_id(_FakeError("401", repo_id="org/a")) == "org/a"


def test_gated_repo_id_from_dataset_not_found_message() -> None:
    exc = _FakeError("Dataset 'org/b' is a gated dataset on the Hub. Visit https://x for more.")
    assert _hf.gated_repo_id(exc) == "org/b"


def test_gated_repo_id_unknown() -> None:
    assert _hf.gated_repo_id(_FakeError("401")) is None
    assert _hf.gated_repo_id(_FakeError("401", repo_id="")) is None


def test_strict_gated_failures_fail(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_FAILURE)
    result = pytester.runpytest("-p", "no:rerunfailures", "--require-hf-access")
    result.assert_outcomes(failed=4)
    result.stdout.fnmatch_lines(["*Hugging Face access denied*"])


def test_strict_gated_error_without_repo_id(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_FAILURE)
    result = pytester.runpytest("-p", "no:rerunfailures", "--require-hf-access")
    result.stdout.fnmatch_lines(["*test_gated_repo*"])
    reports = pytester.inline_run("-p", "no:rerunfailures", "--require-hf-access").getreports(
        "pytest_runtest_logreport"
    )
    [report] = [r for r in reports if r.when == "call" and r.nodeid.endswith("test_gated_repo")]
    assert ("hf_access_denied", "") in report.user_properties


GATED_WITH_REPO_ID = """
import pytest

class GatedRepoError(Exception):
    repo_id = "org/a"

@pytest.mark.huggingface
def test_a():
    raise GatedRepoError("401")

@pytest.mark.huggingface
def test_b():
    raise GatedRepoError("401")
"""


def test_strict_summary_lists_terms_urls(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_WITH_REPO_ID)
    result = pytester.runpytest("-p", "no:rerunfailures", "--require-hf-access")
    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(
        ["*= Hugging Face access denied =*", "*https://huggingface.co/datasets/org/a*"]
    )
    assert result.stdout.str().count("https://huggingface.co/datasets/org/a") >= 1


def test_strict_summary_under_xdist(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("xdist")
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_WITH_REPO_ID)
    result = pytester.runpytest("-p", "no:rerunfailures", "--require-hf-access", "-n", "2")
    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(
        ["*= Hugging Face access denied =*", "*https://huggingface.co/datasets/org/a*"]
    )


def test_gated_errors_not_rerun(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makeconftest("import time\ntime.sleep = lambda seconds: None\n")
    pytester.makepyfile(GATED_FAILURE)
    result = pytester.runpytest()
    result.assert_outcomes(skipped=2, failed=2)
    assert result.parseoutcomes()["rerun"] == 2
