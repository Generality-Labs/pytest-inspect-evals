import time

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


def _set_token(monkeypatch: pytest.MonkeyPatch, token: str | None) -> None:
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)


NO_TOKEN_NOTICE = "HF_TOKEN is not set"


@pytest.mark.parametrize("token", [None, "", "   "])
def test_runs_with_a_notice_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    _set_token(monkeypatch, token)
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*= Hugging Face token =*",
            "HF_TOKEN is not set: 1 huggingface test(s) ran without authentication.*",
        ]
    )


def test_notice_counts_the_tests_that_ran_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pytester.makepyfile(HF_TEST + "\n\n@pytest.mark.huggingface\ndef test_hf_too():\n    pass\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["HF_TOKEN is not set: 2 huggingface test(s) ran*"])


def test_one_notice_under_xdist_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("xdist")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pytester.makepyfile(HF_TEST + "\n\n@pytest.mark.huggingface\ndef test_hf_too():\n    pass\n")
    result = pytester.runpytest("-n", "2")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["HF_TOKEN is not set: 2 huggingface test(s) ran*"])
    assert result.stdout.str().count(NO_TOKEN_NOTICE) == 1


def test_runs_without_hf_token_when_warnings_are_errors(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest("-W", "error")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["HF_TOKEN is not set: 1 huggingface test(s) ran*"])


def test_report_is_tagged_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pytester.makepyfile(HF_TEST + "\n\ndef test_plain():\n    pass\n")
    reports = pytester.inline_run().getreports("pytest_runtest_logreport")
    tagged = {r.nodeid.rsplit("::", 1)[1]: r.user_properties for r in reports if r.when == "setup"}
    assert tagged == {"test_hf": [("hf_no_token", "")], "test_plain": []}


def test_no_notice_with_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.no_fnmatch_line(f"*{NO_TOKEN_NOTICE}*")


def test_no_notice_when_huggingface_tests_are_deselected(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pytester.makepyfile(HF_TEST + "\n\ndef test_plain():\n    pass\n")
    result = pytester.runpytest("-m", "not huggingface")
    result.assert_outcomes(passed=1, deselected=1)
    result.stdout.no_fnmatch_line(f"*{NO_TOKEN_NOTICE}*")


def test_no_notice_when_a_gate_skips_the_huggingface_tests(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("RUN_DATASET_DOWNLOAD_TESTS", raising=False)
    pytester.makepyfile(
        "import pytest\n\n@pytest.mark.huggingface\n@pytest.mark.dataset_download\n"
        "def test_hf_download():\n    pass\n"
    )
    result = pytester.runpytest()
    result.assert_outcomes(skipped=1)
    result.stdout.no_fnmatch_line(f"*{NO_TOKEN_NOTICE}*")


@pytest.mark.parametrize("token", [None, "dummy"])
def test_marked_flaky_with_and_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    _set_token(monkeypatch, token)
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


def test_gated_failures_become_skips_without_hf_token(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
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
    pytester.runpytest().assert_outcomes(passed=1)


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


def test_strict_without_token_under_xdist_fails_hf_tests(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("xdist")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("REQUIRE_HF_ACCESS", "1")
    pytester.makepyfile(HF_TEST + "\n\ndef test_plain():\n    pass\n")
    result = pytester.runpytest("-n", "2")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*REQUIRE_HF_ACCESS*HF_TOKEN*"])
    assert "INTERNALERROR" not in result.stdout.str() + result.stderr.str()
    result.stdout.no_fnmatch_line("*ran without authentication*")


def test_strict_env_zero_overrides_flag(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("REQUIRE_HF_ACCESS", "0")
    pytester.makepyfile(HF_TEST)
    result = pytester.runpytest("--require-hf-access")
    result.assert_outcomes(passed=1)


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
    result.stdout.fnmatch_lines(["*::test_gated_repo (dataset unknown)"])
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


def test_gated_skips_with_reruns_enabled(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In default mode gated failures become skips before pytest-rerunfailures sees
    # them, so this does not test rerun_except. See the strict-mode test below.
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    # Patch time.sleep to avoid the 60 second delay between reruns.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    pytester.makepyfile(GATED_FAILURE)
    result = pytester.runpytest()
    result.assert_outcomes(skipped=2, failed=2)
    assert result.parseoutcomes()["rerun"] == 2


def test_strict_gated_errors_not_rerun(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    # Patch time.sleep to avoid the 60 second delay between reruns.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    pytester.makepyfile(GATED_FAILURE)
    result = pytester.runpytest("--require-hf-access")
    # Only test_real_failure is rerun. The unmarked test has no flaky marker.
    result.assert_outcomes(failed=4)
    assert result.parseoutcomes()["rerun"] == 2


GATED_FIXTURE = """
import pytest

class GatedRepoError(Exception):
    repo_id = "org/a"

@pytest.fixture(scope="module")
def gated_info():
    raise GatedRepoError("401")

@pytest.mark.huggingface
def test_first(gated_info):
    pass

@pytest.mark.huggingface
def test_second(gated_info):
    pass

def test_unmarked_still_errors(gated_info):
    pass
"""


def test_gated_fixture_errors_become_skips(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_FIXTURE)
    result = pytester.runpytest("-p", "no:rerunfailures", "-rs")
    result.assert_outcomes(skipped=2, errors=1)
    result.stdout.fnmatch_lines(["*Gated dataset (test_first)*", "*Gated dataset (test_second)*"])


def test_gated_fixture_errors_without_hf_token_are_in_the_notice(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_FIXTURE)
    result = pytester.runpytest("-p", "no:rerunfailures", "-rs")
    result.assert_outcomes(skipped=2, errors=1)
    result.stdout.fnmatch_lines(["HF_TOKEN is not set: 2 huggingface test(s) ran*"])


def test_strict_gated_fixture_errors_are_tagged(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_FIXTURE)
    reprec = pytester.inline_run("-p", "no:rerunfailures", "--require-hf-access")
    setup_errors = {
        report.nodeid.rsplit("::", 1)[1]: report.user_properties
        for report in reprec.getreports("pytest_runtest_logreport")
        if report.when == "setup" and report.failed
    }
    assert setup_errors == {
        "test_first": [("hf_access_denied", "org/a")],
        "test_second": [("hf_access_denied", "org/a")],
        "test_unmarked_still_errors": [],
    }


def test_strict_gated_fixture_errors_in_summary_under_xdist(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("xdist")
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    pytester.makepyfile(GATED_FIXTURE)
    result = pytester.runpytest("-p", "no:rerunfailures", "--require-hf-access", "-n", "2")
    result.assert_outcomes(errors=3)
    result.stdout.fnmatch_lines(
        ["*= Hugging Face access denied =*", "*https://huggingface.co/datasets/org/a*"]
    )


def test_strict_gated_fixture_errors_not_rerun(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.delenv("REQUIRE_HF_ACCESS", raising=False)
    # Patch time.sleep to avoid the 60 second delay between reruns.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    pytester.makepyfile(GATED_FIXTURE)
    result = pytester.runpytest("--require-hf-access")
    result.assert_outcomes(errors=3)
    assert result.parseoutcomes().get("rerun", 0) == 0
