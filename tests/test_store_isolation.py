import pytest

STORE_TESTS = """
from inspect_ai.util import store

# Captured at import time, before any fixture runs: the process-global default
# store that store() falls back to outside of eval().
PROCESS_DEFAULT_STORE = store()
SENTINEL = "sentinel"


def test_writes_to_its_own_store():
    assert store() is not PROCESS_DEFAULT_STORE
    assert len(store().keys()) == 0
    store().set(SENTINEL, "written")


def test_does_not_see_earlier_writes():
    assert SENTINEL not in store()
    assert SENTINEL not in PROCESS_DEFAULT_STORE
"""

RESTORE_CONFTEST = """
from inspect_ai.util import store

DEFAULT = store()


def pytest_sessionfinish(session, exitstatus):
    # Every test's store has been torn down, so the default store is active again.
    assert store() is DEFAULT
    print("store restored")
"""


def test_each_test_gets_a_fresh_store(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(STORE_TESTS)
    result = pytester.runpytest("-p", "no:randomly")
    result.assert_outcomes(passed=2)


def test_previous_store_is_restored_after_each_test(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(RESTORE_CONFTEST)
    pytester.makepyfile(STORE_TESTS)
    result = pytester.runpytest("-s", "-p", "no:randomly")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*store restored*"])
