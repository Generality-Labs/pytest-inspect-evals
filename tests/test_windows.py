import pytest

from pytest_inspect_evals._windows import windows_skip_unsupported_tests

MARKED = """
import pytest

@pytest.mark.docker
def test_docker():
    pass

@pytest.mark.posix_only
def test_posix():
    pass

def test_plain():
    pass
"""


def _skipped(items: list[pytest.Item]) -> set[str]:
    return {item.name for item in items if any(item.iter_markers("skip"))}


def test_docker_and_posix_only_skipped_on_windows(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(MARKED)
    items, _ = pytester.inline_genitems()
    windows_skip_unsupported_tests(items, platform="win32")
    assert _skipped(items) == {"test_docker", "test_posix"}


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_nothing_skipped_elsewhere(pytester: pytest.Pytester, platform: str) -> None:
    pytester.makepyfile(MARKED)
    items, _ = pytester.inline_genitems()
    windows_skip_unsupported_tests(items, platform=platform)
    assert _skipped(items) == set()


def test_param_id_matching_a_marker_name_is_not_skipped_on_windows(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("kind", ["docker", "posix_only"])
        def test_param(kind):
            pass
        """
    )
    items, _ = pytester.inline_genitems()
    windows_skip_unsupported_tests(items, platform="win32")
    assert _skipped(items) == set()
