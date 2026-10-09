"""Skip tests that can't run on Windows."""

import sys

import pytest


def windows_skip_unsupported_tests(items: list[pytest.Item], platform: str | None = None) -> None:
    """Skip POSIX-only and Docker tests on Windows."""
    if (platform or sys.platform) != "win32":
        return

    for item in items:
        if item.get_closest_marker("posix_only") is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Skipping {item.name}: test requires POSIX system (not supported on Windows)"
                )
            )
        if item.get_closest_marker("docker") is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Skipping {item.name}: Docker tests are not supported on Windows CI"
                )
            )
