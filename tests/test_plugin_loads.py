import pytest


def test_plugin_is_registered(pytester: pytest.Pytester) -> None:
    config = pytester.parseconfig()
    assert config.pluginmanager.has_plugin("pytest_inspect_evals")
