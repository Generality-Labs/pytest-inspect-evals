import pytest


def test_plugin_is_registered(pytester: pytest.Pytester) -> None:
    config = pytester.parseconfig()
    assert config.pluginmanager.has_plugin("pytest_inspect_evals")


def test_package_is_marked_typed() -> None:
    from importlib.resources import files

    assert files("pytest_inspect_evals").joinpath("py.typed").is_file()
