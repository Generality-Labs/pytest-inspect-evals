import pytest

GATES = [
    ("slow", "--runslow", "RUN_SLOW_TESTS", "inspect_evals_slow"),
    (
        "dataset_download",
        "--dataset-download",
        "RUN_DATASET_DOWNLOAD_TESTS",
        "inspect_evals_dataset_download",
    ),
    ("k8s", "--runk8s", "RUN_K8S_TESTS", "inspect_evals_k8s"),
    ("gpu", "--rungpu", "RUN_GPU_TESTS", "inspect_evals_gpu"),
]

GATED_TEST = """
import pytest

@pytest.mark.{marker}
def test_gated():
    pass
"""


@pytest.fixture(autouse=True)
def _clear_gate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for _, _, env_var, _ in GATES:
        monkeypatch.delenv(env_var, raising=False)


# Ids are prefixed because pytest puts parametrize ids into item.keywords, and the
# gates match keywords: an id of exactly "slow" would gate this test itself.
@pytest.fixture(params=GATES, ids=[f"gate-{g[0]}" for g in GATES])
def gate(request: pytest.FixtureRequest, pytester: pytest.Pytester) -> tuple[str, str, str, str]:
    marker = request.param[0]
    pytester.makepyfile(GATED_TEST.format(marker=marker))
    return request.param


def test_skipped_by_default(pytester: pytest.Pytester, gate: tuple[str, str, str, str]) -> None:
    pytester.runpytest().assert_outcomes(skipped=1)


def test_cli_flag_enables(pytester: pytest.Pytester, gate: tuple[str, str, str, str]) -> None:
    pytester.runpytest(gate[1]).assert_outcomes(passed=1)


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_truthy_env_var_enables(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    gate: tuple[str, str, str, str],
    value: str,
) -> None:
    monkeypatch.setenv(gate[2], value)
    pytester.runpytest().assert_outcomes(passed=1)


def test_ini_option_enables(pytester: pytest.Pytester, gate: tuple[str, str, str, str]) -> None:
    pytester.makeini(f"[pytest]\n{gate[3]} = true\n")
    pytester.runpytest().assert_outcomes(passed=1)


def test_toml_boolean_ini_option_enables(
    pytester: pytest.Pytester, gate: tuple[str, str, str, str]
) -> None:
    pytester.makepyprojecttoml(f"[tool.pytest.ini_options]\n{gate[3]} = true\n")
    pytester.runpytest().assert_outcomes(passed=1)


def test_falsy_env_var_beats_cli_flag(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, gate: tuple[str, str, str, str]
) -> None:
    monkeypatch.setenv(gate[2], "0")
    pytester.runpytest(gate[1]).assert_outcomes(skipped=1)


def test_falsy_env_var_beats_ini_option(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, gate: tuple[str, str, str, str]
) -> None:
    monkeypatch.setenv(gate[2], "no")
    pytester.makeini(f"[pytest]\n{gate[3]} = true\n")
    pytester.runpytest().assert_outcomes(skipped=1)


def test_cli_flag_beats_false_ini_option(
    pytester: pytest.Pytester, gate: tuple[str, str, str, str]
) -> None:
    pytester.makeini(f"[pytest]\n{gate[3]} = false\n")
    pytester.runpytest(gate[1]).assert_outcomes(passed=1)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_env_var_counts_as_unset(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    gate: tuple[str, str, str, str],
    value: str,
) -> None:
    monkeypatch.setenv(gate[2], value)
    pytester.runpytest(gate[1]).assert_outcomes(passed=1)


def test_skip_reason_names_every_switch(
    pytester: pytest.Pytester, gate: tuple[str, str, str, str]
) -> None:
    result = pytester.runpytest("-rs")
    result.stdout.fnmatch_lines([f"*{gate[0]} tests disabled*{gate[2]}=1*{gate[1]}*{gate[3]}*"])


def test_markers_are_registered(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(
        [
            "@pytest.mark.slow:*",
            "@pytest.mark.dataset_download:*",
            "@pytest.mark.k8s:*",
            "@pytest.mark.gpu:*",
            "@pytest.mark.huggingface:*",
            "@pytest.mark.docker:*",
            "@pytest.mark.posix_only:*",
        ]
    )


def test_markers_redeclared_by_repo_pass_strict_markers(pytester: pytest.Pytester) -> None:
    pytester.makeini(
        "[pytest]\nmarkers =\n    slow: repo's own description\n    docker: repo's own description\n"
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.slow
        @pytest.mark.docker
        @pytest.mark.posix_only
        def test_marked():
            pass
        """
    )
    result = pytester.runpytest("--strict-markers", "--runslow")
    assert result.ret == 0


def test_conftest_redefining_a_flag_fails_loudly(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        """
        def pytest_addoption(parser):
            parser.addoption("--runslow", action="store_true", default=False)
        """
    )
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest()
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*conflicting option string: --runslow*"])


def test_public_helper_gates_a_repo_marker(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUN_CUSTOM_SMOKE_TESTS", raising=False)
    pytester.makeconftest(
        """
        from pytest_inspect_evals.gates import skip_if_marker_present

        def pytest_addoption(parser):
            parser.addoption("--custom-smoke", action="store_true", default=False)

        def pytest_configure(config):
            config.addinivalue_line("markers", "custom_smoke: repo-specific gate")

        def pytest_collection_modifyitems(config, items):
            skip_if_marker_present(
                config, items, marker="custom_smoke", cli_flag="--custom-smoke",
                env_var="RUN_CUSTOM_SMOKE_TESTS", reason="custom smoke disabled",
                default_enabled=False,
            )
        """
    )
    pytester.makepyfile(GATED_TEST.format(marker="custom_smoke"))
    pytester.runpytest().assert_outcomes(skipped=1)
    pytester.runpytest("--custom-smoke").assert_outcomes(passed=1)
    monkeypatch.setenv("RUN_CUSTOM_SMOKE_TESTS", "1")
    pytester.runpytest().assert_outcomes(passed=1)
