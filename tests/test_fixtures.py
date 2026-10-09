from pathlib import Path

import pytest


def test_set_model_roles_resolves_and_resets(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        from inspect_ai.model import get_model

        def test_roles(set_model_roles):
            model = get_model("mockllm/model")
            with set_model_roles(grader=model):
                assert get_model(role="grader") is model
            with pytest.raises(ValueError):
                get_model(role="grader")
        """
    )
    pytester.runpytest().assert_outcomes(passed=1)


def test_mock_docker_sandbox_runs_eval_without_docker(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    pytester.makepyfile(
        """
        from inspect_ai import Task, eval
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import includes
        from inspect_ai.solver import generate

        def test_eval(mock_docker_sandbox, tmp_path):
            task = Task(
                dataset=[Sample(input="hi", target="x")],
                solver=generate(),
                scorer=includes(),
                sandbox="docker",
            )
            [log] = eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
            assert log.status == "success"
        """
    )
    pytester.runpytest().assert_outcomes(passed=1)
