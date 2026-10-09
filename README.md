# pytest-inspect-evals

A pytest plugin with the test gates, fixtures and helpers shared by [Inspect AI](https://inspect.aisi.org.uk/) evaluation repos. It started as the shared test code in [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals). Installing it replaces a copied `conftest.py` and `tests/utils/` with one pinned dependency.

## Install

```bash
uv add --dev pytest-inspect-evals
# or, to use the Hugging Face dataset helpers:
uv add --dev "pytest-inspect-evals[huggingface]"
```

The plugin loads automatically. There is nothing to add to `conftest.py`.

## Test gates

Four markers mark tests that are skipped unless you switch them on.

| Marker             | CLI flag             | Env var                      | Pytest config option             | Default |
| ------------------ | -------------------- | ---------------------------- | -------------------------------- | ------- |
| `slow`             | `--runslow`          | `RUN_SLOW_TESTS`             | `inspect_evals_slow`             | off     |
| `dataset_download` | `--dataset-download` | `RUN_DATASET_DOWNLOAD_TESTS` | `inspect_evals_dataset_download` | off     |
| `k8s`              | `--runk8s`           | `RUN_K8S_TESTS`              | `inspect_evals_k8s`              | off     |
| `gpu`              | `--rungpu`           | `RUN_GPU_TESTS`              | `inspect_evals_gpu`              | off     |

The env var wins when it is set to a non-empty value: `1`, `true`, `yes` or `on` switch the gate on, anything else switches it off. Otherwise the CLI flag switches it on. Otherwise the config option decides, and the default is off. An empty env var counts as unset.

To run a gate by default in your repo, set its option in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
inspect_evals_dataset_download = true
```

Three more markers have fixed behaviour:

- `huggingface`: skipped when `HF_TOKEN` is unset or blank. Otherwise the test is retried up to twice, 60 seconds apart, and a gated-dataset error (`GatedRepoError`, or `DatasetNotFoundError` mentioning a gated dataset) is reported as a skip.
- `docker` and `posix_only`: skipped on Windows.

The plugin registers all seven markers, so they work under `--strict-markers` without any config.

## Adding your own gate

`skip_if_marker_present` is the function the plugin uses. Call it from your `conftest.py` for repo-specific gates:

```python
from pytest_inspect_evals.gates import skip_if_marker_present


def pytest_addoption(parser):
    parser.addoption("--custom-smoke", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "custom_smoke: tests that call a live service")


def pytest_collection_modifyitems(config, items):
    skip_if_marker_present(
        config,
        items,
        marker="custom_smoke",
        cli_flag="--custom-smoke",
        env_var="RUN_CUSTOM_SMOKE_TESTS",
        reason="custom_smoke tests disabled (set RUN_CUSTOM_SMOKE_TESTS=1 or pass --custom-smoke)",
    )
```

## Fixtures

`set_model_roles` returns a context manager that sets model roles, for testing scorers that call a grader model:

```python
def test_judge_scorer(set_model_roles):
    with set_model_roles(grader=get_model("mockllm/model")):
        ...
```

`mock_docker_sandbox` stubs the Docker sandbox lifecycle, so an eval with `sandbox="docker"` runs without Docker. Combine it with mocked `sandbox().exec` results to test a scorer end to end:

```python
def test_e2e(mock_docker_sandbox):
    [log] = eval(my_task(), model="mockllm/model")
    assert log.status == "success"
```

## Helpers

- `pytest_inspect_evals.assertions`: `assert_task_structure`, `assert_eval_success`, `run_single_sample_eval`, `get_metric_value`.
- `pytest_inspect_evals.solvers`: `mock_solver_with_output`, `run_command`.
- `pytest_inspect_evals.sandbox`: `MockExecResult`, `create_sandbox_tool_task`, `assert_sandbox_test_passed`.
- `pytest_inspect_evals.metrics`: `run_metrics`, `assert_agreeing_epochs_change_nothing`, `MetricEvalError`. These run custom metrics through the real epoch reducer, the way an eval does.
- `pytest_inspect_evals.huggingface` (needs the `huggingface` extra): `assert_huggingface_dataset_structure`, `get_dataset_infos_dict`, `assert_huggingface_dataset_is_valid`, `assert_dataset_contains_subsets`, `assert_dataset_has_columns`. These check a dataset's schema through the Hugging Face dataset viewer API without downloading it.

## Migrating from a copied conftest

If your `conftest.py` came from inspect_evals or inspect-evals-template, remove what the plugin now provides:

- the `pytest_addoption` lines for `--runslow`, `--dataset-download`, `--runk8s` and `--rungpu`;
- the matching skip logic in `pytest_collection_modifyitems`;
- the seven markers above from your `markers =` config (keeping them is harmless).

If you keep the options, pytest stops at startup with `argparse.ArgumentError: argument --runslow: conflicting option string: --runslow`.

## Private inspect_ai APIs

Two fixtures use inspect_ai internals that have no public equivalent: `set_model_roles` calls `inspect_ai.model._model.init_model_roles`, and `mock_docker_sandbox` patches `inspect_ai.util._sandbox.docker.docker.DockerSandboxEnvironment`. A future inspect_ai release could break them.

## Development

```bash
uv sync
uv run pre-commit install   # optional: run the lint stack on every commit
uv run pytest
uv run basedpyright src
```

Each pull request adds a changelog fragment rather than editing `CHANGELOG.md`, so concurrent PRs don't conflict. Run `uv run scriv create`, uncomment the sections that apply in the new file under `changelog.d/`, and commit it with the change, or delete it if the change needs no entry. To release, run the **Prepare release** workflow from the Actions tab. It bumps `version` in `pyproject.toml`, collects the fragments into `CHANGELOG.md`, and opens a release pull request, whose CI starts once you click **Approve workflows to run** on it; merging it tags the release. It needs _Settings → Actions → General_ → **Allow GitHub Actions to create and approve pull requests**. By hand, the same is `uv version --bump minor` and `uv run scriv collect`.

Linting (ruff, [zizmor](https://docs.zizmor.sh/), mdformat) runs via [pre-commit](https://pre-commit.com); CI runs the same stack plus basedpyright and pytest via the shared [`python-ci`](https://github.com/Generality-Labs/python-project-template) reusable workflow.

## Releasing

See [RELEASING.md](RELEASING.md).
