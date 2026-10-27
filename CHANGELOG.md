# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- scriv-insert-here -->

## [0.2.0] - 2026-10-27

### Added

- Strict mode for `huggingface` tests, turned on with `--require-hf-access`, `REQUIRE_HF_ACCESS=1` or the `require_hf_access` ini option. A non-empty environment variable wins over the flag, and the flag wins over the ini option. In strict mode a gated-dataset error fails the test, a missing `HF_TOKEN` stops the run with a usage error (under pytest-xdist, each `huggingface` test fails at setup with the same message instead), and the test report gets an `hf_access_denied` entry in `user_properties` that `--report-log` writes out. The terminal output ends with a "Hugging Face access denied" section that links to each dataset's terms page.

### Changed

- The flaky marker on `huggingface` tests now excludes gated-dataset errors from reruns, so a strict-mode denial fails at once instead of after two retries.
- When the Hugging Face dataset viewer helpers get a non-200 response, they now check access with the Hub. A gated dataset raises `GatedRepoError`, which strict mode fails on and the default mode skips. A missing or private dataset raises `RepositoryNotFoundError`, which is an ordinary failure. Any other non-200 response still raises `AssertionError`.
- `assert_huggingface_dataset_is_valid` and `get_dataset_infos_dict` now raise `GatedRepoError` or `RepositoryNotFoundError` for gated and missing datasets. Callers that catch `AssertionError` for these cases need to catch the Hub errors too.

## [0.1.0] - 2026-10-09

### Added

- First release: test gates for slow, dataset_download, k8s and gpu tests, Hugging Face and Windows handling, the set_model_roles and mock_docker_sandbox fixtures, and the assertion, solver, sandbox, metric and Hugging Face helpers from inspect_evals. Needs inspect_ai 0.3.262 or later.
