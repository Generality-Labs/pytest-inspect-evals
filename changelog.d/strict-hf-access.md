### Added

- Strict mode for `huggingface` tests, turned on with `--require-hf-access`, `REQUIRE_HF_ACCESS=1` or the `require_hf_access` ini option. A non-empty environment variable wins over the flag, and the flag wins over the ini option. In strict mode a gated-dataset error fails the test, a missing `HF_TOKEN` stops the run with a usage error, and the test report gets an `hf_access_denied` entry in `user_properties` that `--report-log` writes out. The terminal output ends with a "Hugging Face access denied" section that links to each dataset's terms page.

### Changed

- Gated-dataset errors on `huggingface` tests are no longer rerun, so they fail or skip without the 60 second retry delay.
- The Hugging Face dataset viewer helpers raise the Hub's `GatedRepoError` or `RepositoryNotFoundError` when the viewer API returns a non-200 response, where they previously raised an `AssertionError`.
