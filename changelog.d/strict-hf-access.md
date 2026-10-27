### Added

- Strict mode for `huggingface` tests, turned on with `--require-hf-access`, `REQUIRE_HF_ACCESS=1` or the `require_hf_access` ini option. A non-empty environment variable wins over the flag, and the flag wins over the ini option. In strict mode a gated-dataset error fails the test, a missing `HF_TOKEN` stops the run with a usage error, and the test report gets an `hf_access_denied` entry in `user_properties` that `--report-log` writes out. The terminal output ends with a "Hugging Face access denied" section that links to each dataset's terms page.

### Changed

- Gated-dataset errors on `huggingface` tests are no longer retried in either mode, so they fail or skip without the 60 second retry delay.
- When the Hugging Face dataset viewer helpers get a non-200 response, they now check access with the Hub. A gated dataset raises `GatedRepoError`, which strict mode fails on and the default mode skips. A missing or private dataset raises `RepositoryNotFoundError`, which is an ordinary failure. Any other non-200 response still raises `AssertionError`.
