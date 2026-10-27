### Fixed

- A gated-dataset error raised in a fixture during setup is now handled like one raised in the test body. Without strict mode the test is skipped instead of reported as an error. With `--require-hf-access`, the setup error is tagged `hf_access_denied` and listed in the "Hugging Face access denied" summary, so CI tooling no longer treats it as an ordinary failure.
