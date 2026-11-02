### Added

- An autouse `isolated_inspect_store` fixture gives each test a fresh inspect_ai `Store` and restores the previous one afterwards. Outside of `eval()`, inspect_ai uses one `Store` for the whole process, so data one test wrote there reached every later test and slowed later solver and agent tests. Remove any copy of this fixture from your `conftest.py` when you upgrade.
