# Geti Prompt Testing Framework

This directory contains the testing framework for the `getiprompt` library. We use `pytest` for test discovery and execution.

## Directory Structure

The `tests` directory mirrors the `src/getiprompt` package structure. This makes it easy to locate the tests for a specific module. For example, tests for a module at `src/getiprompt/filters/masks/example.py` should be placed in `tests/filters/masks/test_example.py`.

```
geti-prompt/
└── lib
    ├──src/
    │  └── getiprompt/
    │        ├── filters/
    │        │   └── masks/
    │        │       └── mask_filter_base_copy.py
    │        └── ...
    └── tests/
        ├── filters/
        │    └── masks/
        │        └── test_box_aware_mask_filter.py
        └── ...
```

## Running Tests

To run the tests, you can use the `pytest` command from the root of the repository.

### Run All Tests

To run the entire test suite:

```bash
pytest
```

### Run a Specific Test File

To run a specific test file, provide the path to the file:

```bash
pytest tests/filters/masks/test_box_aware_mask_filter.py
```

### Run a Specific Test

To run a single test from a file, use the `-k` flag to specify a keyword expression:

```bash
pytest -k "test_basic_containment_filtering"
```

## Writing Tests

- Test filenames should be prefixed with `test_`.
- Test function names should be prefixed with `test_`.
- Use the `assert` statement for all checks.
- Use `pytest` fixtures for setup and teardown logic when needed.

For more information on `pytest`, refer to the [official documentation](https://docs.pytest.org/en/stable/).
