# Development

Several tools are used to enforce style guides, type checking, and linting.

## Style Guide

For Python, follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

## Poetry Setup

To set up the development environment, first install Poetry. Then install the Poetry environment.
The command below will install all dependencies, including the development dependencies.

```shell
poetry install
poetry run pre-commit install
```

**Note**: Some users have experienced errors when installing the Poetry environment on Ubuntu 22.04. Specifically,
the psycopg2 package is not able to be installed as a dependency due to missing system packages, such as `gcc`.
Running the following commands before `poetry install` fixed that issue:

```shell
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install python3-dev
```

## pre-commit

The pre-commit hooks will run automatically anytime `git commit` is run. They can also be run manually via

```shell
poetry run pre-commit run --all-files
```

To run a single pre-commit hook by hook id, look for the `id` of the hook in
[.pre-commit-config.yaml](./.pre-commit-config.yaml) and then run:

```shell
poetry run pre-commit run {id} --all-files
```

For example, to only run the `ruff` hook, run:

```shell
poetry run pre-commit run ruff --all-files
```

One can also run pre-commit hooks on specific files using:

```shell
poetry run pre-commit run --files {path_to_files}
```
## Tests

[Pytest](https://docs.pytest.org/en/8.0.x/) is used for testing (unit and integration).
It is installed as a development dependency in the Poetry environment. Tests should be functions that have the prefix `test_`.

<!-- All Pytest fixtures currently live in [tests/conftest.py](tests/conftest.py). -->

All tests can be run via:

```shell
poetry run pytest tests/
```

Individual tests can be run by referencing a specific file in place of `tests/` in the command above.

Helpful flags to add are verbose mode (`-v`) and to show stdout (`-s`), e.g.

```shell
poetry run pytest -vs tests/
```

## Coverage

[Coverage](https://coverage.readthedocs.io/en/7.4.4/) is used to measure the code
coverage of the Pytest tests. To measure the coverage of all of the Pytest tests over
the code in the `skin_tone_extraction` package, run:

```shell
poetry run coverage run --source=skin_tone_extraction/ -m pytest tests/
```

To generate a human-readable coverage report in the terminal run:

```shell
poetry run coverage report
```

Other useful output formats include the html format:

```shell
poetry run coverage html
```

which creates a `htmlcov` folder with html files containing line-by-line coverage
information in each file over which the coverage was run (see `--source` argument above).
