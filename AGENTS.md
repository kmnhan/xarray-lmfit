# Repository Guidelines

## Project Structure & Modules

- Core package lives in `src/xarray_lmfit/`: `modelfit.py` (model orchestration), `_io.py` (serialization helpers), and `_utils.py` (shared utilities); `__init__.py` wires the public API.
- Tests sit in `tests/` with `test_modelfit.py` and `test_io.py` plus `conftest.py` fixtures. Keep new tests co-located with the feature under test.
- Docs source is under `docs/source/`; build artifacts land in `docs/build/`. Avoid committing generated HTML except for debugging.
- Distribution artifacts (if any) appear in `dist/`; do not edit generated files directly.

## Setup, Build, and Test Commands

- Install (preferred): `uv sync --dev` to resolve `uv.lock` and bring in dev/test/doc extras. If using pip, `pip install -e .` plus equivalent dev tools is acceptable.
- Run tests: `uv run pytest` (configuration in `pyproject.toml` enforces strict markers/import mode). Add `--cov=src --cov-report=term-missing` for coverage parity with CI.
- Lint/format: `uv run ruff check .` and `uv run ruff format .` (line length 88, 4-space indent). Fix warnings before committing.
- Type checks: `uv run mypy src` (tests/docs excluded by default). Add `# type: ignore[code]` only when justified.
- Docs: `make -C docs html` to build locally; ensure new pages render before proposing changes.

## Coding Style & Naming Conventions

- Follow Python/PEP 8 with the repo defaults: 4-space indent, double quotes, and ruff’s configured rule set. Prefer explicit imports and module-level constants in ALL_CAPS.
- Modules and files use `snake_case`; internal helpers may be prefixed with `_`. Keep public API additions surfaced through `__init__.py`.
- Write docstrings in NumPy style when adding public functions/classes; rely on ruff for docstring linting coverage.

## Testing Guidelines

- Name test files/functions `test_*.py` / `test_*` and mirror the module under test. Use parametrization for multi-case coverage.
- Avoid fragile numerical assertions; prefer tolerances via `numpy.testing` or `pytest.approx`.
- Add regression tests alongside bug fixes; update or add fixtures in `tests/conftest.py` rather than ad-hoc setup in individual tests.

## Commit & Pull Request Guidelines

- Commit messages: follow Commitizen (cz-changeup) types visible in `pyproject.toml` (`feat`, `fix`, `perf`, `refactor`, plus `BREAKING CHANGE`). Use `uv run cz c` to stay compliant.
- Keep commits focused and include relevant tests/docs changes. Run linters and pytest before pushing.
- Pull requests should summarize intent, list observable changes, link issues, and include screenshots or example commands when altering user-facing behavior (API, docs, CLI).
