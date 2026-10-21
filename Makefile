
.PHONY: setup check lint format-check typecheck format

setup:
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed"; exit 1; }
	uv sync --frozen
	uv run --no-sync pre-commit install

check: lint format-check typecheck

lint:
	uv run --no-sync ruff check .

format-check:
	uv run --no-sync ruff format --check .

typecheck:
	uv run --no-sync mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .
