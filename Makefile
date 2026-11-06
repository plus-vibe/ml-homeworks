.PHONY: setup check lint format-check typecheck format
.PHONY: s3-up s3-down s3-status
.PHONY: ingest download-raw convert split statistics publish data

DATA_VERSION ?= v1
DATA_DIR = data/processed/cifar10/$(DATA_VERSION)
VAL_SIZE ?= 5000
SEED ?= 42
WORKERS ?= 16

setup:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1; \
	fi
	@UV_BIN="$$(command -v uv || true)"; \
	if [ -z "$$UV_BIN" ]; then UV_BIN="$$HOME/.local/bin/uv"; fi; \
	"$$UV_BIN" sync --frozen && \
	"$$UV_BIN" run --no-sync pre-commit install

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

s3-up:
	docker compose up -d --wait rustfs
	docker compose up --no-recreate s3-init

s3-down:
	docker compose down

s3-status:
	docker compose ps -a

ingest:
	@set -a; . ./.env; set +a; \
	uv run python -m ds_infrastructure_hw.ingest

download-raw:
	@set -a; . ./.env; set +a; \
	uv run python -m ds_infrastructure_hw.data.download

convert:
	uv run python -m ds_infrastructure_hw.data.convert \
		--output-dir $(DATA_DIR)

split:
	uv run python -m ds_infrastructure_hw.data.split \
		--data-dir $(DATA_DIR) \
		--val-size $(VAL_SIZE) \
		--seed $(SEED)

statistics:
	uv run python -m ds_infrastructure_hw.data.statistics \
		--data-dir $(DATA_DIR)

publish:
	@set -a; . ./.env; set +a; \
	uv run python -m ds_infrastructure_hw.data.publish \
		--version $(DATA_VERSION) \
		--seed $(SEED) \
		--val-size $(VAL_SIZE) \
		--workers $(WORKERS)

data:
	@$(MAKE) download-raw
	@$(MAKE) convert DATA_VERSION=$(DATA_VERSION)
	@$(MAKE) split DATA_VERSION=$(DATA_VERSION) VAL_SIZE=$(VAL_SIZE) SEED=$(SEED)
	@$(MAKE) statistics DATA_VERSION=$(DATA_VERSION)
	@$(MAKE) publish DATA_VERSION=$(DATA_VERSION) VAL_SIZE=$(VAL_SIZE) SEED=$(SEED) WORKERS=$(WORKERS)
