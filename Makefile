# AeroSentinel — common tasks. Every target runs through uv (never pip).
.PHONY: setup data-cmapss data-ncmapss test test-all lint train report app api

setup:
	uv sync

data-cmapss:
	uv run python scripts/download_data.py cmapss

data-ncmapss:
	uv run python scripts/download_data.py ncmapss

test:
	uv run pytest -q -m "not slow"

test-all:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

# Usage: make train EXP=<dataset>_<model>_<setting>
train:
	@test -n "$(EXP)" || { echo "usage: make train EXP=<dataset>_<model>_<setting>"; exit 2; }
	uv run python -m aerosentinel.train experiment=$(EXP)

report:
	@if [ -f scripts/make_tables.py ]; then \
		uv run python scripts/make_tables.py; \
	else \
		echo "scripts/make_tables.py does not exist yet (arrives in Phase 1)."; \
	fi

app:
	uv run streamlit run app/dashboard.py

api:
	uv run uvicorn aerosentinel.serve.api:app --reload
