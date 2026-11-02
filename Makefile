.PHONY: install lint test ingest train serve up down

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

# ── Quality gates ──────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	black src/ tests/

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest --cov=src/aqforecast --cov-report=term-missing tests/

test-integration:
	pytest -m integration --cov=src/aqforecast tests/

# ── Pipeline steps (wrappers around dvc / python -m) ───────────────────────
ingest:
	python -m aqforecast.ingest

train:
	dvc repro

# ── API ────────────────────────────────────────────────────────────────────
serve:
	uvicorn aqforecast.api.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker Compose ─────────────────────────────────────────────────────────
up:
	docker compose up --build -d

down:
	docker compose down

