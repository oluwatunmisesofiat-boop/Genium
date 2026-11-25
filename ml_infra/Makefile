.PHONY: install dev test lint format clean run-api sync-greenbook

install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev,cv-training,storage]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/genie --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/genie/

format:
	ruff check --fix src/ tests/

run-api:
	uvicorn genie.api.main:app --host 0.0.0.0 --port 8000 --reload

sync-greenbook:
	python scripts/sync_greenbook.py

build-index:
	python scripts/build_vector_index.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist build *.egg-info
