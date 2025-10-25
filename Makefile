.PHONY: help install test lint format clean build release-patch release-minor release-major

help:
	@echo "Available commands:"
	@echo "  make install        - Install package with dev dependencies"
	@echo "  make test           - Run tests with coverage"
	@echo "  make lint           - Run linters (ruff, mypy, black check)"
	@echo "  make format         - Format code (black, isort, ruff)"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make build          - Build package (wheel + sdist)"
	@echo "  make release-patch  - Bump patch version (0.1.0 → 0.1.1) and push tag"
	@echo "  make release-minor  - Bump minor version (0.1.0 → 0.2.0) and push tag"
	@echo "  make release-major  - Bump major version (0.1.0 → 1.0.0) and push tag"

install:
	pip install -e ".[dev,fast,syntax]"

test:
	pytest tests/ -v --cov=fuzzy_json_repair --cov-report=term-missing

lint:
	ruff check fuzzy_json_repair tests
	mypy fuzzy_json_repair
	black --check fuzzy_json_repair tests

format:
	black fuzzy_json_repair tests
	isort fuzzy_json_repair tests
	ruff check --fix fuzzy_json_repair tests

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage coverage.xml

build: clean
	python -m build
	twine check dist/*

release-patch:
	@echo "🚀 Releasing patch version..."
	bump2version patch
	git push origin main --tags
	@echo "✅ Done! Check GitHub Actions for PyPI publish status"

release-minor:
	@echo "🚀 Releasing minor version..."
	bump2version minor
	git push origin main --tags
	@echo "✅ Done! Check GitHub Actions for PyPI publish status"

release-major:
	@echo "🚀 Releasing major version..."
	bump2version major
	git push origin main --tags
	@echo "✅ Done! Check GitHub Actions for PyPI publish status"
