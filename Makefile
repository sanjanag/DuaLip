.PHONY: all install precommit test checkstyle lint format clean

all: checkstyle test

install:
	python3 -m pip install -e .[dev]
	pre-commit install

# Run tests (pytest picks config from [tool.pytest.ini_options] in pyproject.toml)
test:
	pytest

# Run all pre-commit hooks across the repo
checkstyle:
	pre-commit run --all-files -v

# Convenience: run just lint or just format via pre-commit
lint:
	pre-commit run flake8 --all-files -v

format:
	pre-commit run isort --all-files -v
	pre-commit run black --all-files -v

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	python3 -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
