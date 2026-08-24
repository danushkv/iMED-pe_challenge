.PHONY: setup verify audit test

setup:
	uv sync --extra dev --frozen

verify:
	uv run --frozen python scripts/check_install.py

audit:
	uv run --frozen python scripts/audit_release.py

test:
	uv run --frozen pytest

