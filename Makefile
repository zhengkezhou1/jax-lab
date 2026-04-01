.PHONY: lint

lint:
	ruff format python/
	ruff check --fix python/
