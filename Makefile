.PHONY: help clean dev-tools doc doc-autobuild format-check image package pbd-test test type-check \
	venv venv-activate lint-check

SHELL:=/bin/bash
PKG_NAME:=vit_keras
IMAGE_NAME:=vit-keras

# Select specific Python tests to run using pytest selectors
# e.g., make test TEST_SCOPE='-m "not_integration" tests/api/'
TEST_SCOPE?=tests/

# Prefix for running commands on the host vs in Docker (e.g., dev vs CI)
EXEC:=poetry run
SPHINX_AUTO_EXTRA:=


help:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Tips"
	@echo "----"
	@echo '- Run `make venv-activate` to activate the project virtualenv in your shell'
	@echo '  e.g., make test TEST_SCOPE="-m not_integration tests/api/"'

clean-all: clean ## Make a completely clean source tree, including .venv
	@rm -rf .venv

clean: ## Make a clean source tree, keeping .venv
	@rm -rf .coverage .mypy_cache .pytest_cache build dist docs/build *.egg-info
	@-find . -name '__pycache__' ! -path './.venv/*' -exec rm -rf {} \;

doc: ## Make HTML documentation from Sphinx source
	@mkdir -p docs/build
	@$(EXEC) sphinx-build -M html docs/source docs/build/html

doc-autobuild: ## Make a local HTML doc server that updates on changes to from Sphinx source
	@$(EXEC) sphinx-autobuild -b html docs/source docs/build/html $(SPHINX_AUTO_EXTRA)

format-check: ## Make black check source formatting
	@$(EXEC) black --diff --check .

format: ## Make black unabashedly format source code
	@$(EXEC) black .

lock: ## Make a poetry.lock file
	@$(EXEC) poetry lock

package: ## Make a local build of the Python package, source dist and wheel
	@mkdir -p dist
	@$(EXEC) poetry build

pdb-test: ## Make pytest run tests and drop to pdb on error
	@$(EXEC) pytest -vxrs --pdb $(TEST_SCOPE)

test: ## Make pytest run tests
	@$(EXEC) pytest -vxrs $(TEST_SCOPE)

type-check: ## Make mypy check types
	@$(EXEC) mypy $(PKG_NAME) tests

lint-check: ## Make pylint lint the package
	@$(EXEC) pylint --jobs 0 vit_keras

venv: lock ## Make a poetry virtualenv on the host
	@POETRY_VIRTUALENVS_IN_PROJECT=true poetry install

lab: ## Start a jupyter lab instance
	@$(EXEC) jupyter lab