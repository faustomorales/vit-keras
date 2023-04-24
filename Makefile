.PHONY: docs

PKG_NAME:=vit_keras

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
	@echo '- Run `make shell` to activate the project virtualenv in your shell'
	@echo '  e.g., make test TEST_SCOPE="-m not_integration tests/api/"'

init:  ## Initialize the development environment.
	pip install poetry poetry-dynamic-versioning
	poetry install
	poetry run pip install tensorflow==2.10.0 tensorflow-addons==0.20.0

format-check: ## Make black check source formatting
	@$(EXEC) black --diff --check .

format: ## Make black unabashedly format source code
	@$(EXEC) black .

package: ## Make a local build of the Python package, source dist and wheel
	@mkdir -p dist
	@$(EXEC) poetry build

test: ## Make pytest run tests
	@$(EXEC) pytest -vxrs $(TEST_SCOPE)

type-check: ## Make mypy check types
	@$(EXEC) mypy $(PKG_NAME) tests

lint-check: ## Make pylint lint the package
	@$(EXEC) pylint --rcfile pyproject.toml --jobs 0 $(PKG_NAME)

lab: ## Start a jupyter lab instance
	@$(EXEC) jupyter lab

shell:  ## Jump into poetry shell.
	poetry shell
