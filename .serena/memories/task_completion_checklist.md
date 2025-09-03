# Task Completion Checklist

When completing any development task in this RAG project, run the following commands to ensure code quality:

## Python Backend Changes
1. **Linting**: `make lint` or `python -m ruff check .`
2. **Format**: `make format` or `python -m ruff format .`
3. **Type checking**: `make type-check` or `python -m mypy .`
4. **Tests**: `make test` or `python -m pytest tests/ -v`

## Web Frontend Changes
1. **Linting**: `cd web && yarn lint`
2. **Type checking**: `cd web && yarn type-check`
3. **Build test**: `cd web && yarn build`

## Combined Quality Check
- `make lint` - Runs linting for both Python and web
- `make type-check` - Runs type checking for both Python and web

## Auto-fix Options
- `make lint-fix` - Automatically fix linting issues
- `python -m ruff check . --fix` - Fix Python linting issues
- `cd web && yarn lint:fix` - Fix web linting issues

## Development Testing
- Test the main entrypoints work: `python main.py --help`
- For web changes: Test that `make dev-web` and `make web-dev` work properly
- For backend changes: Test CLI commands work as expected