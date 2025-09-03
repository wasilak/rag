# Code Style and Conventions

## Python Style
- **Type hints**: Use type hints consistently (as seen in `get_rag_logger(name: str = "RAG") -> logging.Logger`)
- **Docstrings**: Use triple-quoted docstrings for functions 
- **Line length**: 120 characters max (configured in setup.cfg)
- **Formatter**: ruff format for Python code formatting
- **Linter**: ruff for Python linting
- **Type checker**: mypy for static type checking
- **Import style**: Follow standard Python import conventions

## Web/TypeScript Style
- **Framework**: React 19 with TypeScript
- **UI Library**: Material-UI (MUI) v7
- **Linting**: ESLint for JavaScript/TypeScript
- **Type checking**: TypeScript compiler (tsc)

## File Structure Patterns
- Core logic in `libs/` directory with modular organization
- Web interface separated in `web/` directory
- Examples in `examples/` directory
- Main CLI entry point in `main.py`

## Naming Conventions
- Snake_case for Python functions and variables
- CamelCase for React components
- Descriptive function names (e.g., `get_rag_logger`, `format_footnotes`)

## Ignored Linting Rules
- E501: Line too long (handled by max-line-length setting)
- W503: Line break before binary operator