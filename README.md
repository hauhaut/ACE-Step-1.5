# ACE-Step-1.5

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Install uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Project Dependencies

```bash
# Sync all dependencies
uv sync
```

### Run the Project

```bash
# Simplest way - run directly with uv
uv run acestep

# Run with parameters
uv run acestep --port 7860 --server-name 0.0.0.0 --share

# Or use the full module path
uv run python -m acestep.acestep_v15_pipeline

# Just Run profiling
uv run profile_inference.py

# Or activate the virtual environment first
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

acestep
```

Available parameters:
- `--port`: Server port (default: 7860)
- `--server-name`: Server address (default: 127.0.0.1, use 0.0.0.0 to listen on all interfaces)
- `--share`: Create a public share link
- `--debug`: Enable debug mode

## Development

Add new dependencies:

```bash
# Add runtime dependencies
uv add package-name

# Add development dependencies
uv add --dev package-name
```

Update dependencies:

```bash
uv sync --upgrade
```