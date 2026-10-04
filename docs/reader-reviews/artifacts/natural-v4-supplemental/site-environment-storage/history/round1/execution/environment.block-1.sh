uv sync --frozen --extra cpu --group notebook
uv run --extra cpu --group notebook python scripts/check_env.py
uv run --extra cpu --group notebook jupyter lab
