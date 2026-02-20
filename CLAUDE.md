# Capitol Trades — Congressional Trading Visualizer

Interactive Plotly Dash app that visualizes congressional stock trading data alongside market performance.

## Developer Context

- Solo developer, vibe coder — I architect and prompt, you implement
- I have low code knowledge. Write readable, well-commented code. Flag pitfalls before they bite
- Python is my only language. No JS/HTML unless Dash requires it inline
- macOS (M1 Max), VS Code, running in venv at project root
- YOU HAVE FULL AUTONOMY: create files, edit files, run commands, install packages, and execute anything within this project folder without asking permission. Just do it and tell me what you did after.
- The ONLY time you should pause and ask me is if you need a design/UX decision (e.g., "should this filter be a dropdown or radio buttons?")

## Git & GitHub

- This project has a GitHub remote already configured
- After EVERY meaningful change (new feature, bug fix, refactor), automatically:
  1. `git add -A`
  2. `git commit -m "<concise description of what changed>"`
  3. `git push`
- Use conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `style:`
- Commit frequently — small commits are better than large ones
- NEVER ask me before committing or pushing. Just do it.
- Do NOT create branches. Work directly on `main`.

## Tech Stack

- **Python 3.11** with venv (activate with `source venv/bin/activate`)
- **Dash 4.0.0** (IMPORTANT: this is the latest major version — see Dash rules below)
- **Plotly 6.x** for charts
- **Pandas** for data manipulation
- **yfinance** for market data
- **NetworkX** for graph analysis
- **Requests** for API calls
- Install any additional packages as needed without asking — just `pip install <pkg>` and update requirements.txt

## CRITICAL: Dash 4.0 Rules

IMPORTANT: You are likely trained on Dash 2.x patterns. This project uses Dash 4.0. Follow these rules:

- Import as: `from dash import Dash, dcc, html, callback, Input, Output, State, dash_table`
- NEVER use `import dash_core_components`, `import dash_html_components`, or `import dash_table` as separate packages
- Use `@callback` decorator, not `@app.callback`
- `dash.dash_table` is deprecated — use AG Grid (`dash_ag_grid`) if tables needed, or basic `html.Table`
- `app.run()` not `app.run_server()` — both work but `run()` is preferred
- Dash 4.0 uses React 18 under the hood
- When in doubt, check https://dash.plotly.com/ for current API

## Project Structure

```
Final Project/
├── CLAUDE.md
├── app.py              # Main Dash app entry point (keep thin)
├── data/
│   ├── fetch.py        # API calls to Quiver Quant + yfinance
│   ├── process.py      # Data cleaning, joins, alpha calc
│   └── cache/          # Local CSV caches to avoid re-fetching
├── components/
│   ├── timeline.py     # Trade timeline overlay on price chart
│   ├── heatmap.py      # Sector heatmap
│   ├── scatter.py      # Alpha scatter plot
│   ├── network.py      # Committee-sector network graph
│   └── leaderboard.py  # Rankings bar chart
├── assets/
│   └── style.css       # Custom CSS (Dash auto-loads from /assets)
├── venv/               # Virtual environment (do NOT touch)
└── requirements.txt
```

## Commands

- `source venv/bin/activate` — activate venv (do this first if not active)
- `python app.py` — run the app (opens at http://127.0.0.1:8050)
- `pip freeze > requirements.txt` — update after any new installs

## Data Sources

- **Quiver Quantitative API** (free tier): congressional trade disclosures. Endpoint: `https://api.quiverquant.com/beta/live/congresstrading`
- **yfinance**: historical stock prices via `yfinance.download()`
- **Committee data**: manual CSV mapping committees → GICS sectors

## Workflow

- Build incrementally — get one visualization working end-to-end before starting the next
- After creating/editing files, run the app yourself to verify it works. If it errors, fix it before telling me it's done
- Cache API data locally as CSVs so I'm not rate-limited during development
- Keep app.py thin — it should only define layout and import callbacks from components/
- If something breaks, fix it and explain what went wrong in one sentence

## Code Style

- Type hints on function signatures
- Docstrings on every function (one-liner is fine)
- f-strings over .format()
- Constants in UPPER_SNAKE_CASE at top of file
- Use plotly.express over plotly.graph_objects when possible (simpler API)
- Well-commented code — assume the reader (me) doesn't know Python deeply

## Known Gotchas

- Quiver Quant free tier has rate limits — always check for cached data first
- yfinance sometimes returns empty DataFrames for delisted tickers — handle gracefully
- Trade disclosure amounts are ranges (e.g., "$1,001-$15,000"), not exact — use midpoint for calculations
- The venv/ folder is gitignored — never modify anything inside it
- Add `venv/`, `data/cache/`, `__pycache__/`, `.DS_Store` to .gitignore
