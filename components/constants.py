"""
components/constants.py
Shared color and style constants used across all visualization components.
"""

# Chart backgrounds
CHART_BG   = "#06090f"
PAPER_BG   = "#06090f"
GRID_COLOR = "#1e2a36"
TEXT_COLOR = "#e2e8f0"

# Accent colors
GOLD       = "#f0c040"
GOLD_DIM   = "#d4a843"

# Party colors
DEM_BLUE   = "#3b82f6"
REP_RED    = "#ef4444"

# Trade direction colors
BUY_COLOR  = "#22c55e"
SELL_COLOR = "#ef4444"

# Chart font
CHART_FONT = {"color": TEXT_COLOR, "family": "JetBrains Mono, IBM Plex Mono, monospace", "size": 11}


def empty_fig(message: str):
    """Return a blank figure with a centered annotation. Shared across all tabs."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR},
        annotations=[{
            "text": message.replace("\n", "<br>"),
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 13, "color": "#7a90b0"},
        }],
    )
    return fig
