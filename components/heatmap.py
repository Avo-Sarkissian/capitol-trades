"""
components/heatmap.py
Tab 2: Sector Heatmap — net congressional buying activity by sector × month.
Green = net buying, Red = net selling, White = neutral.
"""

from __future__ import annotations
from io import StringIO

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State

CHART_BG   = "#06090f"
PAPER_BG   = "#06090f"
GRID_COLOR = "#1e2a36"
TEXT_COLOR = "#e2e8f0"
GOLD       = "#f0c040"


def build_heatmap_tab(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Div:
    """
    Build the Sector Heatmap tab layout.

    Args:
        trades_df:  Filtered trades DataFrame.
        prices_df:  Not used by this tab (kept for consistent signature).

    Returns:
        html.Div with chart.
    """
    return html.Div([
        html.Div(
            style={"marginBottom": "12px", "fontSize": "11px", "color": "#7a90b0"},
            children=(
                "Net congressional buying (buys − sells) by GICS sector and month. "
                "Green = net buying, Red = net selling."
            ),
        ),
        dcc.Loading(
            type="circle",
            color=GOLD,
            children=dcc.Graph(
                id="heatmap-chart",
                config={"displayModeBar": True},
                style={"height": "calc(100vh - 200px)"},
                figure=make_heatmap_figure(trades_df),
            ),
        ),
    ])


def make_heatmap_figure(trades_df: pd.DataFrame) -> go.Figure:
    """
    Build the sector × month heatmap figure.

    Cell value = (sum of buy midpoints) − (sum of sell midpoints) for that cell.
    Normalized to millions of dollars.

    Args:
        trades_df: Filtered trades DataFrame with Sector, TransactionDate, AmountMidpoint.

    Returns:
        Plotly Figure.
    """
    if trades_df.empty or "Sector" not in trades_df.columns:
        return _empty_fig("No data to display.")

    df = trades_df.copy()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["Month"] = df["TransactionDate"].dt.to_period("M").astype(str)

    # Assign sign: buys are positive, sells negative
    df["SignedAmount"] = df.apply(
        lambda r: r["AmountMidpoint"] if "purchase" in str(r["Transaction"]).lower()
                  else -r["AmountMidpoint"],
        axis=1,
    )

    # Aggregate
    pivot = (
        df.groupby(["Sector", "Month"])["SignedAmount"]
        .sum()
        .unstack(fill_value=0)
        / 1_000_000  # convert to $M
    )

    # Drop 'Unknown' sector if it exists
    pivot = pivot[pivot.index != "Unknown"]

    if pivot.empty:
        return _empty_fig("Not enough sector data.")

    # Sort sectors alphabetically, months chronologically
    pivot = pivot.sort_index(axis=0)  # sectors
    pivot = pivot.sort_index(axis=1)  # months

    sectors = pivot.index.tolist()
    months  = pivot.columns.tolist()
    z_vals  = pivot.values.tolist()

    # Custom hover text
    hover_text = []
    for s in sectors:
        row_texts = []
        for m in months:
            net = pivot.loc[s, m]
            # Count trades in this cell
            cell_mask = (df["Sector"] == s) & (df["Month"] == m)
            n_trades  = int(cell_mask.sum())
            top_traders = (
                df[cell_mask]
                .groupby("Representative")["AmountMidpoint"]
                .sum()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
            traders_str = ", ".join(top_traders) if top_traders else "—"
            row_texts.append(
                f"<b>{s}</b> · {m}<br>"
                f"Net: ${net:+.2f}M<br>"
                f"Trades: {n_trades}<br>"
                f"Top: {traders_str}"
            )
        hover_text.append(row_texts)

    # Color scale: red → white → green
    colorscale = [
        [0.0,  "#7f1d1d"],   # deep red (selling)
        [0.3,  "#ef4444"],   # bright red
        [0.5,  "#1e3358"],   # neutral (dark navy)
        [0.7,  "#22c55e"],   # bright green
        [1.0,  "#14532d"],   # deep green (buying)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=months,
            y=sectors,
            colorscale=colorscale,
            zmid=0,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorbar={
                "title": {"text": "Net $M", "font": {"color": TEXT_COLOR}},
                "tickfont": {"color": TEXT_COLOR},
                "bgcolor": CHART_BG,
                "bordercolor": GRID_COLOR,
                "outlinecolor": GRID_COLOR,
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono, monospace", "size": 11},
        margin={"l": 160, "r": 20, "t": 40, "b": 100},
        xaxis={
            "title": "Month",
            "gridcolor": GRID_COLOR,
            "tickangle": -45,
            "tickfont": {"size": 9},
        },
        yaxis={
            "title": "GICS Sector",
            "gridcolor": GRID_COLOR,
            "tickfont": {"size": 11},
            "autorange": "reversed",
        },
    )

    return fig


def _empty_fig(message: str) -> go.Figure:
    """Return a blank figure with an annotation."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR},
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": "#7a90b0"},
        }],
    )
    return fig


# ── Callback: re-render when store updates ─────────────────────────────────────

@callback(
    Output("heatmap-chart", "figure"),
    Input("store-filtered-trades", "data"),
)
def update_heatmap(store_data: str):
    """Redraw the heatmap whenever the global filter store changes."""
    if not store_data:
        return _empty_fig("No data.")
    df = pd.read_json(StringIO(store_data), orient="split")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["AmountMidpoint"]  = pd.to_numeric(df["AmountMidpoint"], errors="coerce").fillna(0)
    return make_heatmap_figure(df)
