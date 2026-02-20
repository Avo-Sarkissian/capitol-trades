"""
components/network.py
Tab 4: Committee-Sector Network Graph.
Nodes = politicians (circles) and sectors (squares).
Edges = politician traded in a sector they have committee oversight over.
Highlights potential conflicts of interest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State

CHART_BG   = "#080e1a"
PAPER_BG   = "#080e1a"
GRID_COLOR = "#1c2e4a"
TEXT_COLOR = "#dce6f5"
GOLD       = "#d4a843"
DEM_BLUE   = "#3b82f6"
REP_RED    = "#ef4444"
SECTOR_CLR = "#d4a843"  # gold for sector nodes


# Simplified committee membership (politician → list of committees)
# In a real app this would come from an API; here we hardcode key examples.
_COMMITTEE_MEMBERSHIPS: dict[str, list[str]] = {
    "Nancy Pelosi":           ["House Intelligence", "House Oversight"],
    "Dan Crenshaw":           ["House Intelligence", "House Energy and Commerce"],
    "Marjorie Taylor Greene": ["House Oversight", "House Armed Services"],
    "Tommy Tuberville":       ["Senate Armed Services", "Senate Agriculture"],
    "Mark Kelly":             ["Senate Armed Services", "Senate Commerce"],
    "Josh Gottheimer":        ["House Financial Services", "House Intelligence"],
    "Virginia Foxx":          ["House Education"],
    "Michael McCaul":         ["House Foreign Affairs", "House Homeland Security"],
    "Ro Khanna":              ["House Armed Services", "House Science Space Technology"],
    "Donald Beyer":           ["House Science Space Technology", "House Ways and Means"],
    "Shelley Moore Capito":   ["Senate Environment and Public Works", "Senate Commerce"],
    "Debbie Stabenow":        ["Senate Agriculture", "Senate Finance"],
    "Pat Toomey":             ["Senate Banking", "Senate Finance"],
    "Richard Burr":           ["Senate Intelligence", "Senate Health Education Labor"],
    "Dianne Feinstein":       ["Senate Judiciary", "Senate Intelligence"],
    "David Perdue":           ["Senate Banking", "Senate Agriculture"],
    "Kelly Loeffler":         ["Senate Banking"],
    "Suzan DelBene":          ["House Ways and Means", "House Financial Services"],
    "Ron Wyden":              ["Senate Finance", "Senate Intelligence"],
    "Maria Cantwell":         ["Senate Commerce", "Senate Energy and Natural Resources"],
    "Mitt Romney":            ["Senate Foreign Relations", "Senate Banking"],
    "Marco Rubio":            ["Senate Intelligence", "Senate Foreign Relations"],
    "Rand Paul":              ["Senate Health Education Labor", "Senate Homeland Security"],
}


def build_network_tab(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Div:
    """
    Build the Committee Network tab layout.

    Args:
        trades_df:  Filtered trades DataFrame.
        prices_df:  Unused — kept for consistent signature.

    Returns:
        html.Div with the network graph.
    """
    return html.Div([
        html.Div(
            style={"marginBottom": "12px", "fontSize": "11px", "color": "#7a90b0"},
            children=(
                "Network of politicians and sectors. An edge exists where a politician "
                "sits on a committee overseeing a sector AND traded stocks in that sector. "
                "Thicker edges = larger dollar volume. Gold nodes = GICS sectors."
            ),
        ),
        dcc.Loading(
            type="circle",
            color=GOLD,
            children=dcc.Graph(
                id="network-chart",
                config={"displayModeBar": True, "scrollZoom": True},
                style={"height": "calc(100vh - 200px)"},
                figure=make_network_figure(trades_df),
            ),
        ),
    ])


def make_network_figure(trades_df: pd.DataFrame) -> go.Figure:
    """
    Build the NetworkX-based, Plotly-rendered conflict-of-interest network.

    Args:
        trades_df: Filtered trades DataFrame with Sector and Representative columns.

    Returns:
        Plotly Figure.
    """
    if trades_df.empty:
        return _empty_fig("No trades in current selection.")

    from data.process import get_committee_sector_mapping
    committee_sector = get_committee_sector_mapping()
    # committee → set of sectors it oversees
    committee_to_sectors: dict[str, set[str]] = {}
    for _, row in committee_sector.iterrows():
        committee_to_sectors.setdefault(row["Committee"], set()).add(row["Sector"])

    # Build politician → set of sectors they have oversight over
    pol_oversight: dict[str, set[str]] = {}
    for pol, committees in _COMMITTEE_MEMBERSHIPS.items():
        sectors: set[str] = set()
        for c in committees:
            sectors |= committee_to_sectors.get(c, set())
        if sectors:
            pol_oversight[pol] = sectors

    # Build politician → sector trading volumes (only conflict trades)
    pol_sector_vol = (
        trades_df.groupby(["Representative", "Sector"])["AmountMidpoint"]
        .sum()
        .reset_index()
    )

    # Build graph: only include edges where oversight AND trading overlap
    G = nx.Graph()
    edges_data: list[dict] = []

    for _, row in pol_sector_vol.iterrows():
        pol    = row["Representative"]
        sector = row["Sector"]
        vol    = row["AmountMidpoint"]

        if sector == "Unknown" or sector == "":
            continue

        # Only draw edge if the politician has committee oversight of this sector
        if pol in pol_oversight and sector in pol_oversight[pol]:
            G.add_node(pol,    node_type="politician")
            G.add_node(sector, node_type="sector")
            G.add_edge(pol, sector, weight=vol)
            edges_data.append({"pol": pol, "sector": sector, "vol": vol})

    if G.number_of_nodes() == 0:
        return _empty_fig(
            "No conflict-of-interest edges found in current filter.\n"
            "Try expanding filters to include more politicians."
        )

    # ── Layout ────────────────────────────────────────────────────
    # Use spring layout, bipartite if possible
    try:
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(G.number_of_nodes()))
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # ── Build Plotly traces ───────────────────────────────────────
    # Edge traces — one per edge so we can set thickness individually
    max_vol = max((e["vol"] for e in edges_data), default=1)
    edge_traces = []

    for edge in G.edges(data=True):
        n1, n2 = edge[0], edge[1]
        x0, y0 = pos[n1]
        x1, y1 = pos[n2]
        weight  = edge[2].get("weight", 1)
        width   = 1 + (weight / max_vol) * 6  # 1-7px

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line={"width": width, "color": "rgba(212,168,67,0.35)"},
            hoverinfo="skip",
            showlegend=False,
        ))

    # Node traces — separate for politicians and sectors
    pol_x, pol_y, pol_text, pol_hover, pol_sizes, pol_colors = [], [], [], [], [], []
    sec_x, sec_y, sec_text, sec_hover, sec_sizes = [], [], [], [], []

    # Compute total volume per node for sizing
    pol_vol  = trades_df.groupby("Representative")["AmountMidpoint"].sum().to_dict()
    sec_vol  = trades_df.groupby("Sector")["AmountMidpoint"].sum().to_dict()
    max_pvol = max(pol_vol.values(), default=1)
    max_svol = max(sec_vol.values(), default=1)

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        if data["node_type"] == "politician":
            party = _get_party(node, trades_df)
            color = DEM_BLUE if party == "Democrat" else REP_RED
            vol   = pol_vol.get(node, 0)
            size  = 14 + (vol / max_pvol) * 20

            committees = _COMMITTEE_MEMBERSHIPS.get(node, [])
            n_trades   = int((trades_df["Representative"] == node).sum())

            pol_x.append(x); pol_y.append(y)
            pol_text.append(node.split()[-1])   # last name only on node
            pol_hover.append(
                f"<b>{node}</b><br>"
                f"Party: {party}<br>"
                f"Committees: {', '.join(committees) or '—'}<br>"
                f"Total trades: {n_trades}<br>"
                f"Est. vol: ${vol/1e6:.2f}M"
            )
            pol_sizes.append(size)
            pol_colors.append(color)
        else:
            vol  = sec_vol.get(node, 0)
            size = 18 + (vol / max_svol) * 20
            n_traders = int(trades_df[trades_df["Sector"] == node]["Representative"].nunique())

            sec_x.append(x); sec_y.append(y)
            sec_text.append(node)
            sec_hover.append(
                f"<b>{node}</b><br>"
                f"Sector total vol: ${vol/1e6:.2f}M<br>"
                f"Unique traders: {n_traders}"
            )
            sec_sizes.append(size)

    politician_trace = go.Scatter(
        x=pol_x, y=pol_y,
        mode="markers+text",
        text=pol_text,
        textposition="top center",
        textfont={"size": 9, "color": TEXT_COLOR},
        marker={
            "symbol": "circle",
            "size": pol_sizes,
            "color": pol_colors,
            "line": {"color": "white", "width": 1},
        },
        hovertext=pol_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        name="Politicians",
    )

    sector_trace = go.Scatter(
        x=sec_x, y=sec_y,
        mode="markers+text",
        text=sec_text,
        textposition="bottom center",
        textfont={"size": 9, "color": GOLD},
        marker={
            "symbol": "square",
            "size": sec_sizes,
            "color": SECTOR_CLR,
            "opacity": 0.85,
            "line": {"color": "white", "width": 1},
        },
        hovertext=sec_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        name="Sectors",
    )

    all_traces = edge_traces + [politician_trace, sector_trace]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono, monospace", "size": 11},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        showlegend=True,
        legend={
            "bgcolor": "rgba(15,31,61,0.9)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
        },
        xaxis={"visible": False},
        yaxis={"visible": False},
        hovermode="closest",
    )
    return fig


def _get_party(politician: str, trades_df: pd.DataFrame) -> str:
    """Look up a politician's party from the trades DataFrame."""
    rows = trades_df[trades_df["Representative"] == politician]["Party"]
    return rows.iloc[0] if not rows.empty else "Unknown"


def _empty_fig(message: str) -> go.Figure:
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


# ── Callback ───────────────────────────────────────────────────────────────────

@callback(
    Output("network-chart", "figure"),
    Input("store-filtered-trades", "data"),
)
def update_network(store_data: str):
    """Redraw the network when filters change."""
    if not store_data:
        return _empty_fig("No data.")
    df = pd.read_json(store_data, orient="split")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["AmountMidpoint"]  = pd.to_numeric(df["AmountMidpoint"], errors="coerce").fillna(0)
    return make_network_figure(df)
