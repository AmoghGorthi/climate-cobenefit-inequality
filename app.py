
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Climate Equity Explorer",
    page_icon="🌍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MASTER_PATH = BASE_DIR / "master_analysis.parquet"
ANNUAL_PATH = BASE_DIR / "annual_total.parquet"

PATHWAYS = [
    "air_quality",
    "congestion",
    "dampness",
    "diet_change",
    "excess_cold",
    "excess_heat",
    "hassle_costs",
    "noise",
    "physical_activity",
    "road_repairs",
    "road_safety",
]

LABELS = {
    "air_quality": "Air quality",
    "congestion": "Congestion",
    "dampness": "Dampness",
    "diet_change": "Diet change",
    "excess_cold": "Excess cold",
    "excess_heat": "Excess heat",
    "hassle_costs": "Hassle costs",
    "noise": "Noise",
    "physical_activity": "Physical activity",
    "road_repairs": "Road repairs",
    "road_safety": "Road safety",
}

REGION_VIEWS = {
    "North East": {"lat": 54.92, "lon": -1.80, "zoom": 6.2},
    "North West": {"lat": 54.15, "lon": -2.65, "zoom": 5.7},
    "Yorkshire and The Humber": {
        "lat": 53.90,
        "lon": -1.35,
        "zoom": 5.8,
    },
    "East Midlands": {"lat": 52.88, "lon": -1.05, "zoom": 5.8},
    "West Midlands": {"lat": 52.52, "lon": -2.05, "zoom": 6.0},
    "East of England": {"lat": 52.25, "lon": 0.40, "zoom": 5.7},
    "London": {"lat": 51.51, "lon": -0.10, "zoom": 8.0},
    "South East": {"lat": 51.30, "lon": 0.05, "zoom": 5.7},
    "South West": {"lat": 50.85, "lon": -3.35, "zoom": 5.5},
}

COLOUR_LOW = "#b2182b"
COLOUR_HIGH = "#2166ac"
INK = "#17324d"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    h1 {
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        color: #17324d;
    }

    .subtitle {
        color: #4d5f70;
        font-size: 1.08rem;
        line-height: 1.6;
        margin-bottom: 1.3rem;
    }

    .definition {
        background: #f5f7fa;
        border-left: 4px solid #2166ac;
        border-radius: 3px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0 1rem 0;
        color: #33485c;
        line-height: 1.55;
    }

    .selected-area {
        background: #f8fafc;
        border: 1px solid #dce4eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e1e7ec;
        border-radius: 8px;
        padding: 0.75rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_master():
    data = pd.read_parquet(MASTER_PATH)

    expected = {
        "small_area",
        "lsoa_name",
        "lad_name",
        "region",
        "imd_score",
        "imd_decile",
        "sum",
        "peer_median",
        "peer_gap",
        "peer_percentile",
        *PATHWAYS,
    }

    missing = expected - set(data.columns)

    if missing:
        raise ValueError(
            f"master_analysis.parquet is missing: {sorted(missing)}"
        )

    return data


@st.cache_data
def load_annual():
    if not ANNUAL_PATH.exists():
        return None

    return pd.read_parquet(ANNUAL_PATH)


@st.cache_resource
def load_region_geojson(region):
    filename = region.replace(" ", "_") + ".geojson"
    path = BASE_DIR / "geo" / filename

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


master = load_master()
annual = load_annual()


# ============================================================
# HELPERS
# ============================================================

def style_figure(
    figure,
    height=430,
    show_legend=False,
):
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#33485c",
        ),
        title_font=dict(
            size=16,
            color=INK,
        ),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    figure.update_xaxes(
        gridcolor="#edf1f4",
        zerolinecolor="#ccd5dd",
    )

    figure.update_yaxes(
        gridcolor="#edf1f4",
        zerolinecolor="#ccd5dd",
    )

    return figure


def rank_label(value):
    if pd.isna(value):
        return "Not available"

    return f"{value:.0f}th percentile"


def scenario_metrics(data, pathways):
    output = data.copy()

    output["scenario_total"] = (
        output[pathways].sum(axis=1)
        if pathways
        else 0.0
    )

    peer_keys = ["region", "imd_decile"]

    output["scenario_peer_median"] = (
        output.groupby(peer_keys)["scenario_total"]
        .transform("median")
    )

    output["scenario_peer_gap"] = (
        output["scenario_total"]
        - output["scenario_peer_median"]
    )

    output["scenario_peer_percentile"] = (
        output.groupby(peer_keys)["scenario_total"]
        .rank(method="average", pct=True)
        .mul(100)
    )

    return output


def add_selected_outline(
    figure,
    geojson,
    selected_code,
):
    figure.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=[selected_code],
            z=[1],
            featureidkey="properties.small_area",
            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [1, "rgba(0,0,0,0)"],
            ],
            marker_opacity=0,
            marker_line_width=3,
            marker_line_color="#111111",
            showscale=False,
            hoverinfo="skip",
        )
    )

    return figure


def region_map(
    data,
    region,
    selected_code,
    value_column,
    title,
):
    geojson = load_region_geojson(region)
    view = REGION_VIEWS[region]

    maximum = data[value_column].abs().quantile(0.98)

    if maximum == 0 or pd.isna(maximum):
        maximum = 1

    figure = px.choropleth_mapbox(
        data,
        geojson=geojson,
        locations="small_area",
        featureidkey="properties.small_area",
        color=value_column,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        range_color=(-maximum, maximum),
        mapbox_style="carto-positron",
        center={
            "lat": view["lat"],
            "lon": view["lon"],
        },
        zoom=view["zoom"],
        opacity=0.82,
        hover_name="lsoa_name",
        hover_data={
            "small_area": True,
            "lad_name": True,
            "imd_decile": True,
            "sum": ":.2f",
            "peer_gap": ":.2f",
            value_column: ":.2f",
        },
        title=title,
    )

    figure.update_traces(
        marker_line_width=0.15,
        marker_line_color="white",
    )

    add_selected_outline(
        figure,
        geojson,
        selected_code,
    )

    figure.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=48, b=0),
        coloraxis_colorbar=dict(
            title="",
            thickness=13,
            len=0.65,
        ),
    )

    return figure


# ============================================================
# HEADER
# ============================================================

st.title("Climate Equity Explorer")

st.markdown(
    """
    <div class="subtitle">
    Find neighbourhoods with comparatively low projected climate-action
    co-benefits, compare them with similar places and test which pathways
    explain the difference.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Analysis controls")

regions = sorted(master["region"].unique())

selected_region = st.sidebar.selectbox(
    "Region",
    regions,
    index=regions.index("North East")
    if "North East" in regions
    else 0,
)

region_data = master[
    master["region"].eq(selected_region)
].copy()

local_authorities = sorted(
    region_data["lad_name"].unique()
)

selected_lad = st.sidebar.selectbox(
    "Local authority",
    ["All local authorities", *local_authorities],
)

selected_deciles = st.sidebar.multiselect(
    "IMD deciles",
    options=list(range(1, 11)),
    default=[1, 2, 3],
    format_func=lambda value: (
        f"{value} — most deprived"
        if value == 1
        else f"{value} — least deprived"
        if value == 10
        else str(value)
    ),
)

low_peer_threshold = st.sidebar.slider(
    "Low peer-benefit threshold",
    min_value=5,
    max_value=50,
    value=25,
    step=5,
    help=(
        "An area is classed as comparatively low when its benefit "
        "falls at or below this percentile among areas in the same "
        "region and IMD decile."
    ),
)

if not selected_deciles:
    st.warning("Select at least one IMD decile.")
    st.stop()

display_data = region_data[
    region_data["imd_decile"].isin(selected_deciles)
].copy()

if selected_lad != "All local authorities":
    display_data = display_data[
        display_data["lad_name"].eq(selected_lad)
    ].copy()

if display_data.empty:
    st.warning("No neighbourhoods match the current controls.")
    st.stop()

display_data["low_peer_group"] = (
    display_data["peer_percentile"]
    <= low_peer_threshold
)

display_data = display_data.sort_values(
    ["peer_gap", "peer_percentile"]
)

area_labels = {
    (
        f"{row.lsoa_name} — {row.small_area}"
    ): row.small_area
    for row in display_data.itertuples()
}

default_label = next(iter(area_labels))

selected_label = st.sidebar.selectbox(
    "Selected neighbourhood",
    options=list(area_labels),
    index=0,
)

selected_code = area_labels[selected_label]

selected_row = master[
    master["small_area"].eq(selected_code)
].iloc[0]

st.sidebar.markdown("---")

st.sidebar.caption(
    "Peer group: same English region and same IMD decile."
)


# ============================================================
# SELECTED AREA BANNER
# ============================================================

st.markdown(
    f"""
    <div class="selected-area">
    <b>Selected neighbourhood:</b> {selected_row['lsoa_name']}
    &nbsp;·&nbsp; {selected_row['small_area']}
    &nbsp;·&nbsp; {selected_row['lad_name']}
    &nbsp;·&nbsp; IMD decile {int(selected_row['imd_decile'])}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

find_tab, explain_tab, test_tab = st.tabs(
    [
        "1. Find an area",
        "2. Explain the difference",
        "3. Test pathways",
    ]
)


# ============================================================
# TAB 1 — FIND
# ============================================================

with find_tab:
    st.subheader("Find comparatively low-benefit neighbourhoods")

    st.markdown(
        f"""
        <div class="definition">
        Areas below the <b>{low_peer_threshold}th peer percentile</b>
        are highlighted as comparatively low. The comparison is made
        against neighbourhoods in the same region and IMD decile,
        rather than against unlike places across England.
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidate_count = int(
        display_data["low_peer_group"].sum()
    )

    negative_count = int(
        (display_data["sum"] < 0).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Neighbourhoods shown",
        f"{len(display_data):,}",
    )

    col2.metric(
        "Comparatively low",
        f"{candidate_count:,}",
    )

    col3.metric(
        "Negative aggregate value",
        f"{negative_count:,}",
    )

    col4.metric(
        "Selected peer percentile",
        rank_label(selected_row["peer_percentile"]),
    )

    map_col, scatter_col = st.columns(
        [1.35, 1],
        gap="large",
    )

    with map_col:
        map_figure = region_map(
            data=display_data,
            region=selected_region,
            selected_code=selected_code,
            value_column="peer_gap",
            title="Benefit difference from the peer median",
        )

        st.plotly_chart(
            map_figure,
            use_container_width=True,
        )

        st.caption(
            "Blue areas are above their peer median; red areas are "
            "below it. The black outline marks the selected area."
        )

    with scatter_col:
        scatter = px.scatter(
            display_data,
            x="imd_score",
            y="sum",
            color="peer_gap",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            hover_name="lsoa_name",
            hover_data={
                "small_area": True,
                "lad_name": True,
                "imd_decile": True,
                "peer_gap": ":.2f",
                "peer_percentile": ":.1f",
                "population": ":,",
            },
            title="Deprivation score and projected total benefit",
        )

        scatter.add_trace(
            go.Scatter(
                x=[selected_row["imd_score"]],
                y=[selected_row["sum"]],
                mode="markers",
                marker=dict(
                    size=16,
                    color="#111111",
                    symbol="diamond-open",
                    line=dict(width=3),
                ),
                name="Selected area",
                hovertemplate=(
                    f"{selected_row['lsoa_name']}"
                    "<extra>Selected area</extra>"
                ),
            )
        )

        scatter.update_layout(
            coloraxis_colorbar=dict(
                title="Peer gap",
                thickness=12,
            )
        )

        scatter.update_xaxes(
            title="IMD score — higher means more deprived"
        )

        scatter.update_yaxes(
            title="Projected aggregate co-benefit"
        )

        st.plotly_chart(
            style_figure(
                scatter,
                height=610,
                show_legend=True,
            ),
            use_container_width=True,
        )

    st.markdown("### Ranked neighbourhood table")

    table = display_data[
        [
            "lsoa_name",
            "small_area",
            "lad_name",
            "imd_decile",
            "sum",
            "peer_median",
            "peer_gap",
            "peer_percentile",
            "dominant_deficit_pathway",
        ]
    ].copy()

    table["dominant_deficit_pathway"] = (
        table["dominant_deficit_pathway"]
        .map(LABELS)
        .fillna("No negative pathway difference")
    )

    table = table.rename(
        columns={
            "lsoa_name": "Neighbourhood",
            "small_area": "LSOA code",
            "lad_name": "Local authority",
            "imd_decile": "IMD decile",
            "sum": "Total benefit",
            "peer_median": "Peer median",
            "peer_gap": "Peer gap",
            "peer_percentile": "Peer percentile",
            "dominant_deficit_pathway": "Largest pathway deficit",
        }
    )

    st.dataframe(
        table.head(250),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total benefit": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Peer median": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Peer gap": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Peer percentile": st.column_config.NumberColumn(
                format="%.1f"
            ),
        },
    )

    st.caption(
        "The table is ordered from the largest negative peer gap "
        "to the largest positive peer gap."
    )


# ============================================================
# TAB 2 — EXPLAIN
# ============================================================

with explain_tab:
    st.subheader("Explain why the selected area differs")

    peer_group = master[
        master["region"].eq(selected_row["region"])
        & master["imd_decile"].eq(
            selected_row["imd_decile"]
        )
    ]

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Selected total",
        f"{selected_row['sum']:.2f}",
    )

    metric2.metric(
        "Peer median",
        f"{selected_row['peer_median']:.2f}",
    )

    metric3.metric(
        "Difference from peers",
        f"{selected_row['peer_gap']:+.2f}",
    )

    metric4.metric(
        "Peer percentile",
        rank_label(selected_row["peer_percentile"]),
    )

    residual_data = pd.DataFrame(
        {
            "pathway": PATHWAYS,
            "selected_value": [
                selected_row[pathway]
                for pathway in PATHWAYS
            ],
            "peer_median": [
                peer_group[pathway].median()
                for pathway in PATHWAYS
            ],
        }
    )

    residual_data["peer_difference"] = (
        residual_data["selected_value"]
        - residual_data["peer_median"]
    )

    residual_data["label"] = (
        residual_data["pathway"].map(LABELS)
    )

    residual_data = residual_data.sort_values(
        "peer_difference"
    )

    residual_col, raw_col = st.columns(
        2,
        gap="large",
    )

    with residual_col:
        residual_figure = go.Figure(
            go.Bar(
                x=residual_data["peer_difference"],
                y=residual_data["label"],
                orientation="h",
                marker_color=[
                    COLOUR_LOW if value < 0 else COLOUR_HIGH
                    for value in residual_data[
                        "peer_difference"
                    ]
                ],
                customdata=residual_data[
                    [
                        "selected_value",
                        "peer_median",
                    ]
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Difference: %{x:.3f}<br>"
                    "Selected: %{customdata[0]:.3f}<br>"
                    "Peer median: %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        residual_figure.add_vline(
            x=0,
            line_color="#8997a5",
            line_width=1,
        )

        residual_figure.update_layout(
            title="Pathway difference from comparable areas",
            xaxis_title="Selected value − peer median",
            yaxis_title="",
        )

        st.plotly_chart(
            style_figure(
                residual_figure,
                height=500,
            ),
            use_container_width=True,
        )

    with raw_col:
        raw_data = residual_data.sort_values(
            "selected_value"
        )

        raw_figure = go.Figure(
            go.Bar(
                x=raw_data["selected_value"],
                y=raw_data["label"],
                orientation="h",
                marker_color=[
                    COLOUR_LOW if value < 0 else COLOUR_HIGH
                    for value in raw_data["selected_value"]
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Value: %{x:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        raw_figure.add_vline(
            x=0,
            line_color="#8997a5",
            line_width=1,
        )

        raw_figure.update_layout(
            title="Raw pathway composition",
            xaxis_title="Projected co-benefit contribution",
            yaxis_title="",
        )

        st.plotly_chart(
            style_figure(
                raw_figure,
                height=500,
            ),
            use_container_width=True,
        )

    dominant_pathway = selected_row[
        "dominant_deficit_pathway"
    ]

    if dominant_pathway != "none":
        dominant_label = LABELS[dominant_pathway]
        dominant_value = selected_row[
            "dominant_deficit_value"
        ]

        st.info(
            f"The largest negative difference from comparable areas "
            f"is {dominant_label.lower()} "
            f"({dominant_value:+.2f} relative to the peer median)."
        )
    else:
        st.info(
            "The selected area is not below its peer median on any "
            "individual pathway."
        )

    if annual is not None:
        st.markdown("### Annual comparison, 2025–2050")

        selected_annual = annual[
            annual["small_area"].eq(selected_code)
        ].sort_values("year")

        years_below = int(
            (selected_annual["annual_peer_gap"] < 0).sum()
        )

        persistence = (
            years_below / len(selected_annual) * 100
            if len(selected_annual)
            else 0
        )

        trend = go.Figure()

        trend.add_trace(
            go.Scatter(
                x=selected_annual["year"],
                y=selected_annual["annual_total"],
                mode="lines+markers",
                name="Selected neighbourhood",
                line=dict(
                    color=COLOUR_LOW,
                    width=2.5,
                ),
            )
        )

        trend.add_trace(
            go.Scatter(
                x=selected_annual["year"],
                y=selected_annual[
                    "annual_peer_median"
                ],
                mode="lines",
                name="Peer median",
                line=dict(
                    color=COLOUR_HIGH,
                    width=2.5,
                    dash="dot",
                ),
            )
        )

        trend.add_hline(
            y=0,
            line_color="#aab4bd",
            line_width=1,
        )

        trend.update_layout(
            title="Selected neighbourhood versus its annual peer median",
            xaxis_title="Year",
            yaxis_title="Annual projected co-benefit",
        )

        st.plotly_chart(
            style_figure(
                trend,
                height=430,
                show_legend=True,
            ),
            use_container_width=True,
        )

        st.caption(
            f"The selected area is below its peer median in "
            f"{years_below} of {len(selected_annual)} years "
            f"({persistence:.0f}% of the projection period)."
        )


# ============================================================
# TAB 3 — TEST PATHWAYS
# ============================================================

with test_tab:
    st.subheader("Test whether pathway choices change the result")

    st.markdown(
        """
        Remove one or more pathways and compare the modified scenario
        with the complete eleven-pathway baseline. All peer medians,
        percentiles and classifications are recalculated.
        """
    )

    included_pathways = st.multiselect(
        "Included pathways in the modified scenario",
        options=PATHWAYS,
        default=PATHWAYS,
        format_func=lambda value: LABELS[value],
    )

    if not included_pathways:
        st.warning("Select at least one pathway.")
    else:
        scenario_all = scenario_metrics(
            master,
            included_pathways,
        )

        scenario_selected = scenario_all[
            scenario_all["small_area"].eq(selected_code)
        ].iloc[0]

        baseline_low = (
            selected_row["peer_percentile"]
            <= low_peer_threshold
        )

        scenario_low = (
            scenario_selected["scenario_peer_percentile"]
            <= low_peer_threshold
        )

        comparison1, comparison2, comparison3, comparison4 = (
            st.columns(4)
        )

        comparison1.metric(
            "Selected total",
            f"{scenario_selected['scenario_total']:.2f}",
            delta=(
                f"{scenario_selected['scenario_total'] - selected_row['sum']:+.2f}"
                " vs baseline"
            ),
        )

        comparison2.metric(
            "Peer gap",
            f"{scenario_selected['scenario_peer_gap']:+.2f}",
            delta=(
                f"{scenario_selected['scenario_peer_gap'] - selected_row['peer_gap']:+.2f}"
                " vs baseline"
            ),
        )

        comparison3.metric(
            "Peer percentile",
            f"{scenario_selected['scenario_peer_percentile']:.1f}",
            delta=(
                f"{scenario_selected['scenario_peer_percentile'] - selected_row['peer_percentile']:+.1f}"
                " points"
            ),
        )

        comparison4.metric(
            "Low peer-benefit group",
            "Yes" if scenario_low else "No",
            delta=(
                "No change"
                if baseline_low == scenario_low
                else "Classification changed"
            ),
        )

        region_scenario = scenario_all[
            scenario_all["region"].eq(selected_region)
        ].copy()

        baseline_decile = (
            region_scenario.groupby("imd_decile")["sum"]
            .mean()
            .rename("Baseline")
        )

        scenario_decile = (
            region_scenario.groupby(
                "imd_decile"
            )["scenario_total"]
            .mean()
            .rename("Modified scenario")
        )

        decile_comparison = pd.concat(
            [
                baseline_decile,
                scenario_decile,
            ],
            axis=1,
        ).reset_index()

        decile_long = decile_comparison.melt(
            id_vars="imd_decile",
            var_name="Scenario",
            value_name="Mean benefit",
        )

        gradient_figure = px.line(
            decile_long,
            x="imd_decile",
            y="Mean benefit",
            color="Scenario",
            markers=True,
            title=(
                f"Deprivation pattern in {selected_region}: "
                "baseline versus modified scenario"
            ),
            color_discrete_map={
                "Baseline": COLOUR_HIGH,
                "Modified scenario": COLOUR_LOW,
            },
        )

        gradient_figure.update_xaxes(
            dtick=1,
            title="IMD decile — 1 is most deprived",
        )

        gradient_figure.update_yaxes(
            title="Mean projected co-benefit",
        )

        st.plotly_chart(
            style_figure(
                gradient_figure,
                height=430,
                show_legend=True,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CLASSIFICATION CHANGE MAP
        # ----------------------------------------------------

        scenario_display = scenario_all[
            scenario_all["region"].eq(selected_region)
            & scenario_all["imd_decile"].isin(
                selected_deciles
            )
        ].copy()

        if selected_lad != "All local authorities":
            scenario_display = scenario_display[
                scenario_display["lad_name"].eq(
                    selected_lad
                )
            ].copy()

        scenario_display["baseline_low"] = (
            scenario_display["peer_percentile"]
            <= low_peer_threshold
        )

        scenario_display["scenario_low"] = (
            scenario_display[
                "scenario_peer_percentile"
            ]
            <= low_peer_threshold
        )

        def classify_change(row):
            if row["baseline_low"] and row["scenario_low"]:
                return "Remains comparatively low"

            if not row["baseline_low"] and row["scenario_low"]:
                return "Enters comparatively low group"

            if row["baseline_low"] and not row["scenario_low"]:
                return "Leaves comparatively low group"

            return "Never comparatively low"

        scenario_display["classification_change"] = (
            scenario_display.apply(
                classify_change,
                axis=1,
            )
        )

        st.markdown("### Which neighbourhoods change classification?")

        change_counts = (
            scenario_display["classification_change"]
            .value_counts()
        )

        count_columns = st.columns(4)

        category_order = [
            "Remains comparatively low",
            "Enters comparatively low group",
            "Leaves comparatively low group",
            "Never comparatively low",
        ]

        for column, category in zip(
            count_columns,
            category_order,
        ):
            column.metric(
                category,
                f"{int(change_counts.get(category, 0)):,}",
            )

        geojson = load_region_geojson(selected_region)
        view = REGION_VIEWS[selected_region]

        change_figure = px.choropleth_mapbox(
            scenario_display,
            geojson=geojson,
            locations="small_area",
            featureidkey="properties.small_area",
            color="classification_change",
            category_orders={
                "classification_change": category_order
            },
            color_discrete_map={
                "Remains comparatively low": "#8b1a1a",
                "Enters comparatively low group": "#e67e22",
                "Leaves comparatively low group": "#2a9d8f",
                "Never comparatively low": "#d8dee4",
            },
            mapbox_style="carto-positron",
            center={
                "lat": view["lat"],
                "lon": view["lon"],
            },
            zoom=view["zoom"],
            opacity=0.85,
            hover_name="lsoa_name",
            hover_data={
                "small_area": True,
                "lad_name": True,
                "imd_decile": True,
                "peer_percentile": ":.1f",
                "scenario_peer_percentile": ":.1f",
            },
            title="Baseline-to-scenario classification change",
        )

        change_figure.update_traces(
            marker_line_width=0.15,
            marker_line_color="white",
        )

        add_selected_outline(
            change_figure,
            geojson,
            selected_code,
        )

        change_figure.update_layout(
            height=610,
            margin=dict(l=0, r=0, t=48, b=0),
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(
            change_figure,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # PATHWAY SENSITIVITY
        # ----------------------------------------------------

        st.markdown("### Sensitivity of the regional D10–D1 gap")

        region_master = master[
            master["region"].eq(selected_region)
        ].copy()

        baseline_means = (
            region_master.groupby("imd_decile")["sum"]
            .mean()
        )

        if 1 in baseline_means.index and 10 in baseline_means.index:
            baseline_gap = (
                baseline_means.loc[10]
                - baseline_means.loc[1]
            )

            sensitivity_rows = []

            for pathway in PATHWAYS:
                remaining = [
                    item
                    for item in PATHWAYS
                    if item != pathway
                ]

                temporary = region_master[
                    ["imd_decile"]
                ].copy()

                temporary["modified_total"] = (
                    region_master[remaining].sum(axis=1)
                )

                means = (
                    temporary.groupby(
                        "imd_decile"
                    )["modified_total"]
                    .mean()
                )

                modified_gap = (
                    means.loc[10]
                    - means.loc[1]
                )

                sensitivity_rows.append(
                    {
                        "pathway": LABELS[pathway],
                        "gap_without_pathway": modified_gap,
                        "change_from_baseline": (
                            modified_gap - baseline_gap
                        ),
                    }
                )

            sensitivity = pd.DataFrame(
                sensitivity_rows
            )

            sensitivity["absolute_change"] = (
                sensitivity[
                    "change_from_baseline"
                ].abs()
            )

            sensitivity = sensitivity.sort_values(
                "absolute_change"
            )

            sensitivity_figure = go.Figure(
                go.Bar(
                    x=sensitivity[
                        "change_from_baseline"
                    ],
                    y=sensitivity["pathway"],
                    orientation="h",
                    marker_color=[
                        COLOUR_LOW if value < 0 else COLOUR_HIGH
                        for value in sensitivity[
                            "change_from_baseline"
                        ]
                    ],
                    customdata=sensitivity[
                        ["gap_without_pathway"]
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Change in gap: %{x:+.3f}<br>"
                        "Gap after removal: "
                        "%{customdata[0]:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

            sensitivity_figure.add_vline(
                x=0,
                line_color="#8997a5",
                line_width=1,
            )

            sensitivity_figure.update_layout(
                title=(
                    "Change in the D10–D1 gap when each pathway "
                    "is removed"
                ),
                xaxis_title=(
                    "Modified gap − complete baseline gap"
                ),
                yaxis_title="",
            )

            st.plotly_chart(
                style_figure(
                    sensitivity_figure,
                    height=500,
                ),
                use_container_width=True,
            )

            st.caption(
                f"The complete eleven-pathway D10–D1 gap in "
                f"{selected_region} is {baseline_gap:.2f}. "
                "Large negative values indicate that removing the "
                "pathway substantially reduces the observed gap."
            )


# ============================================================
# METHODS
# ============================================================

st.markdown("---")

with st.expander("Definitions, data and limitations"):
    st.markdown(
        """
        **Unit of analysis**
        Lower Layer Super Output Areas using 2011 geography.

        **Peer group**
        Neighbourhoods in the same English region and the same
        Index of Multiple Deprivation decile.

        **Peer gap**
        Selected neighbourhood value minus the median value of its
        peer group. A negative value means the neighbourhood is below
        comparable places.

        **Peer percentile**
        Rank within the regional deprivation peer group. A low
        percentile indicates a comparatively low projected benefit.

        **Modified scenario**
        The sum of pathways selected by the user. It is an analytical
        sensitivity test, not a prediction of a different government
        policy.

        **Interpretation limits**
        The data contains modelled projected monetised co-benefits,
        not observed outcomes, investment allocations or causal
        effects. IMD is an external 2019 neighbourhood measure.
        LSOA averages do not describe every resident. No uncertainty
        intervals are available, and spatial autocorrelation is not
        modelled.

        **Prototype scope**
        The tool supports exploratory comparison and sensitivity
        analysis. It does not recommend where funding should be
        allocated.
        """
    )

st.caption(
    "Amogh Gorthi · MSc Data Science and AI · Newcastle University"
)
