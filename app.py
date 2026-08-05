from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Climate Equity Explorer", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).resolve().parent
MASTER_PATH = BASE / "master_analysis.parquet"
ANNUAL_PATH = BASE / "annual_total.parquet"
GEO_DIR = BASE / "geo"

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

REGION_FILES = {
    "East Midlands": "East_Midlands.geojson",
    "East of England": "East_of_England.geojson",
    "London": "London.geojson",
    "North East": "North_East.geojson",
    "North West": "North_West.geojson",
    "South East": "South_East.geojson",
    "South West": "South_West.geojson",
    "West Midlands": "West_Midlands.geojson",
    "Yorkshire and The Humber": "Yorkshire_and_The_Humber.geojson",
}

REGION_VIEW = {
    "East Midlands": (52.9, -0.7, 5.8),
    "East of England": (52.2, 0.5, 5.7),
    "London": (51.51, -0.12, 8.0),
    "North East": (54.9, -1.9, 6.1),
    "North West": (54.2, -2.6, 5.7),
    "South East": (51.3, -0.4, 5.7),
    "South West": (50.8, -3.3, 5.5),
    "West Midlands": (52.5, -2.1, 6.0),
    "Yorkshire and The Humber": (53.9, -1.3, 5.8),
}

PRIMARY = "#1F4E79"
SECONDARY = "#D97706"
SELECTED = "#111827"
POSITIVE = "#0072B2"
NEGATIVE = "#D55E00"
NEUTRAL = "#F3F4F6"
TEXT = "#17202A"
MUTED = "#667085"
PLOT_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": False}

st.markdown(
    """
<style>
:root{--ink:#17202A;--muted:#667085;--line:#E5E7EB;--panel:#F8FAFC;--navy:#1F4E79;--orange:#D97706}
.block-container{max-width:1500px;padding-top:1rem;padding-bottom:3rem}
[data-testid="stSidebar"]{border-right:1px solid var(--line)}
h1{font-size:2rem!important;letter-spacing:-.03em;margin-bottom:.2rem!important}
h2{font-size:1.35rem!important;letter-spacing:-.015em;margin-top:1.4rem!important}
h3{font-size:1.05rem!important;margin-top:1rem!important}
.small{font-size:.82rem;color:var(--muted);line-height:1.45}
.lede{font-size:1.02rem;color:#344054;max-width:1050px;line-height:1.55;margin-bottom:.8rem}
.notice{background:#FFF8E8;border:1px solid #F3D69A;border-radius:10px;padding:.72rem .9rem;color:#5A4513;font-size:.85rem;line-height:1.45;margin:.65rem 0 1rem}
.statebar{display:flex;flex-wrap:wrap;gap:.42rem;margin:.6rem 0 1rem}
.pill{display:inline-flex;align-items:center;border:1px solid #D0D5DD;background:white;border-radius:999px;padding:.34rem .62rem;font-size:.78rem;color:#344054}
.pill b{color:#101828;margin-right:.25rem}
.cardgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.65rem;margin:.55rem 0 1rem}
.metriccard{border:1px solid #E4E7EC;background:white;border-radius:12px;padding:.78rem .85rem;min-height:96px;box-shadow:0 1px 2px rgba(16,24,40,.03)}
.metriclabel{font-size:.76rem;color:#667085;margin-bottom:.35rem}
.metricvalue{font-size:1.48rem;font-weight:700;color:#101828;line-height:1.08}
.metricsub{font-size:.72rem;color:#667085;margin-top:.35rem;line-height:1.35}
.question{border-left:4px solid #1F4E79;padding:.15rem .75rem;margin:.25rem 0 1rem;color:#344054;font-size:.95rem}
.insight{background:#F8FAFC;border:1px solid #D8E1EA;border-radius:12px;padding:.85rem 1rem;margin:.8rem 0;line-height:1.5;color:#25364A}
.progress{display:flex;gap:.35rem;align-items:center;flex-wrap:wrap;margin:.3rem 0 .9rem}
.step{padding:.32rem .58rem;border-radius:8px;background:#F2F4F7;color:#667085;font-size:.76rem;border:1px solid #EAECF0}
.step.active{background:#EAF1F8;color:#153B5C;border-color:#B9CDE0;font-weight:700}
hr{border:none;border-top:1px solid #EAECF0;margin:1.1rem 0}
[data-testid="stDataFrame"]{border:1px solid #E4E7EC;border-radius:10px;overflow:hidden}
.stButton>button,.stDownloadButton>button{border-radius:9px!important}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_master():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(str(MASTER_PATH))
    frame = pd.read_parquet(MASTER_PATH)
    frame["imd_decile"] = frame["imd_decile"].astype(int)
    return frame

@st.cache_data(show_spinner=False)
def load_annual():
    if not ANNUAL_PATH.exists():
        raise FileNotFoundError(str(ANNUAL_PATH))
    return pd.read_parquet(ANNUAL_PATH)

@st.cache_resource(show_spinner=False)
def load_geo(region):
    path = GEO_DIR / REGION_FILES[region]
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

@st.cache_data(show_spinner=False)
def annual_with_context():
    annual = load_annual()
    context = load_master()[["small_area", "region", "imd_decile"]]
    return annual.merge(context, on="small_area", how="left", validate="many_to_one")

def clean(value):
    if value is None or pd.isna(value):
        return "—"
    return html.escape(str(value))

def number(value, decimals=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"

def count(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"

def signed(value, decimals=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+,.{decimals}f}"

def metric_cards(items):
    cards = []
    for label, value, sub in items:
        cards.append(
            f"<div class='metriccard'><div class='metriclabel'>{html.escape(label)}</div>"
            f"<div class='metricvalue'>{html.escape(str(value))}</div>"
            f"<div class='metricsub'>{html.escape(str(sub))}</div></div>"
        )
    st.markdown("<div class='cardgrid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

def style_figure(fig, height=430, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=55, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color=TEXT),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor="#EEF2F6", zerolinecolor="#98A2B3")
    fig.update_yaxes(gridcolor="#EEF2F6", zerolinecolor="#98A2B3")
    return fig

def robust_range(series, diverging=False):
    values = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return (-1, 1) if diverging else (0, 1)
    low, high = values.quantile([0.02, 0.98]).tolist()
    if diverging:
        bound = max(abs(low), abs(high), 1e-9)
        return -bound, bound
    if low == high:
        low, high = values.min(), values.max()
    if low == high:
        high = low + 1
    return float(low), float(high)

def region_view(region):
    lat, lon, zoom = REGION_VIEW[region]
    return {"lat": lat, "lon": lon}, zoom

def numeric_map(frame, region, value, title, label, colorscale, midpoint=None, selected_code=None, shared_range=None):
    geo = load_geo(region)
    center, zoom = region_view(region)
    data = frame.copy()
    value_range = shared_range or robust_range(data[value], diverging=midpoint is not None)
    hover = {
        "small_area": False,
        "imd_decile": True,
        "sum": ":.2f",
        "peer_gap": ":.2f" if "peer_gap" in data.columns else False,
        value: ":.2f",
    }
    hover = {key: val for key, val in hover.items() if key in data.columns}
    fig = px.choropleth_mapbox(
        data,
        geojson=geo,
        locations="small_area",
        featureidkey="properties.small_area",
        color=value,
        color_continuous_scale=colorscale,
        range_color=value_range,
        color_continuous_midpoint=midpoint,
        hover_name="lsoa_name" if "lsoa_name" in data.columns else None,
        hover_data=hover,
        labels={value: label, "imd_decile": "IMD decile", "sum": "Projected total", "peer_gap": "Peer gap"},
        mapbox_style="carto-positron",
        center=center,
        zoom=zoom,
        opacity=0.75,
        title=title,
    )
    if selected_code and selected_code in set(data["small_area"]):
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geo,
                locations=[selected_code],
                z=[1],
                featureidkey="properties.small_area",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_opacity=0.01,
                marker_line_color=SELECTED,
                marker_line_width=3,
                hoverinfo="skip",
            )
        )
    fig.update_layout(coloraxis_colorbar=dict(title=label, thickness=14, len=0.72))
    return style_figure(fig, 500, False), value_range

def category_map(frame, region, selected_code=None):
    geo = load_geo(region)
    center, zoom = region_view(region)
    order = ["Remains low", "Enters low", "Leaves low", "Never low"]
    colors = {
        "Remains low": "#7F3C8D",
        "Enters low": "#D55E00",
        "Leaves low": "#0072B2",
        "Never low": "#B8B8B8",
    }
    fig = px.choropleth_mapbox(
        frame,
        geojson=geo,
        locations="small_area",
        featureidkey="properties.small_area",
        color="classification_change",
        category_orders={"classification_change": order},
        color_discrete_map=colors,
        hover_name="lsoa_name",
        hover_data={
            "small_area": False,
            "imd_decile": True,
            "peer_percentile": ":.1f",
            "scenario_percentile": ":.1f",
            "classification_change": True,
        },
        labels={
            "imd_decile": "IMD decile",
            "peer_percentile": "Baseline percentile",
            "scenario_percentile": "Scenario percentile",
            "classification_change": "Classification",
        },
        mapbox_style="carto-positron",
        center=center,
        zoom=zoom,
        opacity=0.78,
        title="Where the bottom-quartile classification changes",
    )
    if selected_code and selected_code in set(frame["small_area"]):
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geo,
                locations=[selected_code],
                z=[1],
                featureidkey="properties.small_area",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_opacity=0.01,
                marker_line_color=SELECTED,
                marker_line_width=3,
                hoverinfo="skip",
            )
        )
    return style_figure(fig, 520, True)

def region_stats(frame):
    deciles = frame.groupby("imd_decile", as_index=False)["sum"].mean()
    d1 = deciles.loc[deciles["imd_decile"] == 1, "sum"]
    d10 = deciles.loc[deciles["imd_decile"] == 10, "sum"]
    d1 = float(d1.iloc[0]) if len(d1) else np.nan
    d10 = float(d10.iloc[0]) if len(d10) else np.nan
    gap = d10 - d1 if pd.notna(d1) and pd.notna(d10) else np.nan
    parity = d1 / d10 if pd.notna(d10) and d10 != 0 else np.nan
    corr = frame[["imd_score", "sum"]].corr(method="spearman").iloc[0, 1]
    return {
        "n": len(frame),
        "lads": frame["lad_name"].nunique(),
        "mean": frame["sum"].mean(),
        "median": frame["sum"].median(),
        "negative_share": 100 * (frame["sum"] < 0).mean(),
        "high_deprivation_share": 100 * frame["imd_decile"].isin([1, 2, 3]).mean(),
        "d1": d1,
        "d10": d10,
        "gap": gap,
        "parity": parity,
        "spearman": corr,
    }

def scenario_frame(master, selected_pathways):
    values = master[list(selected_pathways)].sum(axis=1)
    result = master[["small_area", "lsoa_name", "lad_name", "region", "imd_decile", "peer_percentile", "sum"]].copy()
    result["scenario_total"] = values
    groups = result.groupby(["region", "imd_decile"])["scenario_total"]
    result["scenario_peer_median"] = groups.transform("median")
    result["scenario_peer_gap"] = result["scenario_total"] - result["scenario_peer_median"]
    result["scenario_percentile"] = groups.rank(method="average", pct=True) * 100
    return result

def first_crossover(years, gaps):
    mask = np.asarray(gaps) >= 0
    indexes = np.where(mask)[0]
    return int(years.iloc[indexes[0]]) if len(indexes) else None

def sustained_crossover(years, gaps, span=3):
    mask = np.asarray(gaps) >= 0
    for start in range(0, len(mask) - span + 1):
        if mask[start:start + span].all():
            return int(years.iloc[start])
    return None

def dominant_deficit_text(row):
    pathway = row["dominant_deficit_pathway"]
    if pathway == "none":
        return "No pathway is below its peer median"
    return LABELS.get(pathway, pathway.replace("_", " ").title())

def interpretation_expander(title, task, benchmark, calculation, meaning, limitation):
    with st.expander("Design and interpretation"):
        st.markdown(
            f"**Why this view?** {title}\n\n"
            f"**User task.** {task}\n\n"
            f"**Benchmark.** {benchmark}\n\n"
            f"**Calculation.** {calculation}\n\n"
            f"**Interpretation.** {meaning}\n\n"
            f"**Limitation.** {limitation}"
        )

def reset_all():
    for key in [
        "primary_region",
        "comparison_region",
        "selected_deciles",
        "threshold",
        "lad_filter",
        "selected_area_label",
        "selected_code",
        "scenario_pathways",
        "outlook_benchmark",
        "page",
    ]:
        st.session_state.pop(key, None)

try:
    master = load_master()
    annual = annual_with_context()
except Exception as error:
    st.error(f"The dashboard could not load its data files: {error}")
    st.stop()

regions = sorted(master["region"].dropna().unique())
if "scenario_pathways" not in st.session_state:
    st.session_state["scenario_pathways"] = PATHWAYS.copy()

with st.sidebar:
    st.markdown("## Climate Equity Explorer")
    st.caption("Region-to-neighbourhood visual analytics")
    default_primary = regions.index("North East") if "North East" in regions else 0
    primary_region = st.selectbox("Primary region", regions, index=default_primary, key="primary_region")
    comparison_options = [region for region in regions if region != primary_region]
    if st.session_state.get("comparison_region") not in comparison_options:
        st.session_state["comparison_region"] = "South East" if "South East" in comparison_options else comparison_options[0]
    default_comparison = comparison_options.index(st.session_state["comparison_region"])
    comparison_region = st.selectbox("Comparison region", comparison_options, index=default_comparison, key="comparison_region")
    selected_deciles = st.multiselect("IMD deciles", list(range(1, 11)), default=list(range(1, 11)), key="selected_deciles")
    if not selected_deciles:
        st.warning("Select at least one IMD decile.")
        st.stop()
    threshold = st.slider("Comparatively low threshold", 5, 50, 25, 5, key="threshold")
    lad_options = ["All local authorities"] + sorted(master.loc[master["region"] == primary_region, "lad_name"].dropna().unique())
    if st.session_state.get("lad_filter") not in lad_options:
        st.session_state["lad_filter"] = "All local authorities"
    lad_filter = st.selectbox("Local authority", lad_options, key="lad_filter")
    area_pool = master[(master["region"] == primary_region) & (master["imd_decile"].isin(selected_deciles))].copy()
    if lad_filter != "All local authorities":
        area_pool = area_pool[area_pool["lad_name"] == lad_filter]
    area_pool = area_pool.sort_values(["lsoa_name", "small_area"])
    if area_pool.empty:
        st.warning("No neighbourhoods match the current filters.")
        st.stop()
    area_pool["area_label"] = area_pool["lsoa_name"] + " · " + area_pool["small_area"]
    labels = area_pool["area_label"].tolist()
    if st.session_state.get("selected_area_label") not in labels:
        st.session_state.pop("selected_area_label", None)
    current_code = st.session_state.get("selected_code")
    current_matches = area_pool.index[area_pool["small_area"] == current_code].tolist()
    area_index = area_pool.index.get_loc(current_matches[0]) if current_matches else 0
    selected_label = st.selectbox("Selected neighbourhood", labels, index=area_index, key="selected_area_label")
    selected_code = area_pool.loc[area_pool["area_label"] == selected_label, "small_area"].iloc[0]
    st.session_state["selected_code"] = selected_code
    st.button("Reset all controls", use_container_width=True, on_click=reset_all)
    st.markdown("<div class='small'>Decile 1 is most deprived. Decile 10 is least deprived.</div>", unsafe_allow_html=True)

selected_row = master.loc[master["small_area"] == selected_code].iloc[0]
scenario_pathways = st.session_state.get("scenario_pathways", PATHWAYS)
scenario_label = "Baseline · all 11 pathways" if len(scenario_pathways) == 11 else f"Modified · {len(scenario_pathways)} of 11 pathways"
benchmark_label = st.session_state.get("outlook_benchmark", "Same-region, same-decile peer median")

st.markdown("# Climate Equity Explorer")
st.markdown(
    "<div class='lede'>A coordinated visual analytics prototype for moving from regional context to a neighbourhood-level explanation and projected outlook.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='notice'><b>Modelled projection notice.</b> Values represent projected outcomes for 2025–2050 under the analysed climate pathway. They are not observed outcomes, guaranteed forecasts, spending allocations or causal estimates. Pathway controls alter the analytical composition of the aggregate measure and do not simulate specific investment decisions.</div>",
    unsafe_allow_html=True,
)

state_items = [
    ("Primary", primary_region),
    ("Comparison", comparison_region),
    ("Area", selected_row["lsoa_name"]),
    ("IMD", f"Decile {int(selected_row['imd_decile'])}"),
    ("Peer", f"{selected_row['peer_group_n']:,} region-decile LSOAs"),
    ("Scenario", scenario_label),
    ("Threshold", f"Bottom {threshold}%"),
    ("Outlook", benchmark_label),
]
st.markdown(
    "<div class='statebar'>" + "".join([f"<span class='pill'><b>{clean(k)}:</b>{clean(v)}</span>" for k, v in state_items]) + "</div>",
    unsafe_allow_html=True,
)

pages = [
    "1 · Region profile",
    "2 · Compare regions",
    "3 · Neighbourhood performance",
    "4 · Explain and test",
    "5 · Future outlook",
]
page = st.radio("Analysis stage", pages, horizontal=True, label_visibility="collapsed", key="page")
st.markdown(
    "<div class='progress'>" + "".join([f"<span class='step {'active' if item == page else ''}'>{html.escape(item)}</span>" for item in pages]) + "</div>",
    unsafe_allow_html=True,
)

primary_frame = master[master["region"] == primary_region].copy()
comparison_frame = master[master["region"] == comparison_region].copy()

if page == pages[0]:
    st.markdown("## What is the projected distribution in this region?")
    st.markdown("<div class='question'>Establish the regional context before comparing places or selecting a neighbourhood.</div>", unsafe_allow_html=True)
    stats = region_stats(primary_frame)
    england_stats = region_stats(master)
    metric_cards([
        ("Neighbourhoods", count(stats["n"]), f"Across {stats['lads']} local authorities"),
        ("Mean projected value", number(stats["mean"]), f"England: {number(england_stats['mean'])}"),
        ("Median projected value", number(stats["median"]), f"England: {number(england_stats['median'])}"),
        ("Negative-value share", f"{stats['negative_share']:.1f}%", "Share of regional LSOAs"),
        ("High-deprivation share", f"{stats['high_deprivation_share']:.1f}%", "IMD deciles 1–3"),
        ("D10–D1 difference", signed(stats["gap"]), f"Parity ratio D1/D10: {stats['parity']:.2f}"),
        ("Spearman association", f"{stats['spearman']:.3f}", "IMD score versus aggregate value"),
    ])
    left, right = st.columns([1.18, 1])
    with left:
        map_fig, map_range = numeric_map(
            primary_frame,
            primary_region,
            "sum",
            f"Projected aggregate value across {primary_region}",
            "Projected value",
            "YlGnBu",
            shared_range=robust_range(primary_frame["sum"]),
        )
        st.plotly_chart(map_fig, use_container_width=True, config=PLOT_CONFIG)
        st.caption(f"Display range is capped at the regional 2nd–98th percentiles ({map_range[0]:.2f} to {map_range[1]:.2f}); true values remain in tooltips. Polygon area represents geography, not population or magnitude.")
    with right:
        decile = primary_frame.groupby("imd_decile", as_index=False)["sum"].agg(["mean", "median"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decile["imd_decile"], y=decile["mean"], mode="lines+markers+text", text=decile["mean"].round(2), textposition="top center", name="Regional mean", line=dict(color=PRIMARY, width=3), marker=dict(size=8)))
        england_decile = master.groupby("imd_decile", as_index=False)["sum"].mean()
        fig.add_trace(go.Scatter(x=england_decile["imd_decile"], y=england_decile["sum"], mode="lines", name="England mean", line=dict(color="#98A2B3", width=2, dash="dash")))
        fig.update_layout(title="Projected value across deprivation deciles", xaxis_title="IMD decile · 1 most deprived, 10 least deprived", yaxis_title="Mean projected value")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(style_figure(fig, 430, True), use_container_width=True, config=PLOT_CONFIG)
        composition = primary_frame[PATHWAYS].mean().rename(index=LABELS).sort_values()
        comp_fig = go.Figure(go.Bar(x=composition.values, y=composition.index, orientation="h", marker_color=[NEGATIVE if value < 0 else PRIMARY for value in composition.values], text=[f"{value:.2f}" for value in composition.values], textposition="outside", hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
        comp_fig.update_layout(title="Mean pathway composition", xaxis_title="Mean projected pathway value", yaxis_title="")
        st.plotly_chart(style_figure(comp_fig, 420, False), use_container_width=True, config=PLOT_CONFIG)
    deprivation = primary_frame.groupby("imd_decile").size().reindex(range(1, 11), fill_value=0)
    deprivation_pct = 100 * deprivation / deprivation.sum()
    dep_fig = go.Figure(go.Bar(x=deprivation_pct.index, y=deprivation_pct.values, marker_color=PRIMARY, text=[f"{value:.1f}%" for value in deprivation_pct.values], textposition="outside"))
    dep_fig.update_layout(title=f"Deprivation composition of {primary_region}", xaxis_title="IMD decile", yaxis_title="Share of regional LSOAs (%)")
    dep_fig.update_xaxes(dtick=1)
    st.plotly_chart(style_figure(dep_fig, 360, False), use_container_width=True, config=PLOT_CONFIG)
    st.markdown(f"<div class='insight'><b>Regional observation.</b> {clean(primary_region)} contains {stats['n']:,} matched LSOAs. Its mean projected value is {stats['mean']:.2f}, and the D10–D1 difference is {stats['gap']:+.2f}. {stats['high_deprivation_share']:.1f}% of its LSOAs are in IMD deciles 1–3.</div>", unsafe_allow_html=True)
    interpretation_expander(
        "The overview establishes regional magnitude, deprivation composition and spatial distribution before detailed comparison.",
        "Describe the selected region and identify broad gradients or spatial concentrations.",
        "England is shown only as a reference; the map displays the selected region.",
        "Regional means, medians, proportions, decile means and a robust map range are calculated from matched LSOAs.",
        "Use charts for magnitude and the map for location. A whole region is not assigned a single IMD decile.",
        "Regional summaries conceal within-region variation and do not control for urban form, housing or transport context.",
    )

elif page == pages[1]:
    st.markdown("## How does this region differ from another?")
    st.markdown("<div class='question'>Compare two regions using identical statistics, pathway order and map colour scales.</div>", unsafe_allow_html=True)
    a = region_stats(primary_frame)
    b = region_stats(comparison_frame)
    metric_cards([
        (f"{primary_region} mean", number(a["mean"]), f"Median {number(a['median'])}"),
        (f"{comparison_region} mean", number(b["mean"]), f"Median {number(b['median'])}"),
        ("Mean difference A−B", signed(a["mean"] - b["mean"]), "Same statistic on both sides"),
        (f"{primary_region} D10–D1", signed(a["gap"]), f"D1/D10 {a['parity']:.2f}"),
        (f"{comparison_region} D10–D1", signed(b["gap"]), f"D1/D10 {b['parity']:.2f}"),
        ("Gap difference A−B", signed(a["gap"] - b["gap"]), "Absolute decile-gap difference"),
    ])
    combined_range = robust_range(pd.concat([primary_frame["sum"], comparison_frame["sum"]], ignore_index=True))
    left, right = st.columns(2)
    with left:
        fig_a, _ = numeric_map(primary_frame, primary_region, "sum", primary_region, "Projected value", "YlGnBu", shared_range=combined_range)
        st.plotly_chart(fig_a, use_container_width=True, config=PLOT_CONFIG)
    with right:
        fig_b, _ = numeric_map(comparison_frame, comparison_region, "sum", comparison_region, "Projected value", "YlGnBu", shared_range=combined_range)
        st.plotly_chart(fig_b, use_container_width=True, config=PLOT_CONFIG)
    st.caption(f"Shared map scale: {combined_range[0]:.2f} to {combined_range[1]:.2f}, capped at the combined 2nd–98th percentiles. True values remain in tooltips.")
    dec_a = primary_frame.groupby("imd_decile", as_index=False)["sum"].mean()
    dec_b = comparison_frame.groupby("imd_decile", as_index=False)["sum"].mean()
    dec_fig = go.Figure()
    dec_fig.add_trace(go.Scatter(x=dec_a["imd_decile"], y=dec_a["sum"], mode="lines+markers", name=primary_region, line=dict(color=PRIMARY, width=3)))
    dec_fig.add_trace(go.Scatter(x=dec_b["imd_decile"], y=dec_b["sum"], mode="lines+markers", name=comparison_region, line=dict(color=SECONDARY, width=3)))
    dec_fig.update_layout(title="Deprivation profiles on a shared scale", xaxis_title="IMD decile · 1 most deprived, 10 least deprived", yaxis_title="Mean projected value")
    dec_fig.update_xaxes(dtick=1)
    st.plotly_chart(style_figure(dec_fig, 430, True), use_container_width=True, config=PLOT_CONFIG)
    path_a = primary_frame[PATHWAYS].mean()
    path_b = comparison_frame[PATHWAYS].mean()
    difference = (path_a - path_b).rename(index=LABELS).sort_values()
    diff_fig = go.Figure(go.Bar(x=difference.values, y=difference.index, orientation="h", marker_color=[NEGATIVE if value < 0 else POSITIVE for value in difference.values], text=[f"{value:+.2f}" for value in difference.values], textposition="outside", hovertemplate="%{y}<br>Region A − Region B: %{x:.3f}<extra></extra>"))
    diff_fig.add_vline(x=0, line_color="#667085", line_width=1.5)
    diff_fig.update_layout(title=f"Which pathways distinguish {primary_region} from {comparison_region}?", xaxis_title=f"Mean pathway difference · {primary_region} minus {comparison_region}", yaxis_title="")
    st.plotly_chart(style_figure(diff_fig, 470, False), use_container_width=True, config=PLOT_CONFIG)
    largest = difference.abs().idxmax()
    st.markdown(f"<div class='insight'><b>Comparison observation.</b> The mean projected-value difference is {a['mean'] - b['mean']:+.2f}. The largest absolute pathway difference is {clean(largest)}, with {primary_region} minus {comparison_region} equal to {difference[largest]:+.2f}.</div>", unsafe_allow_html=True)
    interpretation_expander(
        "Shared scales prevent visual comparison from being distorted by independent axes or colour ranges.",
        "Compare two regions and identify the pathways associated with their difference.",
        "Each region is summarised using LSOA means, medians, proportions and decile-specific means.",
        "The pathway chart displays mean pathway value in the primary region minus the comparison region.",
        "Positive bars indicate higher mean values in the primary region; negative bars indicate higher values in the comparison region.",
        "Regional differences are descriptive and may reflect differing population, urban form and model assumptions.",
    )

elif page == pages[2]:
    st.markdown("## Which neighbourhoods differ from comparable places?")
    st.markdown("<div class='question'>Move from regional averages to LSOAs benchmarked against the same region and IMD decile.</div>", unsafe_allow_html=True)
    display = area_pool.copy()
    selected = display.loc[display["small_area"] == selected_code].iloc[0]
    low_mask = display["peer_percentile"] <= threshold
    high_deprivation_low = low_mask & display["imd_decile"].isin([1, 2, 3])
    metric_cards([
        ("Visible neighbourhoods", count(len(display)), f"{display['lad_name'].nunique()} local authorities"),
        ("Comparatively low", count(low_mask.sum()), f"At or below the {threshold}th peer percentile"),
        ("High-deprivation candidates", count(high_deprivation_low.sum()), "IMD deciles 1–3 and comparatively low"),
        ("Selected peer percentile", f"{selected['peer_percentile']:.1f}", f"Among {int(selected['peer_group_n']):,} peers"),
        ("Selected peer gap", signed(selected["peer_gap"]), "Area total minus peer median"),
        ("Dominant deficit", dominant_deficit_text(selected), f"{selected['dominant_deficit_value']:+.2f} versus peer median"),
    ])
    left, right = st.columns([1.12, 1])
    with left:
        peer_range = robust_range(display["peer_gap"], diverging=True)
        map_fig, _ = numeric_map(display, primary_region, "peer_gap", "Difference from same-region, same-decile peer median", "Peer gap", "PuOr", midpoint=0, selected_code=selected_code, shared_range=peer_range)
        st.plotly_chart(map_fig, use_container_width=True, config=PLOT_CONFIG)
        st.caption(f"Diverging scale is centred at zero and capped at ±{peer_range[1]:.2f}. The black outline identifies the selected neighbourhood; true values remain in tooltips.")
    with right:
        scatter = px.scatter(
            display,
            x="imd_score",
            y="sum",
            color="peer_gap",
            color_continuous_scale="PuOr",
            color_continuous_midpoint=0,
            range_color=peer_range,
            hover_name="lsoa_name",
            hover_data={
                "small_area": False,
                "lad_name": True,
                "imd_decile": True,
                "sum": ":.2f",
                "peer_median": ":.2f",
                "peer_gap": ":.2f",
                "peer_percentile": ":.1f",
            },
            labels={"imd_score": "IMD score · higher means more deprived", "sum": "Projected aggregate value", "peer_gap": "Peer gap", "lad_name": "Local authority", "imd_decile": "IMD decile", "peer_median": "Peer median", "peer_percentile": "Peer percentile"},
            title="Deprivation, projected value and peer-relative position",
        )
        scatter.add_trace(go.Scatter(x=[selected["imd_score"]], y=[selected["sum"]], mode="markers", marker=dict(symbol="diamond", size=15, color=SELECTED, line=dict(color="white", width=1.5)), name="Selected neighbourhood", hovertemplate=f"{selected['lsoa_name']}<br>Selected neighbourhood<extra></extra>"))
        st.plotly_chart(style_figure(scatter, 500, True), use_container_width=True, config=PLOT_CONFIG)
    ranked = display[["lsoa_name", "small_area", "lad_name", "imd_decile", "sum", "peer_median", "peer_gap", "peer_percentile", "dominant_deficit_pathway"]].copy()
    ranked["dominant_deficit_pathway"] = ranked["dominant_deficit_pathway"].map(lambda value: LABELS.get(value, "None" if value == "none" else value))
    ranked = ranked.sort_values(["peer_gap", "peer_percentile"]).rename(columns={
        "lsoa_name": "Neighbourhood",
        "small_area": "LSOA code",
        "lad_name": "Local authority",
        "imd_decile": "IMD decile",
        "sum": "Projected total",
        "peer_median": "Peer median",
        "peer_gap": "Peer gap",
        "peer_percentile": "Peer percentile",
        "dominant_deficit_pathway": "Dominant deficit",
    })
    st.markdown("### Ranked neighbourhood evidence")
    st.dataframe(ranked.head(250), use_container_width=True, hide_index=True, column_config={
        "Projected total": st.column_config.NumberColumn(format="%.2f"),
        "Peer median": st.column_config.NumberColumn(format="%.2f"),
        "Peer gap": st.column_config.NumberColumn(format="%+.2f"),
        "Peer percentile": st.column_config.NumberColumn(format="%.1f"),
    })
    st.download_button("Download filtered neighbourhood table", ranked.to_csv(index=False).encode("utf-8"), file_name=f"{primary_region.replace(' ', '_').lower()}_neighbourhoods.csv", mime="text/csv")
    relation = "below" if selected["peer_gap"] < 0 else "above"
    st.markdown(f"<div class='insight'><b>Neighbourhood observation.</b> {clean(selected['lsoa_name'])} is at the {selected['peer_percentile']:.1f} peer percentile among {int(selected['peer_group_n']):,} LSOAs in {clean(selected['region'])} and IMD decile {int(selected['imd_decile'])}. Its projected total is {abs(selected['peer_gap']):.2f} units {relation} the peer median.</div>", unsafe_allow_html=True)
    interpretation_expander(
        "Peer benchmarking reveals LSOAs that are unusually high or low after broad regional and deprivation context is held constant.",
        "Locate a neighbourhood, compare it with transparent peers and identify extreme cases.",
        "Median of LSOAs in the same region and IMD decile.",
        "Peer gap equals selected total minus peer median; peer percentile is the average within-group rank.",
        "Negative gaps and low percentiles indicate comparatively low projected values, not unequal spending or service provision.",
        "Region and decile are a simple stratification, not statistical matching; rurality, density, housing and transport remain uncontrolled.",
    )

elif page == pages[3]:
    st.markdown("## Which pathways explain the selected area, and does the conclusion survive pathway removal?")
    st.markdown("<div class='question'>Separate raw composition from peer-relative explanation, then preserve the baseline while testing a modified aggregate.</div>", unsafe_allow_html=True)
    if st.button("Reset scenario to all 11 pathways"):
        st.session_state["scenario_pathways"] = PATHWAYS.copy()
        st.rerun()
    selected_pathways = st.multiselect("Included pathways", PATHWAYS, format_func=lambda value: LABELS[value], key="scenario_pathways")
    if not selected_pathways:
        st.warning("Select at least one pathway to construct a scenario.")
        st.stop()
    scenario = scenario_frame(master, tuple(selected_pathways))
    scenario["baseline_low"] = scenario["peer_percentile"] <= threshold
    scenario["scenario_low"] = scenario["scenario_percentile"] <= threshold
    conditions = [
        scenario["baseline_low"] & scenario["scenario_low"],
        ~scenario["baseline_low"] & scenario["scenario_low"],
        scenario["baseline_low"] & ~scenario["scenario_low"],
    ]
    scenario["classification_change"] = np.select(conditions, ["Remains low", "Enters low", "Leaves low"], default="Never low")
    scenario_selected = scenario.loc[scenario["small_area"] == selected_code].iloc[0]
    baseline_class = "Comparatively low" if scenario_selected["baseline_low"] else "Not comparatively low"
    modified_class = "Comparatively low" if scenario_selected["scenario_low"] else "Not comparatively low"
    metric_cards([
        ("Baseline projected total", number(selected_row["sum"]), "All 11 pathways"),
        ("Modified projected total", number(scenario_selected["scenario_total"]), f"{len(selected_pathways)} pathways included"),
        ("Baseline peer gap", signed(selected_row["peer_gap"]), f"Percentile {selected_row['peer_percentile']:.1f}"),
        ("Modified peer gap", signed(scenario_selected["scenario_peer_gap"]), f"Percentile {scenario_selected['scenario_percentile']:.1f}"),
        ("Baseline classification", baseline_class, f"Bottom {threshold}% threshold"),
        ("Modified classification", modified_class, f"Change: {scenario_selected['classification_change']}"),
    ])
    excluded = [LABELS[pathway] for pathway in PATHWAYS if pathway not in selected_pathways]
    st.markdown(f"<div class='notice'><b>Active analytical scenario.</b> {len(selected_pathways)} of 11 pathways are included. Excluded: {clean(', '.join(excluded) if excluded else 'None')}.</div>", unsafe_allow_html=True)
    raw_values = selected_row[PATHWAYS].rename(index=LABELS)
    peer_gaps = pd.Series({LABELS[pathway]: selected_row[f"{pathway}_peer_gap"] for pathway in PATHWAYS})
    ordered = peer_gaps.sort_values().index.tolist()
    left, right = st.columns(2)
    with left:
        raw = raw_values.reindex(ordered)
        raw_fig = go.Figure(go.Bar(x=raw.values, y=raw.index, orientation="h", marker_color=[NEGATIVE if value < 0 else PRIMARY for value in raw.values], text=[f"{value:+.2f}" for value in raw.values], textposition="outside", hovertemplate="%{y}<br>Raw value: %{x:.3f}<extra></extra>"))
        raw_fig.add_vline(x=0, line_color="#667085", line_width=1.5)
        raw_fig.update_layout(title="What forms the selected area’s total?", xaxis_title="Raw projected pathway value", yaxis_title="")
        st.plotly_chart(style_figure(raw_fig, 470, False), use_container_width=True, config=PLOT_CONFIG)
    with right:
        residual = peer_gaps.reindex(ordered)
        residual_fig = go.Figure(go.Bar(x=residual.values, y=residual.index, orientation="h", marker_color=[NEGATIVE if value < 0 else POSITIVE for value in residual.values], text=[f"{value:+.2f}" for value in residual.values], textposition="outside", hovertemplate="%{y}<br>Area − peer median: %{x:.3f}<extra></extra>"))
        residual_fig.add_vline(x=0, line_color="#667085", line_width=1.5)
        residual_fig.update_layout(title="Why does it differ from comparable areas?", xaxis_title="Pathway value minus peer-pathway median", yaxis_title="")
        st.plotly_chart(style_figure(residual_fig, 470, False), use_container_width=True, config=PLOT_CONFIG)
    base_dec = master[master["region"].isin([primary_region, comparison_region])].groupby(["region", "imd_decile"], as_index=False)["sum"].mean()
    mod_dec = scenario[scenario["region"].isin([primary_region, comparison_region])].groupby(["region", "imd_decile"], as_index=False)["scenario_total"].mean()
    gap_rows = []
    for region in [primary_region, comparison_region]:
        base_part = base_dec[base_dec["region"] == region].set_index("imd_decile")["sum"]
        mod_part = mod_dec[mod_dec["region"] == region].set_index("imd_decile")["scenario_total"]
        gap_rows.append({"Region": region, "State": "Baseline", "Gap": base_part.get(10, np.nan) - base_part.get(1, np.nan)})
        gap_rows.append({"Region": region, "State": "Modified", "Gap": mod_part.get(10, np.nan) - mod_part.get(1, np.nan)})
    gap_frame = pd.DataFrame(gap_rows)
    gap_fig = px.bar(gap_frame, x="Region", y="Gap", color="State", barmode="group", text_auto=".2f", color_discrete_map={"Baseline": PRIMARY, "Modified": SECONDARY}, title="How the regional D10–D1 gap changes")
    gap_fig.update_layout(yaxis_title="D10 mean minus D1 mean", xaxis_title="")
    st.plotly_chart(style_figure(gap_fig, 390, True), use_container_width=True, config=PLOT_CONFIG)
    primary_scenario = scenario[scenario["region"] == primary_region].merge(master[["small_area", "imd_score"]], on="small_area", how="left")
    change_map = category_map(primary_scenario, primary_region, selected_code)
    st.plotly_chart(change_map, use_container_width=True, config=PLOT_CONFIG)
    change_counts = scenario["classification_change"].value_counts()
    changed = int(change_counts.get("Enters low", 0) + change_counts.get("Leaves low", 0))
    dominant = dominant_deficit_text(selected_row)
    st.markdown(f"<div class='insight'><b>Scenario observation.</b> {clean(selected_row['lsoa_name'])} moves from the {selected_row['peer_percentile']:.1f} to the {scenario_selected['scenario_percentile']:.1f} peer percentile and is classified as <b>{clean(scenario_selected['classification_change'])}</b>. Its largest baseline negative pathway difference is {clean(dominant)}. Across England, {changed:,} LSOAs enter or leave the bottom-{threshold}% peer group under this scenario.</div>", unsafe_allow_html=True)
    interpretation_expander(
        "The page distinguishes raw pathway composition from the pathway differences that explain peer-relative position.",
        "Identify the largest negative pathway residual and test whether the selected classification depends on pathway inclusion.",
        "Baseline uses all eleven pathways; the modified state uses the selected subset and recomputes all peer statistics.",
        "Scenario totals are summed from retained pathways, then region-decile medians, gaps, percentiles and classifications are recalculated.",
        "A classification change indicates sensitivity of the aggregate interpretation, not a predicted policy effect.",
        "Pathways are components of a modelled aggregate and are not independent interventions with known investment-response functions.",
    )

elif page == pages[4]:
    st.markdown("## Does the projected difference persist to 2050?")
    st.markdown("<div class='question'>Finish the funnel with one selected neighbourhood, one explicit benchmark and a bounded projection window.</div>", unsafe_allow_html=True)
    benchmark_options = [
        "Same-region, same-decile peer median",
        "Primary-region mean",
        "England mean",
        "Same-region decile-10 mean",
    ]
    benchmark_choice = st.selectbox("Outlook benchmark", benchmark_options, key="outlook_benchmark")
    selected_annual = annual[annual["small_area"] == selected_code].sort_values("year").copy()
    if benchmark_choice == benchmark_options[0]:
        selected_annual["benchmark"] = selected_annual["annual_peer_median"]
        benchmark_short = "peer median"
    elif benchmark_choice == benchmark_options[1]:
        benchmark = annual[annual["region"] == primary_region].groupby("year")["annual_total"].mean()
        selected_annual["benchmark"] = selected_annual["year"].map(benchmark)
        benchmark_short = f"{primary_region} mean"
    elif benchmark_choice == benchmark_options[2]:
        benchmark = annual.groupby("year")["annual_total"].mean()
        selected_annual["benchmark"] = selected_annual["year"].map(benchmark)
        benchmark_short = "England mean"
    else:
        benchmark = annual[(annual["region"] == primary_region) & (annual["imd_decile"] == 10)].groupby("year")["annual_total"].mean()
        selected_annual["benchmark"] = selected_annual["year"].map(benchmark)
        benchmark_short = f"{primary_region} decile-10 mean"
    selected_annual["gap"] = selected_annual["annual_total"] - selected_annual["benchmark"]
    years_below = int((selected_annual["gap"] < 0).sum())
    first = first_crossover(selected_annual["year"], selected_annual["gap"])
    sustained = sustained_crossover(selected_annual["year"], selected_annual["gap"], 3)
    cumulative_gap = float((selected_annual["benchmark"] - selected_annual["annual_total"]).sum())
    slope = float(np.polyfit(selected_annual["year"], selected_annual["gap"], 1)[0])
    metric_cards([
        ("Years below benchmark", f"{years_below} of {len(selected_annual)}", f"Benchmark: {benchmark_short}"),
        ("First projected crossover", str(first) if first else "None by 2050", "First year area reaches or exceeds benchmark"),
        ("Sustained crossover", str(sustained) if sustained else "None by 2050", "First of three consecutive years at or above benchmark"),
        ("2025 gap", signed(selected_annual.iloc[0]["gap"], 3), "Area minus benchmark"),
        ("2050 gap", signed(selected_annual.iloc[-1]["gap"], 3), "Area minus benchmark"),
        ("Projected gap trend", signed(slope, 4), "Change in gap per year"),
        ("Cumulative benchmark gap", signed(cumulative_gap, 3), "Benchmark minus area, summed 2025–2050"),
    ])
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=selected_annual["year"], y=selected_annual["annual_total"], mode="lines+markers", name=selected_row["lsoa_name"], line=dict(color=SELECTED, width=3), marker=dict(size=6)))
    line_fig.add_trace(go.Scatter(x=selected_annual["year"], y=selected_annual["benchmark"], mode="lines", name=benchmark_short, line=dict(color=SECONDARY, width=3, dash="dash")))
    if first:
        line_fig.add_vline(x=first, line_color=PRIMARY, line_dash="dot", line_width=1.5)
        line_fig.add_annotation(x=first, y=max(selected_annual["annual_total"].max(), selected_annual["benchmark"].max()), text=f"First crossover: {first}", showarrow=False, xanchor="left", yshift=10)
    line_fig.add_annotation(x=selected_annual["year"].iloc[-1], y=selected_annual["annual_total"].iloc[-1], text="Selected area", showarrow=False, xanchor="left", xshift=8, font=dict(color=SELECTED))
    line_fig.add_annotation(x=selected_annual["year"].iloc[-1], y=selected_annual["benchmark"].iloc[-1], text=benchmark_short, showarrow=False, xanchor="left", xshift=8, font=dict(color=SECONDARY))
    line_fig.update_layout(title="Projected annual trajectory against the selected benchmark", xaxis_title="Year", yaxis_title="Annual projected value")
    st.plotly_chart(style_figure(line_fig, 470, True), use_container_width=True, config=PLOT_CONFIG)
    gap_fig = go.Figure()
    gap_fig.add_trace(go.Scatter(x=selected_annual["year"], y=selected_annual["gap"], mode="lines+markers", fill="tozeroy", line=dict(color=PRIMARY, width=2.5), marker=dict(size=5), name="Annual gap", hovertemplate="Year %{x}<br>Area − benchmark: %{y:.4f}<extra></extra>"))
    gap_fig.add_hline(y=0, line_color="#475467", line_width=1.5)
    gap_fig.update_layout(title="Annual projected gap to benchmark", xaxis_title="Year", yaxis_title="Selected area minus benchmark")
    st.plotly_chart(style_figure(gap_fig, 360, False), use_container_width=True, config=PLOT_CONFIG)
    crossover_text = f"first reaches the benchmark in {first}" if first else "does not reach the benchmark within 2025–2050"
    sustained_text = f"and sustains parity for three years from {sustained}" if sustained else "and has no three-year sustained crossover within the projection window"
    direction = "improving" if slope > 0 else "worsening" if slope < 0 else "stable"
    st.markdown(f"<div class='insight'><b>Final projected observation.</b> {clean(selected_row['lsoa_name'])} is below the {clean(benchmark_short)} in {years_below} of {len(selected_annual)} projected years, {clean(crossover_text)} {clean(sustained_text)}. The annual gap trend is {direction} at {slope:+.4f} source units per year. The projection window ends in 2050.</div>", unsafe_allow_html=True)
    summary = pd.DataFrame([
        {
            "Neighbourhood": selected_row["lsoa_name"],
            "LSOA code": selected_code,
            "Region": selected_row["region"],
            "Local authority": selected_row["lad_name"],
            "IMD decile": int(selected_row["imd_decile"]),
            "Peer percentile": round(float(selected_row["peer_percentile"]), 1),
            "Peer gap": round(float(selected_row["peer_gap"]), 3),
            "Dominant deficit": dominant_deficit_text(selected_row),
            "Outlook benchmark": benchmark_short,
            "Years below benchmark": years_below,
            "First projected crossover": first if first else "None by 2050",
            "Sustained crossover": sustained if sustained else "None by 2050",
            "Cumulative benchmark gap": round(cumulative_gap, 4),
        }
    ])
    st.download_button("Download selected-area evidence summary", summary.to_csv(index=False).encode("utf-8"), file_name=f"{selected_code}_evidence_summary.csv", mime="text/csv")
    st.caption("The outlook uses the stored aggregate annual series. It does not estimate investment requirements and does not extrapolate beyond 2050.")
    interpretation_expander(
        "The outlook converts an annual line chart into explicit persistence and crossover observations for the selected area.",
        "Assess whether the area remains below an explicit benchmark and whether any crossover is temporary or sustained.",
        "The user chooses a peer, regional, England or regional decile-10 benchmark.",
        "Annual gap equals selected annual value minus benchmark; crossover is the first non-negative year; sustained crossover requires three consecutive non-negative years.",
        "Results describe the supplied projection window and should be phrased as projected crossover or no crossover by 2050.",
        "The annual series is model-dependent, contains no uncertainty interval here, and cannot be converted into a required investment amount.",
    )
