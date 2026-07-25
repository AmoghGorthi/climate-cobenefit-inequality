import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import plotly
st.caption(f"plotly {plotly.__version__}")
st.set_page_config(page_title="Climate Co-Benefit Inequality", layout="centered")

BLUES = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']
INK, ACCENT = '#08306b', '#d6604d'
BAR = [BLUES[min(int(i*8/9), 8)] for i in range(10)]
LABELS = {'air_quality':'Air quality','congestion':'Congestion','dampness':'Dampness',
          'diet_change':'Diet change','excess_cold':'Excess cold','excess_heat':'Excess heat',
          'hassle_costs':'Hassle costs','noise':'Noise','physical_activity':'Physical activity',
          'road_repairs':'Road repairs','road_safety':'Road safety'}
CB = list(LABELS)

st.markdown("""<style>
.block-container{max-width:900px;padding-top:2.5rem;}
h1{font-size:2.1rem;letter-spacing:-.02em;}
h3{color:#08306b;margin-top:2.5rem;}
.lede{font-size:1.15rem;line-height:1.6;color:#333;}
.note{background:#f4f7fa;border-left:3px solid #08306b;padding:11px 15px;
      font-size:.88rem;line-height:1.55;color:#333;margin-top:.4rem;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def pq(n): return pd.read_parquet(f"agg_{n}.parquet")

@st.cache_data
def mapdata(): return pd.read_parquet("map_data.parquet")

@st.cache_data
def stats():
    with open("stats.json") as f: return json.load(f)

@st.cache_resource
def geo(path):
    with open(path) as f: return json.load(f)

S = stats()

def sty(f, h=360, legend=False):
    f.update_layout(height=h, plot_bgcolor='white', paper_bgcolor='white',
                    margin=dict(l=8,r=8,t=34,b=8), showlegend=legend,
                    font=dict(size=12,color='#333'), title_font=dict(size=14,color=INK),
                    hoverlabel=dict(bgcolor='white',font_size=12))
    f.update_yaxes(gridcolor='#eef1f4', zeroline=True, zerolinecolor='#ccd4dc')
    f.update_xaxes(gridcolor='#eef1f4')
    return f

def note(t): st.markdown(f"<div class='note'>{t}</div>", unsafe_allow_html=True)

# 1 ── THE CLAIM
st.title("Climate action creates benefits.")
st.markdown(f"<p class='lede'>They are not shared equally. Across <b>{S['n']:,}</b> "
            f"neighbourhoods in England, the least deprived receive <b>{S['ratio']}× more</b> "
            f"benefit from climate action than the most deprived.</p>", unsafe_allow_html=True)

st.markdown("### The gradient")
d, dcb = pq('decile'), pq('decile_cobenefit')
f = go.Figure(go.Bar(x=d['imd_decile'], y=d['mean'], marker_color=BAR,
                     text=d['mean'].round(2), textposition='outside',
                     hovertemplate='Decile %{x}<br>%{y:.2f}<extra></extra>'))
f.update_layout(title='Mean co-benefit by deprivation decile',
                xaxis_title='IMD decile  (1 = most deprived)', yaxis_title='Mean co-benefit')
f.update_xaxes(dtick=1)
st.plotly_chart(sty(f), use_container_width=True)

pick = st.selectbox("Break down a decile", range(1,11),
                    format_func=lambda x: f"Decile {x}" + (" — most deprived" if x==1 else
                                                           " — least deprived" if x==10 else ""))
row = dcb[dcb['imd_decile']==pick].iloc[0]
vals = pd.Series({LABELS[c]: row[c] for c in CB}).sort_values()
f = go.Figure(go.Bar(x=vals.values, y=vals.index, orientation='h',
                     marker_color=[ACCENT if v<0 else INK for v in vals.values],
                     hovertemplate='%{y}<br>%{x:.3f}<extra></extra>'))
f.update_layout(title=f'What decile {pick} is made of', xaxis_title='Mean co-benefit')
st.plotly_chart(sty(f, 330), use_container_width=True)
note(f"Decile 1 exceeds decile 2 by 0.08 despite a lower physical activity value — noise "
     f"reduction in dense urban areas offsets the deficit. From decile 2 upward the gradient "
     f"rises without exception. Spearman ρ = {S['spearman']}, n = {S['n']:,}.")

# 2 ── WHERE
st.markdown("---")
st.markdown("### Where")
la = pq('la')
opts = ['All England (by local authority)'] + sorted(la['region'].unique())
sel = st.selectbox("Area", opts)

if sel.startswith('All England'):
    gj, dm = geo('boundaries_la.geojson'), la.copy()
    loc, key = 'lad_name', 'properties.lad_name'
    hov = {'sum':':.2f','region':True,'n':True}
else:
    gj = geo(f"geo/{sel.replace(' ','_')}.geojson")
    dm = mapdata()
    dm = dm[dm['region']==sel].copy()
    loc, key = 'small_area', 'properties.small_area'
    hov = {'sum':':.2f','imd_decile':True,'lad_name':True}

dm['v'] = dm['sum'].clip(0, S['vmax'])
f = px.choropleth_map(dm, geojson=gj, locations=loc, featureidkey=key, color='v',
                      color_continuous_scale=BLUES, range_color=(0, S['vmax']),
                      map_style='carto-positron', opacity=.82,
                      hover_name=loc, hover_data={**hov, 'v': False})
f.update_traces(marker_line_width=.2, marker_line_color='white')
f.update_layout(height=520, margin=dict(l=0,r=0,t=0,b=0),
                coloraxis_colorbar=dict(title='', thickness=12, len=.6))
st.plotly_chart(f, use_container_width=True)
note(f"Colour capped at the 95th percentile ({S['vmax']}). LSOAs hold roughly equal populations "
     f"but the largest quartile covers 92.6% of England's land — the map overstates per-person "
     f"benefit by {S['area_bias']}%. Read magnitude from the charts, geography from the map.")

# 3 ── WHY
st.markdown("---")
st.markdown("### Why")
st.caption("Remove a co-benefit and watch the gradient. Try taking out physical activity.")
keep = st.multiselect("Included co-benefits", CB, default=CB, format_func=lambda c: LABELS[c])
if keep:
    y = dcb[keep].sum(axis=1)
    rev = y.iloc[9] < y.iloc[0]
    f = go.Figure(go.Bar(x=dcb['imd_decile'], y=y, marker_color=BAR,
                         text=y.round(2), textposition='outside',
                         hovertemplate='Decile %{x}<br>%{y:.2f}<extra></extra>'))
    f.add_hline(y=0, line_color='#bbb', line_width=1)
    f.update_layout(title=f"{len(keep)} of 11 co-benefits included",
                    xaxis_title='IMD decile', yaxis_title='Mean co-benefit')
    f.update_xaxes(dtick=1)
    st.plotly_chart(sty(f), use_container_width=True)
    st.markdown(f"<p style='color:{ACCENT if rev else INK};font-weight:600;'>"
                f"D1 {y.iloc[0]:.2f} → D10 {y.iloc[9]:.2f} · "
                f"{'gradient REVERSED' if rev else 'gradient rises'}</p>", unsafe_allow_html=True)
else:
    st.info("Select at least one co-benefit.")
note(f"Removing physical activity does not flatten the gradient — it reverses it. Every other "
     f"co-benefit combined favours more deprived areas. Physical activity spans {S['pa_gap']} "
     f"units across deciles; the next largest spans {abs(S['next_gap'])}. Excess cold correlates "
     f"more strongly with deprivation (r = −0.584) yet spans only 0.22 — correlation measures "
     f"direction, not magnitude.")

# 4 ── EVERYWHERE
st.markdown("---")
st.markdown("### Everywhere")
r = pq('regional').sort_values('gap')
f = go.Figure()
for _, x in r.iterrows():
    f.add_shape(type='line', x0=0, x1=x['gap'], y0=x['region'], y1=x['region'],
                line=dict(color='#cfdcea', width=2), layer='below')
f.add_trace(go.Scatter(x=r['gap'], y=r['region'], mode='markers',
                       marker=dict(size=11, color=INK),
                       hovertemplate='%{y}<br>gap %{x:.2f}<extra></extra>'))
f.update_layout(title='Gap between least and most deprived, by region',
                xaxis_title='D10 − D1 (mean co-benefit)')
st.plotly_chart(sty(f, 380), use_container_width=True)

rsel = st.selectbox("Compare a region with the national average", sorted(r['region']))
rr = pq('reg_decile').query("region == @rsel")
f = go.Figure()
f.add_trace(go.Bar(x=rr['imd_decile'], y=rr['sum'], marker_color=BAR, name=rsel,
                   hovertemplate='Decile %{x}<br>%{y:.2f}<extra></extra>'))
f.add_trace(go.Scatter(x=d['imd_decile'], y=d['mean'], mode='lines+markers', name='England',
                       line=dict(color=ACCENT, width=2, dash='dot'), marker=dict(size=5)))
f.update_layout(title=f'{rsel} vs England', xaxis_title='IMD decile', yaxis_title='Mean co-benefit')
f.update_xaxes(dtick=1)
st.plotly_chart(sty(f, 340, legend=True), use_container_width=True)
note("All nine regions show a significant negative gradient (p < 0.001). Ratios are sensitive to "
     "small denominators — London's 15.2× reflects a very low decile-1 value (0.66), not an "
     "exceptional decile-10 value. By absolute gap London (9.33) exceeds the North West (5.13) "
     "by 1.8×, not 5×.")

# 5 ── OVER TIME
st.markdown("---")
st.markdown("### And it widens")
t = pq('temporal')
f = go.Figure()
f.add_trace(go.Scatter(x=t['year'], y=t['d10'], name='Least deprived',
                       line=dict(color=INK, width=2.4)))
f.add_trace(go.Scatter(x=t['year'], y=t['d1'], name='Most deprived', fill='tonexty',
                       fillcolor='rgba(198,219,239,.5)', line=dict(color=ACCENT, width=2.4)))
f.add_hline(y=0, line_color='#bbb', line_width=1)
f.update_layout(title='Mean annual co-benefit, 2025–2050', xaxis_title='Year',
                yaxis_title='Mean annual co-benefit')
st.plotly_chart(sty(f, 380, legend=True), use_container_width=True)

td = pq('temporal_decile')
yr = st.slider("Year", int(td['year'].min()), int(td['year'].max()), 2025)
yv = td.query("year == @yr").sort_values('imd_decile')
f = go.Figure(go.Bar(x=yv['imd_decile'], y=yv['val'], marker_color=BAR,
                     hovertemplate='Decile %{x}<br>%{y:.3f}<extra></extra>'))
f.add_hline(y=0, line_color='#bbb', line_width=1)
f.update_layout(title=f'Annual co-benefit in {yr}', xaxis_title='IMD decile',
                yaxis_title='Mean annual co-benefit',
                yaxis_range=[td['val'].min()*1.15, td['val'].max()*1.15])
f.update_xaxes(dtick=1)
st.plotly_chart(sty(f, 320), use_container_width=True)
note(f"The gap widens from {S['gap_2025']} in 2025 to {S['gap_2050']} in 2050 — a "
     f"{S['gap_2050']/S['gap_2025']:.1f}× increase, peaking in the final year. The most deprived "
     f"decile is net negative in 2025–26, bearing costs before receiving benefits. Values are "
     f"annual, not cumulative.")

# methods
st.markdown("---")
with st.expander("Data, methods and limitations"):
    st.markdown(f"""
**Sources** — UK Co-Benefits dataset (Level 1), CO-BENS project, Edinburgh Climate Change
Institute · Index of Multiple Deprivation 2019, MHCLG · LSOA 2011 boundaries, UK Data Service.

**Pipeline** — 46,426 rows → 3 null rows removed (all Scottish) → inner join on LSOA code →
**{S['n']:,} matched English LSOAs**. Scotland, Wales and Northern Ireland use separate
deprivation indices not directly comparable to IMD.

**Selection bias** — 1,034 IMD LSOAs had no co-benefit match, from boundary revisions between
vintages. Welch t-test on deprivation score: t = −0.95, p = 0.34 — no significant difference
between matched and unmatched areas.

**Statistics** — Spearman ρ = {S['spearman']} reported as primary; the relationship is monotonic
but non-linear, with variance rising across deciles. Pearson r = {S['pearson']}, R² = {S['r2']}.
Drivers identified by sensitivity analysis rather than correlation.

**Limitations** — England only · modelled projections, not observed outcomes, with no uncertainty
bounds · correlational, no causal inference · LSOA averages mask within-area variation ·
choropleth overstates per-person benefit by {S['area_bias']}% · spatial autocorrelation not
modelled · IMD 2019 predates the projection window · no user evaluation (outside ethical approval
scope).

*Amogh Gorthi · MSc Data Science and AI · Newcastle University · Supervisor: Dr Xinhuan Shu*
""")
