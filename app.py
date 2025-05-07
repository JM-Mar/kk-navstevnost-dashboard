#!/usr/bin/env python3
"""
app.py

Dash aplikácia:
 • PREHĽAD – pre každé obdobie: LFL trend + ktoré prevádzky rastú/klesajú + trend Obrat & DOH
 • ANALÝZA – STL, YoY, rolling sklony (6m aj 3m) + popisy (detailný + jednoduchý)
 • PROGNÓZA – dynamický text podľa zvoleného obdobia
"""

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import STL

# —— KONFIGURÁCIA ——
visits_path = "navstevnost2.csv"
metrics_path = "doh_obrat_2024.csv"

# 1) načítanie mesačných návštev
df_all = pd.read_csv(
    visits_path,
    sep=';', decimal=',',
    index_col=0, parse_dates=True, dayfirst=True,
    engine='python', encoding='cp1250'
)

# 2) načítanie DOH a Obratu + čistenie Obrat → float
df_met = pd.read_csv(
    metrics_path,
    sep=';', decimal=',',
    thousands=' ', engine='python', encoding='cp1250'
)
df_met['Obrat'] = (
    df_met['Obrat'].astype(str)
        .str.replace('€', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
)
# Parsovanie dátumu
df_met['DateMonthYear'] = df_met['DateMonthYear'].astype(str)
df_met['Date'] = pd.to_datetime(
    df_met['DateMonthYear'],
    format='%m.%y',
    errors='coerce'
)
df_met = (
    df_met.rename(columns={'DOH': 'DOH_h'})
          .set_index(['Date', 'StoreAbbr'])[['Obrat', 'DOH_h']]
          .sort_index()
)

# 3) spojenie návštev s metrikami
df_vis = df_all.stack().rename('visits').to_frame()
df_vis.index.names = ['Date', 'StoreAbbr']
df_join = df_vis.join(df_met, how='left').reset_index()

# 4) globálne sklony (len na prehľad)
slopes = {}
for st in df_all.columns:
    series = df_all[st].dropna()
    if len(series) >= 12:
        trend = STL(series, period=12, robust=True).fit().trend.dropna()
        slopes[st] = np.polyfit(np.arange(len(trend)), trend.values, 1)[0]

# 5) Like-for-Like subset a obdobia
start_ltl = "2022-07-01"
initial_stores = df_all.columns[df_all.loc[start_ltl].notnull()]
df_ltl = df_all[initial_stores].loc[start_ltl:]
last_date = df_all.index.max()
periods = {
    '2. polrok 2022 – súčasnosť': ('2022-07-01', last_date),
    'Rok 2023 – súčasnosť': ('2023-01-01', last_date),
    'Rok 2024 – súčasnosť': ('2024-01-01', last_date),
}

# 6) prehľad LFL trendov + Obrat + DOH_h
overview = {}
for name, (start, end) in periods.items():
    # visits trend
    agg_vis = df_ltl.loc[start:end].sum(axis=1)
    tr_vis = STL(agg_vis, period=12, robust=True).fit().trend.dropna()
    sl_vis = np.polyfit(np.arange(len(tr_vis)), tr_vis.values, 1)[0]
    # revenue trend
    df_period = df_join[(df_join['Date'] >= start) & (df_join['Date'] <= end)]
    agg_rev = df_period.groupby('Date')['Obrat'].sum()
    tr_rev = STL(agg_rev, period=12, robust=True).fit().trend.dropna()
    sl_rev = np.polyfit(np.arange(len(tr_rev)), tr_rev.values, 1)[0]
    # DOH trend
    agg_doh = df_period.groupby('Date')['DOH_h'].sum()
    tr_doh = STL(agg_doh, period=12, robust=True).fit().trend.dropna()
    sl_doh = np.polyfit(np.arange(len(tr_doh)), tr_doh.values, 1)[0]
    # store-level grows/drops
    grows, drops = [], []
    for store in df_all.columns:
        series2 = df_all[store].loc[start:end].dropna()
        if len(series2) >= 12:
            tr2 = STL(series2, period=12, robust=True).fit().trend.dropna()
            sl2 = np.polyfit(np.arange(len(tr2)), tr2.values, 1)[0]
            (grows if sl2 > 0 else drops).append(store)
    overview[name] = {
        'slope_vis': sl_vis,
        'slope_rev': sl_rev,
        'slope_doh': sl_doh,
        'grows': grows,
        'drops': drops,
    }

# 7) zostavenie Dash aplikácie
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Tabs([
        # PREHĽAD
        dcc.Tab(label='Prehľad', children=[
            html.H2('Prehľad trendov (Like-for-Like)'),
            html.H4('Čo grafy v Analýze ukazujú?'),
            html.Ul([
                html.Li('Séria & Trend'),
                html.Li('YoY % zmena'),
                html.Li('6m rolling sklon'),
                html.Li('3m rolling sklon'),
            ], style={'marginBottom': '20px'}),
            *[
                html.Div([
                    html.H3(name),
                    html.P(f"Trend návštev: {'rastie' if overview[name]['slope_vis']>0 else 'klesá'} o {abs(overview[name]['slope_vis']):.0f} návštev/mesiac"),
                    html.P(f"Trend obratu: {'rastie' if overview[name]['slope_rev']>0 else 'klesá'} o {abs(overview[name]['slope_rev']):.2f} €/mesiac"),
                    html.P(f"Trend DOH: {'rastie' if overview[name]['slope_doh']>0 else 'klesá'} o {abs(overview[name]['slope_doh']):.2f} h/mesiac"),
                    html.P("Rastúce: " + (", ".join(overview[name]['grows']) or 'žiadne')),
                    html.P("Klesajúce: " + (", ".join(overview[name]['drops']) or 'žiadne')),
                    html.Hr(),
                ], style={'marginBottom': '25px'}) for name in periods
            ]
        ]),
        # ANALÝZA prevádzok
        dcc.Tab(label='Analýza prevádzok', children=[
            html.Div([
                html.Label('Obdobie:'),
                dcc.Dropdown(
                    id='period-dropdown',
                    options=[{'label': k, 'value': k} for k in periods],
                    value=list(periods.keys())[0],
                    style={'width': '50%'}
                )
            ], style={'marginTop': 20}),
            html.Div([
                html.Label('Prevádzka:'),
                dcc.Dropdown(
                    id='store-dropdown',
                    options=[{'label': s, 'value': s} for s in df_all.columns],
                    value=df_all.columns[0],
                    style={'width': '50%'}
                )
            ], style={'marginTop': 10}),
            # 1) STL
            dcc.Graph(id='stl-graph'),
            html.Div(id='stl-desc'),
            html.Div(id='stl-simple'),
            # 2) YoY
            dcc.Graph(id='yoy-graph'),
            html.Div(id='yoy-desc'),
            html.Div(id='yoy-simple'),
            # 3) 6m rolling sklon
            dcc.Graph(id='rolling-slope-graph'),
            html.Div(id='rs6-desc'),
            html.Div(id='rs6-simple'),
            # 4) 3m rolling sklon
            dcc.Graph(id='rolling-slope-3m-graph'),
            html.Div(id='rs3-desc'),
            html.Div(id='rs3-simple'),
            # 5) Dopad na obrat
            dcc.Graph(id='rev-impact-graph'),
            html.Div(id='rev-desc'),
            html.Div(id='rev-simple'),
            # 6) Dopad na DOH
            dcc.Graph(id='doh-impact-graph'),
            html.Div(id='doh-desc'),
            html.Div(id='doh-simple'),
            html.Div(id='metrics-output', style={'marginTop': 20, 'fontSize': 17}),
        ])
    ])
])

# 8) Callback pre grafy + popisy + prognózu
@app.callback(
    Output('stl-graph', 'figure'), Output('stl-desc', 'children'), Output('stl-simple', 'children'),
    Output('yoy-graph', 'figure'), Output('yoy-desc', 'children'), Output('yoy-simple', 'children'),
    Output('rolling-slope-graph', 'figure'), Output('rs6-desc', 'children'), Output('rs6-simple', 'children'),
    Output('rolling-slope-3m-graph', 'figure'), Output('rs3-desc', 'children'), Output('rs3-simple', 'children'),
    Output('rev-impact-graph', 'figure'), Output('rev-desc', 'children'), Output('rev-simple', 'children'),
    Output('doh-impact-graph', 'figure'), Output('doh-desc', 'children'), Output('doh-simple', 'children'),
    Output('metrics-output', 'children'),
    Input('period-dropdown', 'value'), Input('store-dropdown', 'value')
)
def update_analysis(period_name, store):
    start, end = periods[period_name]
    series = df_all[store].loc[start:end].dropna()
    stl_res = STL(series, period=12, robust=True).fit()
    trend = stl_res.trend.dropna()
    slope = np.polyfit(np.arange(len(trend)), trend.values, 1)[0]

    # 1) STL
    fig1 = go.Figure([
        go.Scatter(x=series.index, y=series, name='Originál'),
        go.Scatter(x=trend.index, y=trend, name='Trend')
    ]).update_layout(title=f"{store} – Séria & Trend")
    desc1 = html.P("STL dekompozícia oddeľuje sezónnosť od trendu, aby ste videli dlhodobý pohyb návštev.")
    simp1 = html.P("Graf ukazuje, či návštevy dlhodobo rastú alebo klesajú.")

    # 2) YoY % zmena
    yoy = series.pct_change(12) * 100
    fig2 = go.Figure([go.Bar(x=yoy.index, y=yoy)]).update_layout(title="YoY % zmena")
    desc2 = html.P("Porovnáva aktuálny mesiac s rovnakým mesiacom pred rokom v percentách.")
    simp2 = html.P("Ukazuje, o koľko percent sa zmenili návštevy oproti minulému roku.")

    # 3) 6m rolling sklon
    rs6 = trend.rolling(6).apply(lambda x: np.polyfit(np.arange(6), x, 1)[0])
    fig3 = go.Figure([go.Scatter(x=rs6.index, y=rs6)]).update_layout(title="6m rolling sklon")
    desc3 = html.P("6-mesačný pohyblivý regresný sklon trendu návštev.")
    simp3 = html.P("Ukazuje, či sa tempo zmien za polrok zrýchľuje alebo spomaľuje.")

    # 4) 3m rolling sklon
    rs3 = trend.rolling(3).apply(lambda x: np.polyfit(np.arange(3), x, 1)[0])
    fig4 = go.Figure([go.Scatter(x=rs3.index, y=rs3)]).update_layout(title="3m rolling sklon")
    desc4 = html.P("3-mesačný pohyblivý regresný sklon pre krátkodobú analýzu trendu.")
    simp4 = html.P("Ukazuje, či sa návštevy za posledné 3 mesiace zrýchľujú alebo spomaľujú.")

    # 5) Dopad na obrat
    dfp = df_join.query("Date >= @start and Date <= @end and StoreAbbr == @store").copy()
    dfp['rev_per_visit'] = dfp['Obrat'] / dfp['visits']
    dfp['delta_rev'] = slope * dfp['rev_per_visit']
    fig5 = go.Figure([go.Bar(x=dfp['Date'], y=dfp['delta_rev'])]).update_layout(title="Dopad trendu na mesačný obrat (€)")
    desc5 = html.P("Vypočíta, o koľko by sa zmenil mesačný obrat pri zachovaní trendu návštev.")
    simp5 = html.P("Ukazuje, koľko eur mesačne získate alebo stratíte z trendu návštev.")

    # 6) Dopad na DOH
    dfp['recommended_DOH'] = dfp.apply(
        lambda r: r['DOH_h'] * (r['visits'] + slope) / r['visits'] if slope < 0 else r['DOH_h'],
        axis=1
    )
    fig6 = go.Figure([
        go.Scatter(x=dfp['Date'], y=dfp['DOH_h'], name='Aktuálne DOH'),
        go.Scatter(x=dfp['Date'], y=dfp['recommended_DOH'], name='Odporúčané DOH')
    ]).update_layout(title="DOH: aktuálne vs odporúčané")
    desc6 = html.P("Ukazuje súčasné a odporúčané pracovné hodiny na základe predpokladaného trendu návštev.")
    simp6 = html.P("Ukazuje, koľko hodín personálu potrebujete podľa trendu návštev.")

    # summary
    dfp_valid = dfp.dropna(subset=['rev_per_visit', 'DOH_h'])
    if dfp_valid.empty:
        summary = html.P("Nedostatok dát pre prognózu v danom období.")
    else:
        last = dfp_valid.iloc[-1]
        rev_change = slope * last['rev_per_visit']
        avg_doh = last['DOH_h']
        rec_doh = last['recommended_DOH']
        summary = html.Div([
            html.H4("Prognóza pri trvalom trende"),
            html.P(f"Obdobie: {period_name}, Prevádzka: {store}"),
            html.P(f"Trend: {'rastúci' if slope>0 else 'klesajúci'} ({slope:.1f} návštev/mes.)"),
            html.P(f"Obrat: {'↑' if rev_change>0 else '↓'} {abs(rev_change):.2f} € / mesiac"),
            html.P(f"DOH: aktuálne {avg_doh:.1f} h → odporúčané {rec_doh:.1f} h")
        ])

    return (
        fig1, desc1, simp1,
        fig2, desc2, simp2,
        fig3, desc3, simp3,
        fig4, desc4, simp4,
        fig5, desc5, simp5,
        fig6, desc6, simp6,
        summary
    )

if __name__ == '__main__':
    app.run(debug=True)
