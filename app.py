#!/usr/bin/env python3
"""
app.py

Dash aplikácia:
 • PREHĽAD – pre každé obdobie: LFL trend + ktoré prevádzky rastú/klesajú
 • ANALÝZA – STL, YoY, rolling sklon a metriky pre každú KK & obdobie
"""

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import STL

# —— KONFIGURÁCIA ——
csv_path = "navstevnost2.csv"

# 1) Načítanie dát
df_all = pd.read_csv(
    csv_path, sep=';', decimal=',',
    index_col=0, parse_dates=True,
    dayfirst=True, engine='python'
)

# 2) Like-for-Like subset (iba prevádzky bežiace k 1.7.2022)
start_ltl = "2022-07-01"
initial_stores = df_all.columns[df_all.loc[start_ltl].notnull()]
df_ltl = df_all[initial_stores].loc[start_ltl:]
last_date = df_all.index.max()

# 3) Definícia období
periods = {
    '2. polrok 2022 – súčasnosť': ('2022-07-01', last_date),
    'Rok 2023 – súčasnosť':       ('2023-01-01', last_date),
    'Rok 2024 – súčasnosť':       ('2024-01-01', last_date),
}

# 4) Predpočítanie pre Prehľad
#   pre každé obdobie LFL slope + grows/drops z ALL stores
overview = {}
for name, (a, b) in periods.items():
    # LFL trend slope
    agg = df_ltl.loc[a:b].sum(axis=1)
    trend = STL(agg, period=12, robust=True).fit().trend.dropna()
    slope = np.polyfit(np.arange(len(trend)), trend.values, 1)[0]
    # grows/drops z all stores
    grows, drops = [], []
    for store in df_all.columns:
        s = df_all[store].loc[a:b].dropna()
        if len(s) >= 12:
            tr = STL(s, period=12, robust=True).fit().trend.dropna()
            if len(tr)>=2 and np.polyfit(np.arange(len(tr)), tr.values,1)[0] > 0:
                grows.append(store)
            else:
                drops.append(store)
    overview[name] = {
        'slope': slope,
        'grows': grows,
        'drops': drops
    }

# 5) Stavba Dash aplikácie
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    dcc.Tabs([

      # — PREHĽAD —
      dcc.Tab(label='Prehľad', children=[
        html.H2('Prehľad trendov (Like-for-Like)'),

        # Vysvetlenie čo grafy ukazujú
        html.H4('Čo grafy v Analýze ukazujú?'),
        html.Ul([
          html.Li("Séria & Trend: modrá = mesačné návštevy, oranžová = trend bez sezóny."),
          html.Li("Medziročná zmena %: porovnanie s rovnakým mesiacom minulého roka."),
          html.Li("6-mesačný kĺzavý sklon: rýchlosť rastu/poklesu trendu za posledných 6 mesiacov."),
        ], style={'marginBottom':'20px'}),

        # Poznámka o očistenom trende
        html.P(
          "Poznámka: Trend je sezónne očistená krivka. STL dekompozícia rozdelí mesačné "
          "návštevy na trend (hladká čiara), sezónu (opakované letné/vianočné vrcholy) "
          "a náhodné odchýlky; potom na trend aplikujeme lineárny sklon."
        ),

        # Pre každé obdobie vypíšeme LFL trend + grows/drops
        *[
          html.Div([
            html.H3(name),
            html.P(
              f"Vo všeobecnosti návštevnosť KK "
              f"{'rastie' if overview[name]['slope']>0 else 'klesá'} "
              f"o {abs(overview[name]['slope']):.0f} návštev/mesiac."
            ),
            html.P("Prevádzky s rastúcou návštevnosťou: " + (', '.join(overview[name]['grows']) or "žiadne")),
            html.P("Prevádzky s klesajúcou alebo stabilnou návštevnosťou: " + (', '.join(overview[name]['drops']) or "žiadne")),
            html.Hr()
          ], style={'marginBottom':'25px'})
          for name in periods
        ]
      ]),

      # — ANALÝZA —
      dcc.Tab(label='Analýza prevádzok', children=[
        html.Div([
          html.Label('Vyber obdobie:'),
          dcc.Dropdown(
            id='period-dropdown',
            options=[{'label':k,'value':k} for k in periods],
            value=list(periods.keys())[0],
            style={'width':'50%'}
          )
        ], style={'marginTop':20}),
        html.Div([
          html.Label('Vyber prevádzku:'),
          dcc.Dropdown(
            id='store-dropdown',
            options=[{'label':s,'value':s} for s in df_all.columns],
            value=df_all.columns[0],
            style={'width':'50%'}
          )
        ], style={'marginTop':10}),
        dcc.Graph(id='stl-graph'),
        dcc.Graph(id='yoy-graph'),
        dcc.Graph(id='rolling-slope-graph'),
        html.Div(id='metrics-output', style={'marginTop':20, 'fontSize':18})
      ]),

    ])
])

# 6) Callback pre detailnú ANALÝZU
@app.callback(
    Output('stl-graph','figure'),
    Output('yoy-graph','figure'),
    Output('rolling-slope-graph','figure'),
    Output('metrics-output','children'),
    Input('period-dropdown','value'),
    Input('store-dropdown','value')
)
def update_analysis(period_name, store):
    a, b = periods[period_name]
    s = df_all[store].loc[a:b].dropna()

    # STL + trend
    stl = STL(s, period=12, robust=True).fit()
    trend = stl.trend.dropna()
    fig1 = go.Figure([
      go.Scatter(x=s.index, y=s, name='Originál'),
      go.Scatter(x=trend.index, y=trend, name='Trend')
    ]).update_layout(title=f"{store} – Série & Trend")

    # YoY %
    yoy = s.pct_change(12)*100
    fig2 = go.Figure([go.Bar(x=yoy.index, y=yoy)]).update_layout(title="YoY % zmena")

    # 6-mesačný rolling sklon na trende
    rs6 = trend.rolling(6).apply(lambda x: np.polyfit(np.arange(6), x,1)[0])
    fig3 = go.Figure([go.Scatter(x=rs6.index, y=rs6)]).update_layout(
      title="6-mesačný kĺzavý sklon (na trende)")

    # metriky sklony: počiatočný 6m, od 2023, od 2024
    texts = []
    t0 = trend.loc[a:(pd.to_datetime(a)+pd.DateOffset(months=5)).strftime('%Y-%m-%d')].dropna()
    if len(t0)>=2:
        m0 = np.polyfit(np.arange(len(t0)), t0.values,1)[0]
        texts.append(f"Počiatočný sklon (6m): {m0:.0f} návštev/mes.")
    else:
        texts.append("Počiatočný sklon: nedostatok dát")
    t23 = trend.loc['2023-01-01':b].dropna()
    if len(t23)>=2:
        m23 = np.polyfit(np.arange(len(t23)), t23.values,1)[0]
        texts.append(f"Skloň od 1.1.2023: {m23:.0f} návštev/mes.")
    else:
        texts.append("Skloň od 2023: nedostatok dát")
    t24 = trend.loc['2024-01-01':b].dropna()
    if len(t24)>=2:
        m24 = np.polyfit(np.arange(len(t24)), t24.values,1)[0]
        texts.append(f"Skloň od 1.1.2024: {m24:.0f} návštev/mes.")
    else:
        texts.append("Skloň od 2024: nedostatok dát")

    metr = html.Ul([html.Li(txt) for txt in texts])
    return fig1, fig2, fig3, metr

# 7) Spustenie aplikácie
if __name__ == '__main__':
    app.run(debug=True)
