from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.viz.data import (
    PANEL_TABLES,
    as_geojson,
    build_municipio_features,
    load_bounds,
    load_panel,
    panel_root,
    period_options,
    rows_for_period,
)
from src.viz.formatting import (
    format_date_pt_br,
    format_number_pt_br,
    format_period_pt_br,
)
from src.viz.map import METRICS, add_choropleth, base_map

st.set_page_config(
    page_title="Radar PRF BR-101 SC",
    page_icon="BR-101",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.4rem; padding-bottom: 1.6rem; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    .stRadio > div { gap: .75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_options(panel: pd.DataFrame) -> list[str]:
    return [metric for metric in METRICS if metric in panel.columns]


def period_label(granularity: str, period, rows: pd.DataFrame) -> str:
    if (
        granularity == "semana"
        and "week_end" in rows
        and rows["week_end"].notna().any()
    ):
        return format_period_pt_br(period, rows["week_end"].dropna().iloc[0])
    return format_date_pt_br(period)


def main() -> None:
    root = panel_root()

    st.title("Radar PRF BR-101 em Santa Catarina")
    st.caption("Acidentes por município no painel gold `br101_sc_municipio_panel`.")

    with st.sidebar:
        st.header("Filtros")
        granularity = st.radio(
            "Granularidade",
            options=["dia", "semana"],
            horizontal=True,
            format_func=lambda value: "Dia" if value == "dia" else "Semana",
        )

        panel = load_panel(str(root), granularity)
        periods = period_options(panel, granularity)
        if not periods:
            st.error("Nenhum período disponível no painel gold.")
            st.stop()

        selected_period = st.select_slider(
            "Período",
            options=periods,
            value=periods[-1],
            format_func=lambda value: (
                f"Semana de {format_date_pt_br(value)}"
                if granularity == "semana"
                else format_date_pt_br(value)
            ),
        )

        available_metrics = metric_options(panel)
        metric = st.selectbox(
            "Indicador",
            options=available_metrics,
            format_func=lambda value: METRICS[value]["label"],
        )

        st.divider()
        st.caption(f"Fonte de dados: `{root}`")

    period_rows = rows_for_period(panel, granularity, selected_period)
    metric_label = METRICS[metric]["label"]
    date_column = PANEL_TABLES[granularity]["date_column"]
    period_text = period_label(granularity, selected_period, period_rows)
    features = build_municipio_features(str(root))
    geojson = as_geojson(features, period_rows, metric)
    center, bounds = load_bounds(str(root))

    total_metric = period_rows[metric].fillna(0).sum()
    active_municipios = int((period_rows["accident_count"].fillna(0) > 0).sum())
    source_label = ", ".join(sorted(period_rows["fonte"].dropna().unique()))
    max_row = period_rows.sort_values(metric, ascending=False).head(1)
    destaque = "-" if max_row.empty else str(max_row.iloc[0]["nome_municipio"])

    top_left, top_mid, top_right, top_fourth = st.columns(4)
    top_left.metric(
        metric_label, format_number_pt_br(total_metric, METRICS[metric]["decimals"])
    )
    top_mid.metric("Municípios com acidente", format_number_pt_br(active_municipios))
    top_right.metric("Período", period_text)
    top_fourth.metric("Maior valor", destaque)

    m = base_map(center, bounds)
    add_choropleth(m, geojson, metric=metric, metric_label=metric_label)
    st_folium(m, height=650, use_container_width=True, returned_objects=[])

    lower_left, lower_right = st.columns([2, 1])
    with lower_left:
        st.subheader("Municípios no período")
        table = period_rows[
            [
                "nome_municipio",
                date_column,
                "accident_count",
                "fatal_accident_count",
                "fatal_victims",
                "people_involved",
                "fonte",
            ]
        ].sort_values(["accident_count", "fatal_victims"], ascending=False)
        table = table.rename(
            columns={
                "nome_municipio": "Município",
                date_column: "Data",
                "accident_count": "Acidentes",
                "fatal_accident_count": "Acidentes fatais",
                "fatal_victims": "Vítimas fatais",
                "people_involved": "Pessoas envolvidas",
                "fonte": "Fonte",
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

    with lower_right:
        st.subheader("Série temporal")
        series = (
            panel.groupby(date_column, as_index=False)["accident_count"]
            .sum()
            .rename(columns={date_column: "Data", "accident_count": "Acidentes"})
        )
        st.line_chart(series, x="Data", y="Acidentes", height=300)
        st.caption(
            "Quando previsões forem materializadas, o app anexa períodos futuros "
            "via `VIZ_FORECAST_PATH` ou pelos caminhos gold padrão."
        )
        st.caption(f"Fonte no período selecionado: {source_label or '-'}")


if __name__ == "__main__":
    main()
