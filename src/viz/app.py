from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.viz.data import (
    PANEL_TABLES,
    as_geojson,
    build_municipio_features,
    load_bounds,
    load_forecast_manifest,
    load_panel,
    panel_root,
    period_options,
    period_source_label,
    period_source_options,
    rows_for_period,
)
from src.viz.formatting import (
    format_date_pt_br,
    format_number_pt_br,
    format_period_pt_br,
)
from src.viz.map import METRICS, add_choropleth, base_map
from src.viz.plots import (
    delta_bar,
    highlighted_timeseries,
    period_heatmap,
    ranking_bar,
    selected_vs_average,
    share_area,
)

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


COMPARISON_LEVELS = {
    "regiao": {
        "label": "Região",
        "column": "nm_rgint",
        "option": "Regiões",
    },
    "municipio": {
        "label": "Município",
        "column": "nome_municipio",
        "option": "Municípios",
    },
}

WINDOW_OPTIONS = {
    "Últimos 90 períodos": 90,
    "Últimos 180 períodos": 180,
    "Últimos 365 períodos": 365,
    "Todo o período": None,
}


def period_window(
    panel: pd.DataFrame,
    date_column: str,
    periods: list,
    selected_period,
    window_size: int | None,
) -> pd.DataFrame:
    if selected_period not in periods:
        return panel.copy()

    end_index = periods.index(selected_period)
    if window_size is None:
        selected_periods = periods[: end_index + 1]
    else:
        start_index = max(0, end_index - window_size + 1)
        selected_periods = periods[start_index : end_index + 1]
    return panel[panel[date_column].isin(selected_periods)].copy()


def previous_period(periods: list, selected_period):
    if selected_period not in periods:
        return None
    selected_index = periods.index(selected_period)
    if selected_index == 0:
        return None
    return periods[selected_index - 1]


def filter_periods_by_source(
    periods: list, source_by_period: dict, source_filter: str
) -> list:
    if source_filter == "Todos":
        return periods
    return [
        period
        for period in periods
        if source_by_period.get(period, "-") == source_filter
    ]


def entity_options(panel: pd.DataFrame, entity_column: str) -> list[str]:
    return sorted(panel[entity_column].fillna("Sem região").astype(str).unique())


def default_entity(
    period_rows: pd.DataFrame,
    entity_column: str,
    metric: str,
    options: list[str],
) -> str:
    if not options:
        return ""
    ranked = (
        period_rows.assign(
            **{
                entity_column: period_rows[entity_column]
                .fillna("Sem região")
                .astype(str),
                metric: pd.to_numeric(period_rows[metric], errors="coerce").fillna(0),
            }
        )
        .groupby(entity_column)[metric]
        .sum()
        .sort_values(ascending=False)
    )
    if ranked.empty:
        return options[0]
    return str(ranked.index[0])


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

        source_by_period = period_source_options(panel, granularity, periods)
        source_filter = st.radio(
            "Tipo de período",
            options=["Todos", "Histórico", "Previsão", "Misto"],
            horizontal=True,
        )
        selectable_periods = filter_periods_by_source(
            periods,
            source_by_period,
            source_filter,
        )
        if not selectable_periods:
            st.warning("Nenhum período disponível para o tipo selecionado.")
            st.stop()

        def format_period_option(value) -> str:
            date_label = (
                f"Semana de {format_date_pt_br(value)}"
                if granularity == "semana"
                else format_date_pt_br(value)
            )
            return f"{date_label} | {source_by_period.get(value, '-')}"

        selected_period = st.selectbox(
            "Período",
            options=selectable_periods,
            index=len(selectable_periods) - 1,
            format_func=format_period_option,
        )

        available_metrics = metric_options(panel)
        metric = st.selectbox(
            "Indicador",
            options=available_metrics,
            format_func=lambda value: METRICS[value]["label"],
        )

        sidebar_period_rows = rows_for_period(panel, granularity, selected_period)
        comparison_level = st.radio(
            "Comparar por",
            options=list(COMPARISON_LEVELS.keys()),
            horizontal=True,
            format_func=lambda value: COMPARISON_LEVELS[value]["option"],
        )
        entity_column = COMPARISON_LEVELS[comparison_level]["column"]
        entity_label = COMPARISON_LEVELS[comparison_level]["label"]
        entities = entity_options(panel, entity_column)
        selected_default = default_entity(
            sidebar_period_rows,
            entity_column,
            metric,
            entities,
        )
        selected_entity = st.selectbox(
            f"Destaque em {entity_label.lower()}",
            options=entities,
            index=entities.index(selected_default)
            if selected_default in entities
            else 0,
        )
        window_label = st.selectbox(
            "Janela dos gráficos",
            options=list(WINDOW_OPTIONS.keys()),
            index=1 if granularity == "dia" else 0,
        )

        st.divider()
        manifest = load_forecast_manifest()
        if manifest:
            st.caption(
                "Previsão: "
                f"{manifest.get('model_variant', '-')}; "
                f"gerada em {manifest.get('generated_at', '-')}; "
                f"base até {manifest.get('source_panel_max_date', '-')}; "
                f"horizonte {manifest.get('horizon_days', '-')} dias."
            )
        st.caption(f"Fonte de dados: `{root}`")

    period_rows = rows_for_period(panel, granularity, selected_period)
    selected_source = period_source_label(period_rows)
    metric_label = METRICS[metric]["label"]
    date_column = PANEL_TABLES[granularity]["date_column"]
    period_text = period_label(granularity, selected_period, period_rows)
    chart_panel = period_window(
        panel,
        date_column,
        periods,
        selected_period,
        WINDOW_OPTIONS[window_label],
    )
    previous = previous_period(periods, selected_period)
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
    top_fourth.metric("Fonte", selected_source)
    if selected_source == "Previsão":
        st.info("Período previsto: valores estimados pelo modelo champion no MLflow.")
    elif selected_source == "Misto":
        st.warning("Período misto: combina dados históricos e previsões.")
    st.caption(f"Maior valor no período: {destaque}")

    m = base_map(center, bounds)
    add_choropleth(m, geojson, metric=metric, metric_label=metric_label)
    st_folium(m, height=650, use_container_width=True, returned_objects=[])

    dados, comparacoes, evolucao, calor = st.tabs(
        ["Dados", "Comparações", "Evolução", "Mapa de calor"]
    )

    with dados:
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
        styled_table = table.style.apply(
            lambda row: [
                "background-color: #fff7ed" if row.get("Fonte") == "previsão" else ""
                for _ in row
            ],
            axis=1,
        )
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        st.caption(
            "Quando previsões forem materializadas, o app anexa períodos futuros "
            "via `VIZ_FORECAST_PATH` ou pelos caminhos gold padrão."
        )
        st.caption(f"Fonte no período selecionado: {source_label or '-'}")

    with comparacoes:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                ranking_bar(
                    period_rows,
                    entity_column=entity_column,
                    entity_label=entity_label,
                    metric=metric,
                    metric_label=metric_label,
                    selected_entity=selected_entity,
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                delta_bar(
                    panel,
                    date_column=date_column,
                    current_period=selected_period,
                    previous_period=previous,
                    entity_column=entity_column,
                    entity_label=entity_label,
                    metric=metric,
                    metric_label=metric_label,
                    selected_entity=selected_entity,
                ),
                use_container_width=True,
            )

    with evolucao:
        st.plotly_chart(
            highlighted_timeseries(
                chart_panel,
                date_column=date_column,
                entity_column=entity_column,
                entity_label=entity_label,
                metric=metric,
                metric_label=metric_label,
                selected_entity=selected_entity,
            ),
            use_container_width=True,
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                selected_vs_average(
                    chart_panel,
                    date_column=date_column,
                    entity_column=entity_column,
                    entity_label=entity_label,
                    metric=metric,
                    metric_label=metric_label,
                    selected_entity=selected_entity,
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                share_area(
                    chart_panel,
                    date_column=date_column,
                    entity_column=entity_column,
                    entity_label=entity_label,
                    metric=metric,
                    metric_label=metric_label,
                    selected_entity=selected_entity,
                ),
                use_container_width=True,
            )

    with calor:
        st.plotly_chart(
            period_heatmap(
                chart_panel,
                date_column=date_column,
                entity_column=entity_column,
                entity_label=entity_label,
                metric=metric,
                metric_label=metric_label,
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
