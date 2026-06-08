from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.viz.formatting import format_date_pt_br

HIGHLIGHT_COLOR = "#e4572e"
PRIMARY_COLOR = "#2f80ed"
MUTED_COLOR = "#cbd5e1"
NEGATIVE_COLOR = "#c23b3b"
NEUTRAL_COLOR = "#64748b"
OTHER_COLOR = "#94a3b8"
PLOT_TEMPLATE = "plotly_white"


def _empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": "#475569"},
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin={"l": 28, "r": 24, "t": 48, "b": 36},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _metric_frame(
    data: pd.DataFrame,
    *,
    entity_column: str,
    metric: str,
) -> pd.DataFrame:
    frame = data[[entity_column, metric]].copy()
    frame[entity_column] = frame[entity_column].fillna("Sem região").astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce").fillna(0.0)
    return frame


def aggregate_by_period(
    data: pd.DataFrame,
    *,
    date_column: str,
    entity_column: str,
    metric: str,
) -> pd.DataFrame:
    frame = data[[date_column, entity_column, metric]].copy()
    frame[entity_column] = frame[entity_column].fillna("Sem região").astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce").fillna(0.0)
    grouped = frame.groupby([date_column, entity_column], as_index=False)[metric].sum()
    return grouped.sort_values([date_column, entity_column])


def ranking_bar(
    rows: pd.DataFrame,
    *,
    entity_column: str,
    entity_label: str,
    metric: str,
    metric_label: str,
    selected_entity: str,
) -> go.Figure:
    if rows.empty:
        return _empty_figure("Nenhum dado para o período selecionado.")

    grouped = (
        _metric_frame(rows, entity_column=entity_column, metric=metric)
        .groupby(entity_column, as_index=False)[metric]
        .sum()
        .sort_values(metric, ascending=True)
    )
    grouped["destaque"] = grouped[entity_column].where(
        grouped[entity_column] == selected_entity,
        "Demais",
    )
    grouped["destaque"] = grouped["destaque"].where(
        grouped["destaque"] == "Demais",
        "Selecionado",
    )

    fig = px.bar(
        grouped,
        x=metric,
        y=entity_column,
        orientation="h",
        color="destaque",
        color_discrete_map={
            "Selecionado": HIGHLIGHT_COLOR,
            "Demais": PRIMARY_COLOR,
        },
        labels={entity_column: entity_label, metric: metric_label},
        title=f"Ranking de {metric_label.lower()} por {entity_label.lower()}",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        height=max(360, min(820, 120 + 24 * len(grouped))),
        margin={"l": 12, "r": 18, "t": 56, "b": 36},
        showlegend=False,
        yaxis_title=None,
        xaxis_title=metric_label,
    )
    return fig


def delta_bar(
    panel: pd.DataFrame,
    *,
    date_column: str,
    current_period: Any,
    previous_period: Any | None,
    entity_column: str,
    entity_label: str,
    metric: str,
    metric_label: str,
    selected_entity: str,
    max_entities: int = 18,
) -> go.Figure:
    if previous_period is None:
        return _empty_figure("Não há período anterior para comparar.")

    scoped = panel[panel[date_column].isin([previous_period, current_period])].copy()
    grouped = aggregate_by_period(
        scoped,
        date_column=date_column,
        entity_column=entity_column,
        metric=metric,
    )
    current = grouped[grouped[date_column] == current_period][
        [entity_column, metric]
    ].rename(columns={metric: "atual"})
    previous = grouped[grouped[date_column] == previous_period][
        [entity_column, metric]
    ].rename(columns={metric: "anterior"})
    delta = current.merge(previous, on=entity_column, how="outer").fillna(0.0)
    if delta.empty:
        return _empty_figure("Nenhum dado nos períodos comparados.")

    delta["variacao"] = delta["atual"] - delta["anterior"]
    delta["status"] = "Estável"
    delta.loc[delta["variacao"] > 0, "status"] = "Aumento"
    delta.loc[delta["variacao"] < 0, "status"] = "Redução"
    delta.loc[delta[entity_column] == selected_entity, "status"] = "Selecionado"
    delta = delta.sort_values(
        "variacao", key=lambda values: values.abs(), ascending=False
    )

    if len(delta) > max_entities:
        selected = delta[delta[entity_column] == selected_entity]
        rest = delta[delta[entity_column] != selected_entity].head(max_entities)
        delta = pd.concat([selected, rest], ignore_index=True).drop_duplicates(
            entity_column,
            keep="first",
        )
    delta = delta.sort_values("variacao", ascending=True)

    fig = px.bar(
        delta,
        x="variacao",
        y=entity_column,
        orientation="h",
        color="status",
        color_discrete_map={
            "Selecionado": HIGHLIGHT_COLOR,
            "Aumento": PRIMARY_COLOR,
            "Redução": NEGATIVE_COLOR,
            "Estável": NEUTRAL_COLOR,
        },
        hover_data={"atual": ":.2f", "anterior": ":.2f", "variacao": ":.2f"},
        labels={
            entity_column: entity_label,
            "variacao": f"Variação de {metric_label.lower()}",
            "status": "",
        },
        title="Variação em relação ao período anterior",
        template=PLOT_TEMPLATE,
    )
    fig.add_vline(x=0, line_width=1, line_color="#334155")
    fig.update_layout(
        height=max(360, min(760, 120 + 26 * len(delta))),
        margin={"l": 12, "r": 18, "t": 56, "b": 36},
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
        yaxis_title=None,
        xaxis_title=f"Atual - anterior ({metric_label})",
    )
    return fig


def highlighted_timeseries(
    panel: pd.DataFrame,
    *,
    date_column: str,
    entity_column: str,
    entity_label: str,
    metric: str,
    metric_label: str,
    selected_entity: str,
) -> go.Figure:
    columns = [date_column, entity_column, metric]
    if "fonte" in panel.columns:
        columns.append("fonte")
    frame = panel[columns].copy()
    frame[entity_column] = frame[entity_column].fillna("Sem região").astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce").fillna(0.0)
    if "fonte" not in frame.columns:
        frame["fonte"] = "histórico"
    grouped = (
        frame.groupby([date_column, entity_column, "fonte"], as_index=False)[metric]
        .sum()
        .sort_values([date_column, entity_column, "fonte"])
    )
    if grouped.empty:
        return _empty_figure("Nenhuma série disponível.")

    fig = go.Figure()
    for (entity, source), entity_rows in grouped.groupby([entity_column, "fonte"]):
        is_selected = entity == selected_entity
        is_forecast = source == "previsão"
        fig.add_trace(
            go.Scatter(
                x=entity_rows[date_column],
                y=entity_rows[metric],
                mode="lines",
                name=(
                    f"{entity} ({source})"
                    if is_selected and is_forecast
                    else (str(entity) if is_selected else "Demais")
                ),
                line={
                    "color": HIGHLIGHT_COLOR if is_selected else MUTED_COLOR,
                    "width": 3.2 if is_selected else 1,
                    "dash": "dash" if is_forecast else "solid",
                },
                opacity=1.0 if is_selected else 0.35,
                hovertemplate=(
                    f"{entity_label}: {entity}<br>"
                    "Período: %{x}<br>"
                    f"Fonte: {source}<br>"
                    f"{metric_label}: %{{y:.2f}}<extra></extra>"
                ),
                showlegend=is_selected,
            )
        )

    forecast_dates = grouped.loc[grouped["fonte"] == "previsão", date_column]
    if not forecast_dates.empty:
        fig.add_vrect(
            x0=min(forecast_dates),
            x1=max(forecast_dates),
            fillcolor="#fff7ed",
            opacity=0.35,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=f"Evolução de {metric_label.lower()} por {entity_label.lower()}",
        template=PLOT_TEMPLATE,
        height=430,
        margin={"l": 24, "r": 18, "t": 56, "b": 36},
        xaxis_title="Período",
        yaxis_title=metric_label,
        hovermode="x unified",
    )
    return fig


def selected_vs_average(
    panel: pd.DataFrame,
    *,
    date_column: str,
    entity_column: str,
    entity_label: str,
    metric: str,
    metric_label: str,
    selected_entity: str,
    rolling_window: int = 4,
) -> go.Figure:
    grouped = aggregate_by_period(
        panel,
        date_column=date_column,
        entity_column=entity_column,
        metric=metric,
    )
    if grouped.empty:
        return _empty_figure("Nenhuma série disponível.")

    wide = grouped.pivot_table(
        index=date_column,
        columns=entity_column,
        values=metric,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    if selected_entity not in wide:
        return _empty_figure(f"{entity_label} selecionado sem dados na janela.")

    selected = wide[selected_entity]
    average = wide.mean(axis=1)
    selected_rolling = selected.rolling(rolling_window, min_periods=1).mean()
    average_rolling = average.rolling(rolling_window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wide.index,
            y=selected,
            mode="lines",
            name=f"{selected_entity} por período",
            line={"color": HIGHLIGHT_COLOR, "width": 1},
            opacity=0.28,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wide.index,
            y=selected_rolling,
            mode="lines",
            name=f"{selected_entity} - média móvel",
            line={"color": HIGHLIGHT_COLOR, "width": 3.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wide.index,
            y=average_rolling,
            mode="lines",
            name="Média da BR-101 SC",
            line={"color": PRIMARY_COLOR, "width": 2.8, "dash": "dot"},
        )
    )
    fig.update_layout(
        title=f"{entity_label} selecionado vs média do corredor",
        template=PLOT_TEMPLATE,
        height=390,
        margin={"l": 24, "r": 18, "t": 56, "b": 36},
        xaxis_title="Período",
        yaxis_title=metric_label,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
        hovermode="x unified",
    )
    return fig


def share_area(
    panel: pd.DataFrame,
    *,
    date_column: str,
    entity_column: str,
    entity_label: str,
    metric: str,
    metric_label: str,
    selected_entity: str,
    max_entities: int = 8,
) -> go.Figure:
    grouped = aggregate_by_period(
        panel,
        date_column=date_column,
        entity_column=entity_column,
        metric=metric,
    )
    if grouped.empty:
        return _empty_figure("Nenhuma participação disponível.")

    totals = grouped.groupby(entity_column)[metric].sum().sort_values(ascending=False)
    keep = set(totals.head(max_entities).index)
    keep.add(selected_entity)
    grouped["grupo"] = grouped[entity_column].where(
        grouped[entity_column].isin(keep),
        "Demais",
    )
    collapsed = grouped.groupby([date_column, "grupo"], as_index=False)[metric].sum()
    period_total = collapsed.groupby(date_column)[metric].transform("sum")
    collapsed["participacao"] = collapsed[metric] / period_total.where(
        period_total != 0,
        1,
    )
    collapsed["ordem"] = collapsed["grupo"].map(
        lambda value: (
            0 if value == selected_entity else (99 if value == "Demais" else 1)
        )
    )
    collapsed = collapsed.sort_values(["ordem", "grupo", date_column])

    color_map = {selected_entity: HIGHLIGHT_COLOR, "Demais": OTHER_COLOR}
    fig = px.area(
        collapsed,
        x=date_column,
        y="participacao",
        color="grupo",
        color_discrete_map=color_map,
        labels={
            date_column: "Período",
            "participacao": "Participação",
            "grupo": entity_label,
            metric: metric_label,
        },
        title=f"Participação no total de {metric_label.lower()}",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        height=410,
        margin={"l": 24, "r": 18, "t": 56, "b": 36},
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def period_heatmap(
    panel: pd.DataFrame,
    *,
    date_column: str,
    entity_column: str,
    entity_label: str,
    metric: str,
    metric_label: str,
) -> go.Figure:
    grouped = aggregate_by_period(
        panel,
        date_column=date_column,
        entity_column=entity_column,
        metric=metric,
    )
    if grouped.empty:
        return _empty_figure("Nenhum dado para o mapa de calor.")

    totals = grouped.groupby(entity_column)[metric].sum().sort_values(ascending=False)
    pivot = (
        grouped.pivot_table(
            index=entity_column,
            columns=date_column,
            values=metric,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(totals.index)
        .sort_index(axis=1)
    )
    x_labels = [format_date_pt_br(value) for value in pivot.columns]
    fig = px.imshow(
        pivot,
        x=x_labels,
        y=pivot.index,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "Período", "y": entity_label, "color": metric_label},
        title=f"Mapa de calor de {metric_label.lower()} por período",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(
        height=max(420, min(900, 140 + 24 * len(pivot))),
        margin={"l": 12, "r": 18, "t": 56, "b": 80},
        xaxis_tickangle=-45,
    )
    return fig
