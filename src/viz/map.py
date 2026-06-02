from __future__ import annotations

import branca.colormap as cm
import folium
import pandas as pd

from src.viz.formatting import format_number_pt_br

METRICS = {
    "accident_count": {"label": "Acidentes", "decimals": 0},
    "fatal_accident_count": {"label": "Acidentes fatais", "decimals": 0},
    "fatal_victims": {"label": "Vítimas fatais", "decimals": 1},
    "people_involved": {"label": "Pessoas envolvidas", "decimals": 1},
}


def build_colormap(values: pd.Series, label: str) -> cm.LinearColormap:
    clean_values = pd.to_numeric(values, errors="coerce").fillna(0)
    max_value = max(float(clean_values.max()), 1.0)
    colormap = cm.linear.YlOrRd_09.scale(0, max_value)
    colormap.caption = label
    return colormap


def base_map(center: list[float], bounds: list[list[float]]) -> folium.Map:
    m = folium.Map(
        location=center,
        tiles=None,
        zoom_start=8,
        control_scale=True,
        max_bounds=True,
    )
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=False,
    ).add_to(m)
    m.fit_bounds(bounds)
    return m


def add_choropleth(
    m: folium.Map,
    geojson: dict,
    metric: str,
    metric_label: str,
) -> folium.Map:
    values = pd.Series(
        [feature["properties"].get(metric, 0) for feature in geojson["features"]]
    )
    colormap = build_colormap(values, metric_label)

    def style_function(feature: dict) -> dict:
        value = float(feature["properties"].get(metric) or 0)
        return {
            "fillColor": colormap(value),
            "color": "#374151",
            "weight": 0.8,
            "fillOpacity": 0.86,
        }

    def highlight_function(_: dict) -> dict:
        return {"weight": 2.4, "color": "#111827", "fillOpacity": 0.95}

    for feature in geojson["features"]:
        value = feature["properties"].get(metric, 0)
        feature["properties"][f"{metric}_ptbr"] = format_number_pt_br(
            value,
            decimals=METRICS.get(metric, {}).get("decimals", 0),
        )

    folium.GeoJson(
        geojson,
        name=metric_label,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "nome_municipio",
                f"{metric}_ptbr",
                "fonte",
            ],
            aliases=[
                "Município",
                metric_label,
                "Fonte",
            ],
            localize=True,
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                "nome_municipio",
                f"{metric}_ptbr",
                "fonte",
                "nm_rgint",
            ],
            aliases=[
                "Município",
                metric_label,
                "Fonte",
                "Região geográfica intermediária",
            ],
            localize=True,
        ),
    ).add_to(m)
    colormap.add_to(m)
    return m
