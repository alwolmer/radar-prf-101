from __future__ import annotations

import json

import pandas as pd

from src.viz.data import (
    FORECAST_HORIZON_DAYS,
    forecast_is_current,
    normalize_forecast,
    period_source_label,
)


def test_normalize_forecast_aggregates_daily_rows_for_weekly_display() -> None:
    raw = pd.DataFrame(
        [
            {
                "codigo_municipio": "4205407",
                "nome_municipio": "Florianópolis",
                "accident_date": "2026-06-08",
                "predicted_accident_count": 1.25,
            },
            {
                "codigo_municipio": "4205407",
                "nome_municipio": "Florianópolis",
                "accident_date": "2026-06-09",
                "predicted_accident_count": 2.75,
            },
            {
                "codigo_municipio": "4202008",
                "nome_municipio": "Balneário Camboriú",
                "accident_date": "2026-06-14",
                "predicted_accident_count": 1.0,
            },
        ]
    )

    weekly = normalize_forecast(raw, "semana")

    assert set(weekly["codigo_municipio"]) == {"4205407", "4202008"}
    florianopolis = weekly[weekly["codigo_municipio"] == "4205407"].iloc[0]
    assert florianopolis["week_start"] == pd.Timestamp("2026-06-08").date()
    assert florianopolis["week_end"] == pd.Timestamp("2026-06-14").date()
    assert florianopolis["accident_count"] == 4.0
    assert florianopolis["fonte"] == "previsão"


def test_period_source_label_marks_historical_forecast_and_mixed() -> None:
    assert period_source_label(pd.DataFrame({"fonte": ["histórico"]})) == "Histórico"
    assert period_source_label(pd.DataFrame({"fonte": ["previsão"]})) == "Previsão"
    assert (
        period_source_label(pd.DataFrame({"fonte": ["histórico", "previsão"]}))
        == "Misto"
    )


def test_forecast_current_requires_matching_cutoff_and_30_day_horizon(
    tmp_path, monkeypatch
) -> None:
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    monkeypatch.setenv("VIZ_FORECAST_PATH", str(forecast_dir))

    manifest_path = forecast_dir / "forecast_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_panel_max_date": "2026-06-08",
                "horizon_days": FORECAST_HORIZON_DAYS,
            }
        ),
        encoding="utf-8",
    )

    assert forecast_is_current(pd.Timestamp("2026-06-08").date())

    manifest_path.write_text(
        json.dumps(
            {
                "source_panel_max_date": "2026-06-07",
                "horizon_days": FORECAST_HORIZON_DAYS,
            }
        ),
        encoding="utf-8",
    )
    assert not forecast_is_current(pd.Timestamp("2026-06-08").date())

    manifest_path.write_text(
        json.dumps({"source_panel_max_date": "2026-06-08", "horizon_days": 14}),
        encoding="utf-8",
    )
    assert not forecast_is_current(pd.Timestamp("2026-06-08").date())


def test_forecast_current_accepts_cutoff_inside_weekly_period(
    tmp_path, monkeypatch
) -> None:
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    monkeypatch.setenv("VIZ_FORECAST_PATH", str(forecast_dir))
    (forecast_dir / "forecast_manifest.json").write_text(
        json.dumps(
            {
                "source_panel_max_date": "2026-02-28",
                "horizon_days": FORECAST_HORIZON_DAYS,
            }
        ),
        encoding="utf-8",
    )

    assert forecast_is_current(
        pd.Timestamp("2026-03-01").date(),
        pd.Timestamp("2026-02-23").date(),
    )
