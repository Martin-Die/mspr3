"""Constantes partagées par la pipeline ETL."""

RENAME_MAP = {
    "Consommation": "consommation",
    "Prévision J-1": "prevision_j1",
    "Nucléaire": "nucleaire",
    "Eolien": "eolien",
    "Solaire": "solaire",
    "Hydraulique": "hydraulique",
    "Gaz": "gaz",
    "Taux de Co2": "co2",
    "Consommation (MW)": "consommation",
    "Prévision J-1 (MW)": "prevision_j1",
    "Nucléaire (MW)": "nucleaire",
    "Eolien (MW)": "eolien",
    "Solaire (MW)": "solaire",
    "Hydraulique (MW)": "hydraulique",
    "Gaz (MW)": "gaz",
    "Taux de CO2 (g/kWh)": "co2",
    "Date": "date",
    "Heures": "heures",
}

TARGET = "consommation"
BENCHMARK_COL = "prevision_j1"
NUMERIC_FEATURES = [
    "prevision_j1", "nucleaire", "eolien", "solaire", "hydraulique", "gaz", "co2",
]

JOURS_FERIES_FR = {
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18",
    "2023-05-29", "2023-07-14", "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-09",
    "2024-05-20", "2024-07-14", "2024-08-15", "2024-11-01", "2024-11-11", "2024-12-25",
}

FEATURE_COLS = [
    "prevision_j1", "nucleaire", "eolien", "solaire", "hydraulique", "gaz", "co2",
    "consommation_max", "consommation_min",
    "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday", "saison",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_7", "prevision_j1_lag1",
]
