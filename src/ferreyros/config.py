"""Project-wide constants and paths.

Single source of truth for the dataset contract documented in
``docs/brief-problema.md`` and the acceptance criteria of
``docs/brief-alineamiento-eda.md``.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LABEL_DIR = PROJECT_ROOT / "label"
DOCS_DIR = PROJECT_ROOT / "docs"
RAW_CSV = DATA_DIR / "compras_data.csv"
DATA_DICTIONARY = LABEL_DIR / "Diccionario de Datos.txt"

# --------------------------------------------------------------------------
# Dataset contract
# --------------------------------------------------------------------------

BRANDS: tuple[int, ...] = (1, 2, 3, 4, 5)

PRICE_COLS: list[str] = [f"precio_marca_{j}" for j in BRANDS]
PROMO_COLS: list[str] = [f"promo_marca_{j}" for j in BRANDS]

TARGET_INCIDENCE = "incidencia_compra"
TARGET_BRAND = "id_marca"
TARGET_QUANTITY = "cantidad"

KEY_CLIENT = "id"
KEY_TIME = "dia_visita"

DEMOGRAPHIC_COLS: list[str] = [
    "genero",
    "estado_civil",
    "edad",
    "nivel_educacion",
    "ingreso_anual",
    "ocupacion",
]

LAG_COLS: list[str] = ["ultima_marca_comprada", "ultima_cantidad_comprada"]

#: Excluded by explicit instruction of the data dictionary (control E1).
EXCLUDED_COLS: list[str] = ["tamanio_ciudad"]

#: Outcomes of the visit. Never features of the incidence model (controls E2, E3).
POST_OUTCOME_COLS: list[str] = [TARGET_BRAND, TARGET_QUANTITY]

#: Temporal split boundary (control E8). Train <= SPLIT_DAY, test > SPLIT_DAY.
SPLIT_DAY = 550

# --------------------------------------------------------------------------
# Reference figures -- control cell of docs/brief-alineamiento-eda.md section 4
# --------------------------------------------------------------------------

REFERENCE_FIGURES: dict[str, float | int] = {
    "n_rows": 58_693,
    "n_cols": 24,
    "n_clients": 500,
    "n_days": 730,
    "day_min": 1,
    "day_max": 730,
    "n_nulls": 0,
    "n_duplicates": 0,
    "purchase_rate": 0.2494,
    "n_purchases": 14_638,
    "total_revenue": 82_756.58,
    "mean_quantity_given_purchase": 2.77,
    "train_rows": 45_497,
    "train_rate": 0.2498,
    "test_rows": 13_196,
    "test_rate": 0.2480,
    "targeting_lift_top_decile": 2.43,
    "pareto_top_10pct": 37.0,
    "pareto_top_20pct": 52.7,
}

#: Per-brand reference figures, indexed by brand id.
REFERENCE_BY_BRAND: dict[str, dict[int, float]] = {
    "revenue_share_pct": {1: 7.6, 2: 26.3, 3: 7.3, 4: 23.0, 5: 35.8},
    "n_price_points": {1: 37, 2: 30, 3: 21, 4: 26, 5: 44},
    "promo_frequency": {1: 0.3438, 2: 0.3156, 3: 0.0428, 4: 0.1179, 5: 0.0359},
    "promo_share_lift": {1: 1.08, 2: 1.39, 3: 1.86, 4: 1.16, 5: 1.66},
    "units_per_day": {1: 6.46, 2: 17.57, 3: 4.10, 4: 12.16, 5: 15.34},
}

#: Tolerance for reference-figure comparison (relative, except exact ints).
REFERENCE_TOLERANCE = 0.01

# --------------------------------------------------------------------------
# Human-readable labels for categorical codes
# --------------------------------------------------------------------------

EDUCATION_LABELS = {0: "0 - Basico", 1: "1 - Medio", 2: "2 - Superior", 3: "3 - Posgrado"}
OCCUPATION_LABELS = {0: "0 - Desempleado/otro", 1: "1 - Empleado", 2: "2 - Directivo/autonomo"}
GENDER_LABELS = {0: "0", 1: "1"}
MARITAL_LABELS = {0: "0", 1: "1"}

#: Brand display order and palette, consistent across every figure.
BRAND_ORDER: list[int] = list(BRANDS)
BRAND_COLORS: dict[int, str] = {
    1: "#5B8FF9",
    2: "#5AD8A6",
    3: "#F6BD16",
    4: "#E8684A",
    5: "#9270CA",
}
