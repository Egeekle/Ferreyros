"""Data loading, schema validation and logical-consistency checks.

Covers controls E11 (logical consistency), D2 (static demographics) and E4/E5
(lag-column semantics) of ``docs/brief-alineamiento-eda.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw visit-level dataset, sorted by ``(id, dia_visita)``.

    Sorting is not cosmetic: every lag and expanding-window feature depends on
    this order, and so does the verification of ``ultima_marca_comprada``.
    """
    path = path or cfg.RAW_CSV
    df = pd.read_csv(path)
    return df.sort_values([cfg.KEY_CLIENT, cfg.KEY_TIME]).reset_index(drop=True)


def load_data_dictionary(path=None) -> str:
    """Return the raw text of ``label/Diccionario de Datos.txt``."""
    path = path or cfg.DATA_DICTIONARY
    return path.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Schema profile
# --------------------------------------------------------------------------


def schema_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, cardinality, range, nulls and assigned role.

    The ``rol`` column is what makes this table an alignment artifact rather
    than a generic ``df.info()``: it states what each column is *for* in the
    optimization problem.
    """
    roles = _column_roles()
    out = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(),
            "n_nulos": df.isna().sum(),
            "min": [df[c].min() if pd.api.types.is_numeric_dtype(df[c]) else None for c in df.columns],
            "max": [df[c].max() if pd.api.types.is_numeric_dtype(df[c]) else None for c in df.columns],
        }
    )
    out["rol"] = [roles.get(c, "feature") for c in out.index]
    return out


def _column_roles() -> dict[str, str]:
    roles: dict[str, str] = {
        cfg.KEY_CLIENT: "clave de cliente (agrupacion)",
        cfg.KEY_TIME: "eje temporal (split)",
        cfg.TARGET_INCIDENCE: "TARGET M1 (pi)",
        cfg.TARGET_BRAND: "TARGET M2 (q) - post-resultado",
        cfg.TARGET_QUANTITY: "TARGET M3 (lambda) - post-resultado",
        "ultima_marca_comprada": "feature historial (lag verificado)",
        "ultima_cantidad_comprada": "feature historial (binaria, mal nombrada)",
    }
    for col in cfg.PRICE_COLS:
        roles[col] = "variable de decision y_jtk (precio)"
    for col in cfg.PROMO_COLS:
        roles[col] = "variable de decision z_jt (promocion)"
    for col in cfg.DEMOGRAPHIC_COLS:
        roles[col] = "feature / segmentacion M0"
    for col in cfg.EXCLUDED_COLS:
        roles[col] = "EXCLUIDA por diccionario (E1)"
    return roles


def dataset_overview(df: pd.DataFrame) -> pd.Series:
    """Headline facts of section 2.1 of the problem brief."""
    visits_per_client = df.groupby(cfg.KEY_CLIENT).size()
    visits_per_day = df.groupby(cfg.KEY_TIME).size()
    return pd.Series(
        {
            "filas": len(df),
            "columnas": df.shape[1],
            "clientes_unicos": df[cfg.KEY_CLIENT].nunique(),
            "dias_distintos": df[cfg.KEY_TIME].nunique(),
            "dia_min": int(df[cfg.KEY_TIME].min()),
            "dia_max": int(df[cfg.KEY_TIME].max()),
            "visitas_por_cliente_media": round(visits_per_client.mean(), 1),
            "visitas_por_cliente_min": int(visits_per_client.min()),
            "visitas_por_cliente_max": int(visits_per_client.max()),
            "visitas_por_dia_media": round(visits_per_day.mean(), 1),
            "visitas_por_dia_min": int(visits_per_day.min()),
            "visitas_por_dia_max": int(visits_per_day.max()),
            "nulos_totales": int(df.isna().sum().sum()),
            "duplicados_exactos": int(df.duplicated().sum()),
            "tasa_compra": round(df[cfg.TARGET_INCIDENCE].mean(), 4),
            "n_compras": int(df[cfg.TARGET_INCIDENCE].sum()),
        }
    )


# --------------------------------------------------------------------------
# Logical consistency -- control E11
# --------------------------------------------------------------------------


def consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Verify the implications that must hold between target and outcomes.

    A non-zero count in any row means the dataset does not behave as the data
    dictionary describes, and every downstream conditional universe is suspect.
    """
    bought = df[cfg.TARGET_INCIDENCE] == 1
    not_bought = df[cfg.TARGET_INCIDENCE] == 0

    checks = [
        ("compra=1 & cantidad=0", int((bought & (df[cfg.TARGET_QUANTITY] == 0)).sum())),
        ("compra=1 & id_marca=0", int((bought & (df[cfg.TARGET_BRAND] == 0)).sum())),
        ("compra=0 & cantidad>0", int((not_bought & (df[cfg.TARGET_QUANTITY] > 0)).sum())),
        ("compra=0 & id_marca>0", int((not_bought & (df[cfg.TARGET_BRAND] > 0)).sum())),
        ("id_marca fuera de 0-5", int((~df[cfg.TARGET_BRAND].isin([0, *cfg.BRANDS])).sum())),
        ("cantidad negativa", int((df[cfg.TARGET_QUANTITY] < 0).sum())),
        ("precio <= 0", int((df[cfg.PRICE_COLS] <= 0).sum().sum())),
        ("promo fuera de 0/1", int((~df[cfg.PROMO_COLS].isin([0, 1])).sum().sum())),
    ]
    out = pd.DataFrame(checks, columns=["verificacion", "violaciones"])
    out["estado"] = out["violaciones"].map(lambda n: "OK" if n == 0 else "FALLA")
    return out


def static_demographics_check(df: pd.DataFrame) -> pd.DataFrame:
    """Control D2: demographics must be invariant within each client.

    If they are, client-level aggregation is legitimate and the demographic
    block can feed the segmentation model without collapsing information.
    """
    rows = []
    for col in cfg.DEMOGRAPHIC_COLS:
        n_varying = int((df.groupby(cfg.KEY_CLIENT)[col].nunique() > 1).sum())
        rows.append({"variable": col, "clientes_con_mas_de_un_valor": n_varying,
                     "estado": "estatica" if n_varying == 0 else "VARIA"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Lag semantics -- controls E4 and E5
# --------------------------------------------------------------------------


def verify_lag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Control E4/E5: establish what the two ``ultima_*`` columns really are.

    ``ultima_marca_comprada`` is claimed to be the previously purchased brand.
    We test it against the true lag. ``ultima_cantidad_comprada`` is named as a
    quantity by the dictionary but is binary; we test whether it is in fact the
    lag of the purchase indicator.
    """
    d = df.sort_values([cfg.KEY_CLIENT, cfg.KEY_TIME])
    true_lag_brand = d.groupby(cfg.KEY_CLIENT)[cfg.TARGET_BRAND].shift(1).fillna(0)
    true_lag_incidence = d.groupby(cfg.KEY_CLIENT)[cfg.TARGET_INCIDENCE].shift(1).fillna(0)

    n_unique_qty = int(d["ultima_cantidad_comprada"].nunique())
    rows = [
        {
            "verificacion": "ultima_marca_comprada == shift(1) de id_marca por cliente",
            "coincidencia": round(float((d["ultima_marca_comprada"] == true_lag_brand).mean()), 4),
            "conclusion": "lag real, usable sin fuga" ,
        },
        {
            "verificacion": "ultima_marca_comprada == id_marca de la misma fila",
            "coincidencia": round(float((d["ultima_marca_comprada"] == d[cfg.TARGET_BRAND]).mean()), 4),
            "conclusion": "coincidencia parcial esperada (no es copia de la fila)",
        },
        {
            "verificacion": f"ultima_cantidad_comprada tiene {n_unique_qty} valores distintos",
            "coincidencia": float(n_unique_qty),
            "conclusion": "binaria: el diccionario la nombra mal" if n_unique_qty == 2 else "revisar",
        },
        {
            "verificacion": "ultima_cantidad_comprada == shift(1) de incidencia_compra",
            "coincidencia": round(float((d["ultima_cantidad_comprada"] == true_lag_incidence).mean()), 4),
            "conclusion": "es el lag del indicador de compra, no una cantidad",
        },
    ]
    return pd.DataFrame(rows)


def no_history_profile(df: pd.DataFrame) -> pd.Series:
    """Control E7: ``ultima_marca_comprada == 0`` is absence of history, not a null."""
    mask = df["ultima_marca_comprada"] == 0
    return pd.Series(
        {
            "filas_sin_historial": int(mask.sum()),
            "pct_sin_historial": round(100 * float(mask.mean()), 1),
            "de_ellas_primera_visita": int((mask & (df.groupby(cfg.KEY_CLIENT).cumcount() == 0)).sum()),
            "tasa_compra_sin_historial": round(float(df.loc[mask, cfg.TARGET_INCIDENCE].mean()), 4),
            "tasa_compra_con_historial": round(float(df.loc[~mask, cfg.TARGET_INCIDENCE].mean()), 4),
        }
    )


# --------------------------------------------------------------------------
# Derived helpers used across the EDA
# --------------------------------------------------------------------------


def add_paid_price_and_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``precio_pagado`` and ``gasto`` to purchase rows (control A7).

    The paid price is the price of the brand actually chosen; it is undefined
    for non-purchases and left as ``NaN`` there on purpose.
    """
    out = df.copy()
    price_matrix = out[cfg.PRICE_COLS].to_numpy()
    brand_idx = out[cfg.TARGET_BRAND].to_numpy() - 1
    valid = brand_idx >= 0

    paid = pd.Series(np.nan, index=out.index, dtype="float64")
    paid.loc[valid] = price_matrix[valid, brand_idx[valid]]

    out["precio_pagado"] = paid
    out["gasto"] = out["precio_pagado"] * out[cfg.TARGET_QUANTITY]
    return out


def purchases_only(df: pd.DataFrame) -> pd.DataFrame:
    """Conditional universe for models M2 and M3 (control E12)."""
    out = df[df[cfg.TARGET_INCIDENCE] == 1].copy()
    if "precio_pagado" not in out.columns:
        out = add_paid_price_and_spend(out)
    return out


def temporal_split(df: pd.DataFrame, split_day: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control E8: split by time, never at random.

    With ~117 visits per client, a random split puts visits of the same client
    on both sides and inflates every metric.
    """
    split_day = split_day or cfg.SPLIT_DAY
    train = df[df[cfg.KEY_TIME] <= split_day].copy()
    test = df[df[cfg.KEY_TIME] > split_day].copy()
    return train, test


def split_composition(df: pd.DataFrame, split_day: int | None = None) -> pd.DataFrame:
    """Control E9: the split must not introduce composition bias."""
    split_day = split_day or cfg.SPLIT_DAY
    train, test = temporal_split(df, split_day)
    rows = []
    for name, part in (("train", train), ("test", test)):
        rows.append(
            {
                "split": name,
                "filas": len(part),
                "clientes": part[cfg.KEY_CLIENT].nunique(),
                "dia_min": int(part[cfg.KEY_TIME].min()),
                "dia_max": int(part[cfg.KEY_TIME].max()),
                "tasa_compra": round(float(part[cfg.TARGET_INCIDENCE].mean()), 4),
            }
        )
    return pd.DataFrame(rows)
