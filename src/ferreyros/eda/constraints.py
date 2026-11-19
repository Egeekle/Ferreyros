"""EDA block C -- bounds for the optimization constraints.

Produces the evidence demanded by section 2.3 of
``docs/brief-alineamiento-eda.md``. Each function bounds one constraint of the
problem with observed data, so the MILP is parameterized from evidence rather
than from assumption wherever that is possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data import purchases_only
from .decisions import price_overlap_check, promo_calendar_profile  # re-exported


# --------------------------------------------------------------------------
# C4 -- R8: capacity / inventory
# --------------------------------------------------------------------------


def capacity_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """C4: observed units sold per brand, per day -- bounds ``Cap_jt`` in R8."""
    comp = purchases_only(df)
    n_days = int(df[cfg.KEY_TIME].nunique())

    daily_units = (
        comp.groupby([cfg.KEY_TIME, cfg.TARGET_BRAND])[cfg.TARGET_QUANTITY]
        .sum()
        .unstack(fill_value=0)
    )

    rows = []
    for j in cfg.BRANDS:
        serie = daily_units[j] if j in daily_units.columns else pd.Series(0, index=daily_units.index)
        rows.append(
            {
                "marca": j,
                "unidades_totales": int(serie.sum()),
                "unidades_por_dia_media": round(float(serie.sum() / n_days), 2),
                "unidades_por_dia_p95": round(float(serie.quantile(0.95)), 1),
                "unidades_por_dia_max": int(serie.max()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["lectura"] = (
        "La media acota la capacidad en regimen; el p95 y el maximo acotan el pico. "
        "Cap_jt debe fijarse sobre el pico, no sobre la media."
    )
    return out


# --------------------------------------------------------------------------
# C5 -- R3, R4: contact fatigue and operational scale
# --------------------------------------------------------------------------


def fatigue_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """C5: visit frequency per client and per day -- bounds ``F_i`` in R4.

    R3 (one offer per client per period) is feasible by construction; R4 needs
    a realistic cap, and the observed visit frequency is its natural anchor.
    """
    per_client = df.groupby(cfg.KEY_CLIENT).size()
    per_day = df.groupby(cfg.KEY_TIME).size()
    unique_clients_per_day = df.groupby(cfg.KEY_TIME)[cfg.KEY_CLIENT].nunique()
    horizon = int(df[cfg.KEY_TIME].nunique())

    rows = [
        ("visitas por cliente (horizonte completo)", per_client.mean(), per_client.min(), per_client.max()),
        ("visitas por dia", per_day.mean(), per_day.min(), per_day.max()),
        ("clientes distintos por dia", unique_clients_per_day.mean(),
         unique_clients_per_day.min(), unique_clients_per_day.max()),
    ]
    out = pd.DataFrame(rows, columns=["metrica", "media", "min", "max"])
    out["media"] = out["media"].round(1)
    out.attrs["horizonte_dias"] = horizon
    out.attrs["lectura"] = (
        f"Con {per_client.mean():.0f} visitas por cliente en {horizon} dias, el cliente aparece "
        "cada ~6 dias. F_i debe fijarse muy por debajo de esa frecuencia para no saturar."
    )
    return out


def visit_interval_profile(df: pd.DataFrame) -> pd.DataFrame:
    """C5: distribution of the gap in days between consecutive visits."""
    d = df.sort_values([cfg.KEY_CLIENT, cfg.KEY_TIME])
    gaps = d.groupby(cfg.KEY_CLIENT)[cfg.KEY_TIME].diff().dropna()
    desc = gaps.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2)
    return desc.rename("dias_entre_visitas").to_frame().reset_index(names="estadistico")


# --------------------------------------------------------------------------
# C1 -- R5: promotional-day budget
# --------------------------------------------------------------------------


def promo_budget_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """C1: observed promotional days per brand -- bounds ``T_j`` in R5."""
    out = promo_calendar_profile(df)
    horizon = int(df[cfg.KEY_TIME].nunique())
    out["horizonte_dias"] = horizon
    out["T_j_observado"] = out["dias_en_promo"]
    out.attrs["lectura"] = (
        "T_j se fija a partir de lo observado. El total de dias-promocion del horizonte "
        f"({int(out['dias_en_promo'].sum())}) es el presupuesto a reasignar, no a aumentar."
    )
    return out


# --------------------------------------------------------------------------
# C2 -- R6: margin floor (the honest gap)
# --------------------------------------------------------------------------


def margin_floor_gap() -> pd.DataFrame:
    """C2: state that R6 cannot be parameterized from the data.

    Returning this table is the acceptance criterion for C2: the evaluation
    requires the gap to be declared, not silently assumed away.
    """
    rows = [
        {
            "restriccion": "R6 piso de margen",
            "parametro_requerido": "c_j (costo unitario), rho (margen minimo)",
            "disponible_en_el_dato": "No",
            "tratamiento": (
                "Parametrizar como supuesto de margen bruto por marca y entregar el "
                "resultado como analisis de sensibilidad, no como cifra unica."
            ),
        },
        {
            "restriccion": "R2 presupuesto",
            "parametro_requerido": "a (costo de contacto), d_j (costo de promo), B",
            "disponible_en_el_dato": "No",
            "tratamiento": "Input de negocio; el umbral de decision de P1 depende de a.",
        },
        {
            "restriccion": "R7 arquitectura de marcas",
            "parametro_requerido": "Ordenamiento de precios exigido",
            "disponible_en_el_dato": "Parcial (medias ordenadas, rangos solapados)",
            "tratamiento": "Restriccion activable; se reporta el costo de imponerla.",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Consolidated view
# --------------------------------------------------------------------------


def constraint_parameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per constraint: is it bounded by evidence, or by assumption?"""
    cap = capacity_bounds(df)
    fat = fatigue_bounds(df)
    promo = promo_budget_bounds(df)

    rows = [
        {
            "restriccion": "R3 una oferta por cliente y periodo",
            "fuente": "estructura del problema",
            "parametrizada_con": "factible por construccion",
        },
        {
            "restriccion": "R4 fatiga de contacto (F_i)",
            "fuente": "evidencia",
            "parametrizada_con": f"{fat.loc[0, 'media']:.0f} visitas/cliente observadas",
        },
        {
            "restriccion": "R5 dias de promocion (T_j)",
            "fuente": "evidencia",
            "parametrizada_con": f"{int(promo['dias_en_promo'].sum())} dias-promo observados",
        },
        {
            "restriccion": "R6 piso de margen",
            "fuente": "SUPUESTO",
            "parametrizada_con": "c_j y rho no estan en el dato",
        },
        {
            "restriccion": "R7 arquitectura de marcas",
            "fuente": "decision de modelado",
            "parametrizada_con": "rangos de precio solapados",
        },
        {
            "restriccion": "R8 capacidad (Cap_jt)",
            "fuente": "evidencia",
            "parametrizada_con": f"pico diario hasta {int(cap['unidades_por_dia_max'].max())} unidades",
        },
    ]
    return pd.DataFrame(rows)
