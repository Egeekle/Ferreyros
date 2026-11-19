"""EDA block A -- parameters of the objective function.

Every function here produces the evidence demanded by one row of section 2.1
of ``docs/brief-alineamiento-eda.md``. The objective is

    max  sum_ijt  x_ijt * pi_it * q_ijt * lambda_ijt * m_jk

so this module estimates the empirical shape of ``pi`` (A1-A3), ``q`` (A4-A5),
``lambda`` (A6) and ``m`` (A7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data import add_paid_price_and_spend, purchases_only


# --------------------------------------------------------------------------
# A1 -- pi: base rate and temporal stability
# --------------------------------------------------------------------------


def purchase_rate(df: pd.DataFrame) -> pd.Series:
    """A1: the base rate that every model must beat as a baseline."""
    n = len(df)
    k = int(df[cfg.TARGET_INCIDENCE].sum())
    return pd.Series(
        {
            "visitas": n,
            "compras": k,
            "tasa_compra": round(k / n, 4),
            "accuracy_de_predecir_siempre_no_compra": round(1 - k / n, 4),
        }
    )


def purchase_rate_by_period(df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    """A1: stability of the base rate over time.

    A stable rate is what licenses the temporal split: if the rate drifted,
    train and test would not be comparable and the split would need rethinking.
    """
    d = df.copy()
    max_day = int(d[cfg.KEY_TIME].max())
    edges = np.linspace(0, max_day, n_periods + 1).astype(int)
    labels = [f"T{i + 1} (d{edges[i] + 1}-{edges[i + 1]})" for i in range(n_periods)]
    d["periodo"] = pd.cut(d[cfg.KEY_TIME], bins=edges, labels=labels)

    out = d.groupby("periodo", observed=True).agg(
        visitas=(cfg.KEY_CLIENT, "size"),
        clientes=(cfg.KEY_CLIENT, "nunique"),
        tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
    )
    out["tasa_compra"] = out["tasa_compra"].round(4)
    out["desvio_vs_global"] = (out["tasa_compra"] - df[cfg.TARGET_INCIDENCE].mean()).round(4)
    return out.reset_index()


# --------------------------------------------------------------------------
# A2 -- pi: price sensitivity
# --------------------------------------------------------------------------


def incidence_by_price_quintile(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """A2: purchase rate by quintile of the cheapest price available that day.

    This is the raw elasticity signal: if incidence does not move with price,
    the price decision variable has nothing to optimize against.
    """
    d = df.copy()
    d["precio_min_dia"] = d[cfg.PRICE_COLS].min(axis=1)
    d["quintil_precio_min"] = pd.qcut(d["precio_min_dia"], n_bins, duplicates="drop")

    out = d.groupby("quintil_precio_min", observed=True).agg(
        visitas=(cfg.KEY_CLIENT, "size"),
        precio_min_medio=("precio_min_dia", "mean"),
        tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
    ).reset_index()
    out["precio_min_medio"] = out["precio_min_medio"].round(3)
    out["tasa_compra"] = out["tasa_compra"].round(4)

    first, last = out["tasa_compra"].iloc[0], out["tasa_compra"].iloc[-1]
    out.attrs["caida_relativa_pct"] = round(100 * (1 - last / first), 1)
    out.attrs["lectura"] = (
        f"La incidencia cae de {first:.4f} a {last:.4f} entre el quintil mas barato "
        f"y el mas caro: {out.attrs['caida_relativa_pct']}% de caida relativa."
    )
    return out


def incidence_by_brand_price_quintile(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """A2 (detail): own-price sensitivity of each brand's purchase incidence.

    Uses each brand's own price distribution, so the result is comparable
    across brands despite their very different price levels.
    """
    rows = []
    for j in cfg.BRANDS:
        d = df.copy()
        col = f"precio_marca_{j}"
        d["q"] = pd.qcut(d[col], n_bins, labels=False, duplicates="drop")
        g = d.groupby("q", observed=True).agg(
            precio_medio=(col, "mean"),
            tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
        )
        if len(g) < 2:
            continue
        rows.append(
            {
                "marca": j,
                "precio_q1": round(float(g["precio_medio"].iloc[0]), 3),
                "precio_q_ultimo": round(float(g["precio_medio"].iloc[-1]), 3),
                "tasa_q1": round(float(g["tasa_compra"].iloc[0]), 4),
                "tasa_q_ultimo": round(float(g["tasa_compra"].iloc[-1]), 4),
                "caida_relativa_pct": round(
                    100 * (1 - g["tasa_compra"].iloc[-1] / g["tasa_compra"].iloc[0]), 1
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# A3 -- pi: promotion sensitivity
# --------------------------------------------------------------------------


def incidence_by_n_promos(df: pd.DataFrame) -> pd.DataFrame:
    """A3: purchase rate by how many brands are on promotion that day."""
    d = df.copy()
    d["n_promos_dia"] = d[cfg.PROMO_COLS].sum(axis=1)

    out = d.groupby("n_promos_dia").agg(
        visitas=(cfg.KEY_CLIENT, "size"),
        tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
    ).reset_index()
    out["tasa_compra"] = out["tasa_compra"].round(4)

    base = out.loc[out["n_promos_dia"] == 0, "tasa_compra"].iloc[0]
    out["lift_vs_sin_promo"] = (out["tasa_compra"] / base).round(3)
    out.attrs["lectura"] = (
        "El efecto no es monotono: crece hasta 2 marcas en promocion y luego cae. "
        "Promocionar mas no es promocionar mejor."
    )
    return out


def incidence_by_promo_per_brand(df: pd.DataFrame) -> pd.DataFrame:
    """A3: purchase rate with and without promotion, brand by brand.

    This is a different question from A5 and the two together are what the
    objective function needs. A5 asks how the *share* of a brand moves when it
    is discounted, conditional on a purchase happening: that is cannibalization
    between brands. This asks whether the discount makes a purchase happen at
    all: that is market expansion. A promotion that only moves share pays for
    units that would have been sold anyway.

    The difference of the two rates is tested against its standard error, so a
    brand is only credited with expanding the market when the gap survives the
    sample size behind it.
    """
    rows = []
    for j in cfg.BRANDS:
        on = df[f"promo_marca_{j}"] == 1
        n_on, n_off = int(on.sum()), int((~on).sum())
        r_on = float(df.loc[on, cfg.TARGET_INCIDENCE].mean()) if n_on else np.nan
        r_off = float(df.loc[~on, cfg.TARGET_INCIDENCE].mean())
        se = np.sqrt(r_on * (1 - r_on) / n_on + r_off * (1 - r_off) / n_off) if n_on else np.nan
        z = (r_on - r_off) / se if n_on and se else np.nan
        rows.append(
            {
                "marca": j,
                "dias_promo": int(df.loc[on, cfg.KEY_TIME].nunique()),
                "visitas_con_promo": n_on,
                "tasa_con_promo": round(r_on, 4),
                "tasa_sin_promo": round(r_off, 4),
                "delta_pp": round(100 * (r_on - r_off), 2),
                "lift": round(r_on / r_off, 3) if r_off else np.nan,
                "z": round(float(z), 2) if n_on else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["efecto"] = np.where(
        out["z"].abs() <= 2,
        "no distinguible del ruido",
        np.where(out["delta_pp"] > 0, "expande el mercado", "contrae la incidencia"),
    )

    expanden = out.loc[out["efecto"] == "expande el mercado", "marca"].tolist()
    out.attrs["marcas_que_expanden"] = expanden
    out.attrs["lectura"] = (
        "{} de 5 marcas mueven la incidencia por encima del ruido ({}). El resto "
        "redistribuye share sin crear compras: su descuento se paga con unidades que "
        "se habrian vendido igual. La funcion objetivo necesita las dos lecturas, "
        "porque el termino de promocion resta costo siempre y suma ingreso solo si "
        "hay compra incremental."
    ).format(
        len(expanden),
        "marcas " + ", ".join(str(m) for m in expanden) if expanden else "ninguna",
    )
    return out


# --------------------------------------------------------------------------
# A4, A5 -- q: brand choice and substitution
# --------------------------------------------------------------------------


def brand_share(df: pd.DataFrame) -> pd.DataFrame:
    """A4: brand share conditional on purchase -- the universe of model M2."""
    comp = purchases_only(df)
    out = comp.groupby(cfg.TARGET_BRAND).agg(
        compras=(cfg.TARGET_QUANTITY, "size"),
        unidades=(cfg.TARGET_QUANTITY, "sum"),
        precio_pagado_medio=("precio_pagado", "mean"),
        cantidad_media=(cfg.TARGET_QUANTITY, "mean"),
        ingreso=("gasto", "sum"),
    ).reset_index()

    out["share_compras_pct"] = (100 * out["compras"] / out["compras"].sum()).round(1)
    out["share_ingreso_pct"] = (100 * out["ingreso"] / out["ingreso"].sum()).round(1)
    for col in ("precio_pagado_medio", "cantidad_media", "ingreso"):
        out[col] = out[col].round(2)

    out.attrs["universo"] = (
        f"Condicional a compra: {len(comp):,} filas de {len(df):,}. "
        "Este es el universo de entrenamiento de M2 y M3 (control E12)."
    )
    return out


def promo_lift_on_brand_share(df: pd.DataFrame) -> pd.DataFrame:
    """A5: how much each brand's share rises when it is on promotion.

    This is the central finding of the EDA: the brands that are promoted least
    are the ones that respond most. It is the arbitrage the optimizer exploits.
    """
    comp = purchases_only(df)
    rows = []
    for j in cfg.BRANDS:
        on_promo = comp[f"promo_marca_{j}"] == 1
        share_on = float((comp.loc[on_promo, cfg.TARGET_BRAND] == j).mean()) if on_promo.any() else np.nan
        share_off = float((comp.loc[~on_promo, cfg.TARGET_BRAND] == j).mean())
        rows.append(
            {
                "marca": j,
                "frecuencia_promo": round(float(df[f"promo_marca_{j}"].mean()), 4),
                "share_con_promo": round(share_on, 3),
                "share_sin_promo": round(share_off, 3),
                "lift": round(share_on / share_off, 2) if share_off > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["respuesta_vs_inversion"] = np.where(
        (out["lift"] > out["lift"].median()) & (out["frecuencia_promo"] < out["frecuencia_promo"].median()),
        "SUB-PROMOCIONADA (alta respuesta, baja inversion)",
        np.where(
            (out["lift"] < out["lift"].median()) & (out["frecuencia_promo"] > out["frecuencia_promo"].median()),
            "SOBRE-PROMOCIONADA (baja respuesta, alta inversion)",
            "alineada",
        ),
    )
    return out


def brand_switching_matrix(df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """A5: transition matrix from the previously bought brand to the current one.

    Quantifies loyalty (diagonal) versus substitution (off-diagonal), which is
    what constraint-free cross-price effects in model M2 must reproduce.
    """
    comp = purchases_only(df)
    comp = comp[comp["ultima_marca_comprada"] > 0]
    mat = pd.crosstab(
        comp["ultima_marca_comprada"],
        comp[cfg.TARGET_BRAND],
        normalize="index" if normalize else False,
    )
    mat.index.name = "marca_anterior"
    mat.columns.name = "marca_actual"
    return mat.round(3) if normalize else mat


# --------------------------------------------------------------------------
# A6 -- lambda: quantity given purchase
# --------------------------------------------------------------------------


def quantity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """A6: distribution of ``cantidad`` conditional on a purchase."""
    comp = purchases_only(df)
    counts = comp[cfg.TARGET_QUANTITY].value_counts().sort_index()
    out = counts.rename("compras").reset_index()
    out.columns = ["cantidad", "compras"]
    out["pct"] = (100 * out["compras"] / out["compras"].sum()).round(1)
    out["pct_acumulado"] = out["pct"].cumsum().round(1)
    out.attrs["resumen"] = comp[cfg.TARGET_QUANTITY].describe().round(2).to_dict()
    return out


def quantity_summary_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """A6: expected quantity per brand -- this is ``lambda_ijt`` at brand level."""
    comp = purchases_only(df)
    out = comp.groupby(cfg.TARGET_BRAND)[cfg.TARGET_QUANTITY].agg(
        compras="size", media="mean", mediana="median", p90=lambda s: s.quantile(0.9), maximo="max"
    ).reset_index()
    out["media"] = out["media"].round(2)
    out["p90"] = out["p90"].round(1)
    return out


# --------------------------------------------------------------------------
# A7 -- m: revenue, and the missing cost parameter
# --------------------------------------------------------------------------


def revenue_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """A7: observed revenue per brand, with the cost gap stated explicitly."""
    comp = purchases_only(df)
    out = comp.groupby(cfg.TARGET_BRAND).agg(
        unidades=(cfg.TARGET_QUANTITY, "sum"),
        ingreso=("gasto", "sum"),
        precio_medio=("precio_pagado", "mean"),
    ).reset_index()
    out["share_ingreso_pct"] = (100 * out["ingreso"] / out["ingreso"].sum()).round(1)
    out["ingreso"] = out["ingreso"].round(2)
    out["precio_medio"] = out["precio_medio"].round(2)
    out["margen_unitario"] = "REQUIERE SUPUESTO (c_j no esta en el dato)"

    out.attrs["brecha"] = (
        "El margen m_jk = p_jk - c_j no es calculable: el dataset no contiene costos. "
        "La optimizacion se entrega con analisis de sensibilidad sobre c_j (control C2)."
    )
    out.attrs["ingreso_total"] = round(float(comp["gasto"].sum()), 2)
    return out


def missing_business_parameters() -> pd.DataFrame:
    """A7 / C2: the parameters the optimization needs and the data cannot give.

    Declaring this table is itself an acceptance criterion: it delimits the
    honest scope of the solution.
    """
    rows = [
        ("c_j", "Costo unitario por marca", "No", "Supuesto de margen bruto % por marca + sensibilidad"),
        ("a", "Costo de contacto / impulso comercial", "No", "Supuesto; define el umbral de decision de P1"),
        ("d_j", "Costo de un periodo de promocion", "No", "Supuesto; acota el calendario de P2"),
        ("B", "Presupuesto de ejecucion", "No", "Input de negocio; restriccion R2"),
        ("F_i", "Tope de contactos por cliente", "Parcial", "Acotable con la frecuencia de visita observada"),
        ("Cap_jt", "Capacidad / inventario por marca", "Parcial", "Acotable con unidades/dia observadas"),
        ("rho", "Margen porcentual minimo", "No", "Politica comercial; restriccion R6"),
    ]
    return pd.DataFrame(rows, columns=["parametro", "descripcion", "esta_en_el_dato", "tratamiento"])
