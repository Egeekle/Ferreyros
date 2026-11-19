"""EDA block B -- decision variables of the optimization problem.

Produces the evidence demanded by section 2.2 of
``docs/brief-alineamiento-eda.md``:

* ``y_jtk`` -- the discrete price ladder (B1) and its dynamics (B2)
* ``z_jt``  -- the promotional calendar (B3)
* ``x_ijt`` -- targeting: client heterogeneity (B4), value concentration (B5)
               and the baseline the model must beat (B6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data import purchases_only


# --------------------------------------------------------------------------
# B1 -- y_jtk: the discrete price ladder
# --------------------------------------------------------------------------


def price_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """B1: enumerate the price points actually observed for each brand.

    This table is what turns the MINLP into a tractable MILP: the price is not
    a continuous variable, it is a menu of 21 to 44 already-observed points.
    """
    rows = []
    for j in cfg.BRANDS:
        col = f"precio_marca_{j}"
        values = np.sort(df[col].unique())
        rows.append(
            {
                "marca": j,
                "n_puntos_precio": len(values),
                "precio_min": round(float(values[0]), 2),
                "precio_max": round(float(values[-1]), 2),
                "precio_medio": round(float(df[col].mean()), 2),
                "rango_relativo_pct": round(100 * (values[-1] / values[0] - 1), 1),
                "paso_mediano": round(float(np.median(np.diff(values))), 3) if len(values) > 1 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["lectura"] = (
        f"El espacio de decision de precio es finito: "
        f"{' x '.join(str(n) for n in out['n_puntos_precio'])} combinaciones por periodo. "
        "Discretizar permite precalcular pi, q y lambda por escenario y resolver un MILP."
    )
    return out


def price_points(df: pd.DataFrame, brand: int) -> np.ndarray:
    """B1: the sorted menu ``K_j`` of price points for one brand."""
    return np.sort(df[f"precio_marca_{brand}"].unique())


# --------------------------------------------------------------------------
# B2 -- y_jtk: price dynamics
# --------------------------------------------------------------------------


def daily_price_series(df: pd.DataFrame) -> pd.DataFrame:
    """B2: one row per day with the five prices, for the trajectory plot."""
    return df.groupby(cfg.KEY_TIME)[cfg.PRICE_COLS].first().reset_index()


def price_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """B2: do brand prices move together?

    High correlation means the brands are repriced jointly, so the optimizer
    cannot treat each brand's price as an independent lever in practice.
    """
    daily = daily_price_series(df)
    return daily[cfg.PRICE_COLS].corr().round(3)


def price_overlap_check(df: pd.DataFrame) -> pd.DataFrame:
    """C3 / B2: does the brand price architecture hold strictly?

    Means are ordered, but ranges overlap. That makes constraint R7 (strict
    price ordering) a modeling decision rather than a fact of the data.
    """
    ladder = price_ladder(df).set_index("marca")
    rows = []
    for j in cfg.BRANDS[:-1]:
        nxt = j + 1
        overlap = ladder.loc[j, "precio_max"] > ladder.loc[nxt, "precio_min"]
        rows.append(
            {
                "par": f"marca {j} vs marca {nxt}",
                "max_marca_inferior": ladder.loc[j, "precio_max"],
                "min_marca_superior": ladder.loc[nxt, "precio_min"],
                "se_solapan": bool(overlap),
                "medias_ordenadas": bool(ladder.loc[j, "precio_medio"] < ladder.loc[nxt, "precio_medio"]),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["lectura"] = (
        "Las medias estan ordenadas pero los rangos se solapan. El ordenamiento estricto "
        "de precios (R7) es una decision de modelado para proteger el posicionamiento, "
        "no un hecho del dato: se implementa activable y se reporta su costo."
    )
    return out


def relative_price_gap(df: pd.DataFrame) -> pd.DataFrame:
    """B2: relative price of each brand against the daily average and the cheapest.

    The absolute price of a brand says little, because all five move: what a
    client sees is the **gap**. Two readings come out of this table. The level
    of ``precio_relativo`` is the brand's position, which is what feature
    ``precio_relativo_j`` encodes. Its dispersion over time is whether that
    position actually moves -- if the gaps were frozen, no pricing lever would
    exist for the optimizer and own and cross elasticity would be
    unidentifiable from observational data.
    """
    daily = daily_price_series(df)
    media_dia = daily[cfg.PRICE_COLS].mean(axis=1)
    min_dia = daily[cfg.PRICE_COLS].min(axis=1)

    rows = []
    for j in cfg.BRANDS:
        p = daily[f"precio_marca_{j}"]
        rel = p / media_dia
        rows.append(
            {
                "marca": j,
                "precio_medio": round(float(p.mean()), 3),
                "precio_relativo_medio": round(float(rel.mean()), 3),
                "precio_relativo_min": round(float(rel.min()), 3),
                "precio_relativo_max": round(float(rel.max()), 3),
                "desvio_std_relativo": round(float(rel.std()), 4),
                "brecha_vs_mas_barata": round(float((p - min_dia).mean()), 3),
                "pct_dias_mas_barata": round(100 * float((p <= min_dia + 1e-12).mean()), 1),
            }
        )
    out = pd.DataFrame(rows)

    lider = int(out.loc[out["pct_dias_mas_barata"].idxmax(), "marca"])
    mas_movil = int(out.loc[out["desvio_std_relativo"].idxmax(), "marca"])
    out.attrs["marca_mas_barata"] = lider
    out.attrs["marca_mas_movil"] = mas_movil
    out.attrs["dispersion_media"] = round(float(out["desvio_std_relativo"].mean()), 4)
    out.attrs["lectura"] = (
        "La marca {} es la mas barata el {:.1f}% de los dias y la marca {} es la que "
        "mueve mas su posicion relativa (desvio {:.4f}). La dispersion media del precio "
        "relativo es {:.4f}: las brechas se mueven, asi que hay palanca de precio que "
        "optimizar y variacion suficiente para identificar elasticidades."
    ).format(
        lider,
        float(out["pct_dias_mas_barata"].max()),
        mas_movil,
        float(out["desvio_std_relativo"].max()),
        out.attrs["dispersion_media"],
    )
    return out


# --------------------------------------------------------------------------
# B3 -- z_jt: the promotional calendar
# --------------------------------------------------------------------------


def promo_calendar_profile(df: pd.DataFrame) -> pd.DataFrame:
    """B3: how often each brand is promoted, and in what blocks.

    The observed frequency bounds ``T_j`` in constraint R5.
    """
    daily = df.groupby(cfg.KEY_TIME)[cfg.PROMO_COLS].first()
    rows = []
    for j in cfg.BRANDS:
        serie = daily[f"promo_marca_{j}"]
        blocks = (serie.diff() == 1).sum() + int(serie.iloc[0] == 1)
        rows.append(
            {
                "marca": j,
                "dias_en_promo": int(serie.sum()),
                "frecuencia_promo": round(float(serie.mean()), 4),
                "n_bloques_promo": int(blocks),
                "duracion_media_bloque": round(float(serie.sum() / blocks), 1) if blocks else np.nan,
            }
        )
    return pd.DataFrame(rows)


def promo_vs_response(df: pd.DataFrame) -> pd.DataFrame:
    """B3: confront promotional investment with promotional response.

    The core misalignment: brands 1 and 2 absorb a third of the promotional
    days and barely respond; brands 3 and 5 are almost never promoted and
    respond most.
    """
    from .objective import promo_lift_on_brand_share

    lift = promo_lift_on_brand_share(df).set_index("marca")
    revenue = purchases_only(df).groupby(cfg.TARGET_BRAND)["gasto"].sum()
    share = (100 * revenue / revenue.sum()).round(1)

    out = pd.DataFrame(
        {
            "frecuencia_promo": lift["frecuencia_promo"],
            "lift_share_con_promo": lift["lift"],
            "share_ingreso_pct": share,
            "diagnostico": lift["respuesta_vs_inversion"],
        }
    ).reset_index()
    out.attrs["lectura"] = (
        "Hipotesis de arbitraje: reasignar dias de promocion de las marcas 1-2 "
        "hacia 3 y 5. El optimizador (P2) cuantifica el margen liberado."
    )
    return out


# --------------------------------------------------------------------------
# B4 -- x_ijt: client heterogeneity
# --------------------------------------------------------------------------


def client_profile(df: pd.DataFrame) -> pd.DataFrame:
    """B4: one row per client with behaviour and demographics.

    Demographics are taken with ``first`` because control D2 proves they are
    invariant within a client.
    """
    comp_flag = df[cfg.TARGET_INCIDENCE]
    agg = {
        "visitas": (cfg.KEY_TIME, "size"),
        "compras": (cfg.TARGET_INCIDENCE, "sum"),
        "primer_dia": (cfg.KEY_TIME, "min"),
        "ultimo_dia": (cfg.KEY_TIME, "max"),
    }
    for col in cfg.DEMOGRAPHIC_COLS:
        agg[col] = (col, "first")

    out = df.assign(_flag=comp_flag).groupby(cfg.KEY_CLIENT).agg(**agg)
    out["tasa_compra"] = (out["compras"] / out["visitas"]).round(4)

    if "gasto" in df.columns:
        spend = df.groupby(cfg.KEY_CLIENT)["gasto"].sum()
    else:
        spend = purchases_only(df).groupby(cfg.KEY_CLIENT)["gasto"].sum()
    out["gasto_total"] = spend.reindex(out.index).fillna(0).round(2)
    out["ticket_medio"] = (out["gasto_total"] / out["compras"].replace(0, np.nan)).round(2)
    return out.reset_index()


def heterogeneity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """B4: dispersion of purchase rate across clients.

    Without this dispersion the targeting problem would have no solution worth
    computing: every client would be equally worth contacting.
    """
    prof = client_profile(df)
    desc = prof[["visitas", "compras", "tasa_compra", "gasto_total"]].describe().round(3)
    desc.attrs["lectura"] = (
        f"La tasa de compra va de {prof['tasa_compra'].min():.3f} a "
        f"{prof['tasa_compra'].max():.3f} con mediana {prof['tasa_compra'].median():.3f}. "
        "Esa dispersion es la razon de existir del problema de targeting."
    )
    return desc


# --------------------------------------------------------------------------
# B5 -- x_ijt: value concentration
# --------------------------------------------------------------------------


def pareto_concentration(df: pd.DataFrame, quantiles=(0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """B5: share of revenue held by the top-N% of clients."""
    prof = client_profile(df)
    spend = prof.set_index(cfg.KEY_CLIENT)["gasto_total"].sort_values(ascending=False)
    total = spend.sum()

    rows = []
    for q in quantiles:
        n = int(len(spend) * q)
        rows.append(
            {
                "top_pct_clientes": int(q * 100),
                "n_clientes": n,
                "ingreso": round(float(spend.head(n).sum()), 2),
                "share_ingreso_pct": round(100 * float(spend.head(n).sum()) / total, 1),
            }
        )
    return pd.DataFrame(rows)


def pareto_curve(df: pd.DataFrame) -> pd.DataFrame:
    """B5: cumulative revenue share by cumulative client share, for plotting."""
    prof = client_profile(df)
    spend = prof["gasto_total"].sort_values(ascending=False).to_numpy()
    cum = np.cumsum(spend) / spend.sum()
    return pd.DataFrame(
        {
            "pct_clientes": 100 * np.arange(1, len(spend) + 1) / len(spend),
            "pct_ingreso_acumulado": 100 * cum,
        }
    )


# --------------------------------------------------------------------------
# B6 -- x_ijt: the baseline the model must beat
# --------------------------------------------------------------------------


def targeting_baseline_lift(df: pd.DataFrame, deciles=(0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """B6: lift obtained by ranking clients on their historical purchase rate.

    This needs no model. It is the floor the ML model has to clear to justify
    its existence, and the comparison point for P1's objective value.
    """
    prof = client_profile(df)
    base = float(df[cfg.TARGET_INCIDENCE].mean())
    ranked = prof.sort_values("tasa_compra", ascending=False)

    rows = []
    for q in deciles:
        n = int(len(ranked) * q)
        top = ranked.head(n)
        rows.append(
            {
                "top_pct_clientes": int(q * 100),
                "n_clientes": n,
                "tasa_compra_del_grupo": round(float(top["tasa_compra"].mean()), 4),
                "lift_vs_base": round(float(top["tasa_compra"].mean()) / base, 2),
                "share_ingreso_pct": round(100 * float(top["gasto_total"].sum()) / prof["gasto_total"].sum(), 1),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["tasa_base"] = round(base, 4)
    out.attrs["lectura"] = (
        f"Ordenar por tasa historica da lift x{out['lift_vs_base'].iloc[0]} en el decil superior "
        "sin usar ningun modelo. M1 debe superarlo."
    )
    return out
