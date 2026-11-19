"""EDA -- anomalies.

Fase 1 of ``docs/brief-metodologia.md`` asks for three specific anomalies:
"visitas con ``incidencia_compra = 1`` y ``cantidad = 0``, clientes con
comportamiento atipico, cantidades extremas (maximo observado: 15)".

The first one is coherence of the three outcome columns and is also checked by
``data.consistency_checks``; here it is reported next to the other two so the
anomaly block of the notebook is one place and not three.

The rule this module follows: an anomaly is **described and decided**, never
silently dropped. Every row of the summary carries the decision taken about it,
because a filter applied in the EDA and forgotten in the modelling notebook is
the cheapest way to make two notebooks disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from ..data import add_paid_price_and_spend, purchases_only


# --------------------------------------------------------------------------
# Coherence between the three outcome columns
# --------------------------------------------------------------------------


def coherence_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Every way in which incidence, brand and quantity could contradict.

    Enumerating the impossible combinations and finding them empty is what
    licenses using ``incidencia_compra`` as the target of M1 and the
    purchases-only universe for M2 and M3.
    """
    inc = df[cfg.TARGET_INCIDENCE]
    marca = df[cfg.TARGET_BRAND]
    cant = df[cfg.TARGET_QUANTITY]
    n = len(df)

    casos = {
        "incidencia=1 y cantidad=0": (inc == 1) & (cant == 0),
        "incidencia=1 y id_marca=0": (inc == 1) & (marca == 0),
        "incidencia=0 y cantidad>0": (inc == 0) & (cant > 0),
        "incidencia=0 y id_marca>0": (inc == 0) & (marca > 0),
        "id_marca=0 y cantidad>0": (marca == 0) & (cant > 0),
        "id_marca>0 y cantidad=0": (marca > 0) & (cant == 0),
    }

    filas = []
    for nombre, mask in casos.items():
        k = int(mask.sum())
        filas.append(
            {
                "caso": nombre,
                "es_posible": "no",
                "filas": k,
                "pct": round(100 * k / n, 4),
                "decision": "nada que corregir" if k == 0 else "REVISAR antes de modelar",
            }
        )

    out = pd.DataFrame(filas)
    total = int(out["filas"].sum())
    out.attrs["incoherencias"] = total
    out.attrs["coherente"] = total == 0
    out.attrs["lectura"] = (
        "Las seis combinaciones imposibles suman {} filas sobre {:,}. {}"
    ).format(
        total,
        n,
        "Las tres columnas de resultado son coherentes entre si, asi que el universo "
        "condicional de M2 y M3 se define solo con incidencia_compra."
        if total == 0
        else "Hay filas que contradicen el contrato de datos y bloquean el modelado.",
    )
    return out


# --------------------------------------------------------------------------
# Extreme quantities
# --------------------------------------------------------------------------


def quantity_outliers(df: pd.DataFrame, cola: int = 6) -> pd.DataFrame:
    """Distribution of ``cantidad`` on purchases, with its upper tail opened.

    ``cola`` is the value from which the tail is reported as a block. The
    question the table answers is not "is 15 an outlier" but "how much of the
    revenue lives above the body of the distribution", because that is what
    decides whether M3 can be a mean-based model.
    """
    compras = purchases_only(add_paid_price_and_spend(df))
    n = len(compras)

    tabla = (
        compras.groupby(cfg.TARGET_QUANTITY, observed=True)
        .agg(
            compras=(cfg.KEY_CLIENT, "size"),
            clientes=(cfg.KEY_CLIENT, "nunique"),
            unidades=(cfg.TARGET_QUANTITY, "sum"),
            ingreso=("gasto", "sum"),
        )
        .reset_index()
        .rename(columns={cfg.TARGET_QUANTITY: "cantidad"})
    )
    tabla["pct_compras"] = (100 * tabla["compras"] / n).round(2)
    tabla["pct_unidades"] = (100 * tabla["unidades"] / tabla["unidades"].sum()).round(2)
    tabla["pct_ingreso"] = (100 * tabla["ingreso"] / tabla["ingreso"].sum()).round(2)
    tabla["pct_compras_acum"] = tabla["pct_compras"].cumsum().round(2)
    tabla["ingreso"] = tabla["ingreso"].round(2)

    cola_mask = tabla["cantidad"] >= cola
    serie = compras[cfg.TARGET_QUANTITY]

    tabla.attrs["maximo"] = int(serie.max())
    tabla.attrs["media"] = round(float(serie.mean()), 4)
    tabla.attrs["mediana"] = float(serie.median())
    tabla.attrs["p99"] = float(serie.quantile(0.99))
    tabla.attrs["umbral_cola"] = cola
    tabla.attrs["pct_compras_en_cola"] = round(float(tabla.loc[cola_mask, "pct_compras"].sum()), 2)
    tabla.attrs["pct_unidades_en_cola"] = round(
        float(tabla.loc[cola_mask, "pct_unidades"].sum()), 2
    )
    tabla.attrs["clientes_en_cola"] = int(
        compras.loc[compras[cfg.TARGET_QUANTITY] >= cola, cfg.KEY_CLIENT].nunique()
    )
    tabla.attrs["lectura"] = (
        "Maximo {} unidades, media {:.2f}, mediana {:.0f}, p99 {:.0f}. La cola de "
        "cantidad >= {} es el {:.2f}% de las compras pero el {:.2f}% de las unidades, "
        "repartida en {} clientes distintos: es comportamiento de cliente mayorista, "
        "no error de captura."
    ).format(
        tabla.attrs["maximo"],
        tabla.attrs["media"],
        tabla.attrs["mediana"],
        tabla.attrs["p99"],
        cola,
        tabla.attrs["pct_compras_en_cola"],
        tabla.attrs["pct_unidades_en_cola"],
        tabla.attrs["clientes_en_cola"],
    )
    return tabla


# --------------------------------------------------------------------------
# Atypical clients
# --------------------------------------------------------------------------


def atypical_clients(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Clients whose purchase rate or visit count sits beyond ``z`` sigmas.

    Two reasons to look: an extreme client can dominate a metric, and the test
    split keeps the same 500 clients, so an atypical client is atypical on both
    sides of the boundary and never averages out.
    """
    dfs = add_paid_price_and_spend(df)
    perfil = (
        dfs.groupby(cfg.KEY_CLIENT, observed=True)
        .agg(
            visitas=(cfg.KEY_TIME, "size"),
            compras=(cfg.TARGET_INCIDENCE, "sum"),
            tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
            unidades=(cfg.TARGET_QUANTITY, "sum"),
            gasto=("gasto", "sum"),
        )
        .reset_index()
    )

    for col, nombre in (("tasa_compra", "z_tasa"), ("visitas", "z_visitas"), ("gasto", "z_gasto")):
        mu = float(perfil[col].mean())
        sd = float(perfil[col].std(ddof=0))
        perfil[nombre] = ((perfil[col] - mu) / sd).round(2) if sd else 0.0

    perfil["atipico"] = (
        (perfil["z_tasa"].abs() > z) | (perfil["z_visitas"].abs() > z) | (perfil["z_gasto"].abs() > z)
    )
    perfil["motivo"] = np.select(
        [
            perfil["z_tasa"].abs() > z,
            perfil["z_visitas"].abs() > z,
            perfil["z_gasto"].abs() > z,
        ],
        ["tasa de compra", "numero de visitas", "gasto"],
        default="",
    )
    perfil["tasa_compra"] = perfil["tasa_compra"].round(4)
    perfil["gasto"] = perfil["gasto"].round(2)

    atipicos = perfil.loc[perfil["atipico"]].sort_values("z_tasa", ascending=False)
    ids = atipicos[cfg.KEY_CLIENT].tolist()

    tasa_global = float(df[cfg.TARGET_INCIDENCE].mean())
    sin_atipicos = df.loc[~df[cfg.KEY_CLIENT].isin(ids), cfg.TARGET_INCIDENCE].mean()
    sin_atipicos = float(sin_atipicos) if len(ids) else tasa_global

    atipicos.attrs["n_atipicos"] = len(ids)
    atipicos.attrs["ids"] = ids
    atipicos.attrs["umbral_z"] = z
    atipicos.attrs["tasa_global"] = round(tasa_global, 4)
    atipicos.attrs["tasa_sin_atipicos"] = round(sin_atipicos, 4)
    atipicos.attrs["delta_tasa"] = round(sin_atipicos - tasa_global, 4)
    atipicos.attrs["pct_ingreso"] = round(
        100 * float(perfil.loc[perfil["atipico"], "gasto"].sum()) / float(perfil["gasto"].sum()), 2
    )
    atipicos.attrs["rango_tasa"] = (
        round(float(perfil["tasa_compra"].min()), 4),
        round(float(perfil["tasa_compra"].max()), 4),
    )
    atipicos.attrs["lectura"] = (
        "{} clientes de {} superan {} sigmas en tasa, visitas o gasto, y concentran el "
        "{:.2f}% del ingreso. Excluirlos moveria la tasa global de {:.4f} a {:.4f} "
        "({:+.4f}). No se excluyen: son clientes reales y el optimizador los querra "
        "exactamente a ellos."
    ).format(
        len(ids),
        len(perfil),
        z,
        atipicos.attrs["pct_ingreso"],
        tasa_global,
        sin_atipicos,
        atipicos.attrs["delta_tasa"],
    )
    return atipicos


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """The three anomalies the methodology brief names, with their decision."""
    coh = coherence_anomalies(df)
    qty = quantity_outliers(df)
    atp = atypical_clients(df)

    filas = [
        {
            "anomalia": "incidencia=1 con cantidad=0 (y las otras 5 incoherencias)",
            "hallazgo": "{} filas".format(coh.attrs["incoherencias"]),
            "decision": (
                "no hay nada que filtrar; el target de M1 se usa tal cual"
                if coh.attrs["coherente"]
                else "bloquea el modelado hasta resolverse"
            ),
        },
        {
            "anomalia": "cantidades extremas (maximo {})".format(qty.attrs["maximo"]),
            "hallazgo": "cola >= {} = {:.2f}% de las compras y {:.2f}% de las unidades".format(
                qty.attrs["umbral_cola"],
                qty.attrs["pct_compras_en_cola"],
                qty.attrs["pct_unidades_en_cola"],
            ),
            "decision": (
                "no se truncan; M3 se evalua con MAE ademas de RMSE para que la cola "
                "no dicte el ajuste"
            ),
        },
        {
            "anomalia": "clientes con comportamiento atipico",
            "hallazgo": "{} clientes sobre {} sigmas, {:.2f}% del ingreso; tasa por cliente de {:.4f} a {:.4f}".format(
                atp.attrs["n_atipicos"],
                atp.attrs["umbral_z"],
                atp.attrs["pct_ingreso"],
                atp.attrs["rango_tasa"][0],
                atp.attrs["rango_tasa"][1],
            ),
            "decision": (
                "se conservan; la generalizacion a clientes nuevos se mide con "
                "GroupKFold por id, que es donde ese riesgo se ve"
            ),
        },
    ]
    out = pd.DataFrame(filas)
    out.attrs["lectura"] = (
        "Ninguna anomalia se corrige borrando filas. Las tres cambian como se evalua el "
        "modelo, no el dato que lo entrena."
    )
    return out
