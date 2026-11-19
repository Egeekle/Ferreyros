"""EDA block D -- demographics and the segmentation model M0.

Produces the evidence demanded by section 2.4 of
``docs/brief-alineamiento-eda.md``. The central conclusion this block must
support is D3: demographics *segment* but do not *predict*, so they belong in
the clustering and in commercial action, not in the predictive core.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from .decisions import client_profile


# --------------------------------------------------------------------------
# D1 -- demographic profile of the 500 clients
# --------------------------------------------------------------------------


def demographic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """D1: distribution of each demographic variable at client level.

    Measured over the 500 clients, not the 58,693 visits: using visit level
    would weight each client by how often they shop.
    """
    prof = client_profile(df)
    rows = []
    for col in cfg.DEMOGRAPHIC_COLS:
        serie = prof[col]
        rows.append(
            {
                "variable": col,
                "n_niveles": int(serie.nunique()),
                "tipo": "continua" if serie.nunique() > 10 else "categorica/ordinal",
                "media": round(float(serie.mean()), 2),
                "min": round(float(serie.min()), 2),
                "p50": round(float(serie.median()), 2),
                "max": round(float(serie.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def categorical_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """D1: level-by-level counts for the categorical demographics."""
    prof = client_profile(df)
    labels = {
        "genero": cfg.GENDER_LABELS,
        "estado_civil": cfg.MARITAL_LABELS,
        "nivel_educacion": cfg.EDUCATION_LABELS,
        "ocupacion": cfg.OCCUPATION_LABELS,
    }
    rows = []
    for col, mapping in labels.items():
        counts = prof[col].value_counts().sort_index()
        for level, n in counts.items():
            rows.append(
                {
                    "variable": col,
                    "nivel": mapping.get(level, str(level)),
                    "clientes": int(n),
                    "pct_clientes": round(100 * n / len(prof), 1),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D3 -- predictive power of demographics
# --------------------------------------------------------------------------


def purchase_rate_by_demographic(df: pd.DataFrame) -> pd.DataFrame:
    """D3: purchase rate by demographic level, measured at visit level.

    The spread across levels is what the acceptance criterion looks at: if it
    is narrow, demographics cannot carry the prediction.
    """
    categorical = ["genero", "estado_civil", "nivel_educacion", "ocupacion"]
    labels = {
        "genero": cfg.GENDER_LABELS,
        "estado_civil": cfg.MARITAL_LABELS,
        "nivel_educacion": cfg.EDUCATION_LABELS,
        "ocupacion": cfg.OCCUPATION_LABELS,
    }
    rows = []
    for col in categorical:
        g = df.groupby(col)[cfg.TARGET_INCIDENCE].agg(["mean", "size"])
        for level, (rate, n) in g.iterrows():
            rows.append(
                {
                    "variable": col,
                    "nivel": labels[col].get(level, str(level)),
                    "visitas": int(n),
                    "tasa_compra": round(float(rate), 4),
                }
            )
    out = pd.DataFrame(rows)
    base = float(df[cfg.TARGET_INCIDENCE].mean())
    out["lift_vs_base"] = (out["tasa_compra"] / base).round(2)
    return out


def demographic_spread(df: pd.DataFrame) -> pd.DataFrame:
    """D3: for each demographic variable, the spread of purchase rate it explains.

    Compared against the client-level spread (0.046 to 0.922), a narrow
    demographic spread is the evidence for "segments but does not predict".
    """
    rates = purchase_rate_by_demographic(df)
    rows = []
    for col, grp in rates.groupby("variable"):
        rows.append(
            {
                "variable": col,
                "tasa_min": grp["tasa_compra"].min(),
                "tasa_max": grp["tasa_compra"].max(),
                "amplitud": round(float(grp["tasa_compra"].max() - grp["tasa_compra"].min()), 4),
                "ratio_max_min": round(float(grp["tasa_compra"].max() / grp["tasa_compra"].min()), 2),
            }
        )
    out = pd.DataFrame(rows).sort_values("amplitud", ascending=False).reset_index(drop=True)

    prof = client_profile(df)
    client_spread = float(prof["tasa_compra"].max() - prof["tasa_compra"].min())
    out.attrs["amplitud_entre_clientes"] = round(client_spread, 4)
    out.attrs["lectura"] = (
        f"La mayor amplitud demografica es {out['amplitud'].iloc[0]:.4f}, frente a "
        f"{client_spread:.4f} de amplitud entre clientes individuales. "
        "La demografia segmenta pero no predice: el poder predictivo esta en el "
        "comportamiento y el contexto comercial."
    )
    return out


def continuous_demographic_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """D3: purchase rate by bin of age and annual income."""
    rows = []
    for col in ("edad", "ingreso_anual"):
        d = df.copy()
        d["bin"] = pd.qcut(d[col], n_bins, duplicates="drop")
        g = d.groupby("bin", observed=True)[cfg.TARGET_INCIDENCE].agg(["mean", "size"])
        for level, (rate, n) in g.iterrows():
            rows.append(
                {
                    "variable": col,
                    "bin": str(level),
                    "visitas": int(n),
                    "tasa_compra": round(float(rate), 4),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Segmentation input matrix for model M0
# --------------------------------------------------------------------------


def segmentation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Client-level demographic matrix, ready for standardization and K-Means.

    One row per client, indexed by ``id``. This is the input of model M0; the
    behavioural variables are deliberately excluded so the segments stay
    interpretable as commercial groups rather than as outcome proxies.
    """
    prof = client_profile(df)
    return prof.set_index(cfg.KEY_CLIENT)[cfg.DEMOGRAPHIC_COLS]


def behaviour_by_client_for_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Client-level behaviour, held aside to profile segments after clustering.

    Kept separate from :func:`segmentation_matrix` on purpose: clustering on
    behaviour and then explaining behaviour by cluster would be circular.
    """
    prof = client_profile(df)
    return prof.set_index(cfg.KEY_CLIENT)[
        ["visitas", "compras", "tasa_compra", "gasto_total", "ticket_medio"]
    ]
