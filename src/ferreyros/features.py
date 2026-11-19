"""Derived-variable construction -- section 3 of ``docs/brief-problema.md``.

Five families: competitive price, promotion, client history, temporal and
demographic. Every history feature is built strictly backwards (control E6):
it uses ``shift`` or expanding windows over the past, never the current row.

Two different notions of "past" live here, and confusing them is the classic
leak:

* **Client history** is computed over the whole timeline, strictly backwards.
  A client's history in the test period legitimately includes their train
  visits -- that is past information, not leakage.
* **Reference statistics** (the price mean a brand is compared against, the
  price maximum that defines the implicit discount, the quantile edges of the
  demographic bins) are *fitted* quantities. Fitting them over train+test would
  let the test distribution shape a train feature, which is control E10. They
  live in :class:`FeatureReference`, fitted by :func:`fit_reference` on train
  only and then applied unchanged to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

# Columns produced by this module, grouped by family. Used by the audit to
# prove that no post-outcome column leaked into the feature matrix.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "precio": [
        "precio_min_dia",
        "precio_medio_dia",
        "precio_max_dia",
        "rango_precios_dia",
        *[f"precio_relativo_{j}" for j in cfg.BRANDS],
        *[f"desviacion_precio_{j}" for j in cfg.BRANDS],
        "gap_vs_ultima_marca",
    ],
    "promocion": [
        "n_promos_dia",
        "promo_en_ultima_marca",
        *[f"descuento_implicito_{j}" for j in cfg.BRANDS],
    ],
    "historial": [
        "n_visitas_previas",
        "n_compras_previas",
        "tasa_compra_historica",
        "dias_desde_ultima_compra",
        "racha_sin_comprar",
        "cantidad_media_historica",
        "ticket_medio_historico",
        "n_marcas_distintas_previas",
        *[f"lealtad_marca_{j}" for j in cfg.BRANDS],
    ],
    "temporal": ["dia_semana", "semana", "mes", "visitas_ultimos_7d", "visitas_ultimos_30d"],
    "demografica": ["edad_bin", "ingreso_bin"],
}


def all_feature_names() -> list[str]:
    """Flat list of every engineered feature name."""
    return [name for family in FEATURE_FAMILIES.values() for name in family]


# --------------------------------------------------------------------------
# Fitted reference statistics -- control E10
# --------------------------------------------------------------------------


@dataclass
class FeatureReference:
    """Statistics fitted on train and applied unchanged to test.

    Three feature groups need a reference value that is not in the row itself:
    the historical mean each brand's price is compared against, the historical
    maximum that defines the implicit discount, and the quantile edges of the
    demographic bins. Fitting any of them over train+test is preprocessing
    leakage (control E10), which is why they are a fitted object and not an
    expression inline in the feature function.
    """

    precio_medio: dict[str, float] = field(default_factory=dict)
    precio_max: dict[str, float] = field(default_factory=dict)
    edad_bins: list[float] = field(default_factory=list)
    ingreso_bins: list[float] = field(default_factory=list)
    n_filas_ajuste: int = 0
    dia_max_ajuste: int = 0
    n_bins: int = 5

    @property
    def esta_ajustada(self) -> bool:
        """True once :func:`fit_reference` has populated the statistics."""
        return self.n_filas_ajuste > 0

    def describe(self) -> pd.DataFrame:
        """One row per fitted statistic, for the notebook to print as evidence."""
        rows = [
            {"estadistico": f"precio_medio[{col}]", "valor": round(v, 4)}
            for col, v in self.precio_medio.items()
        ]
        rows += [
            {"estadistico": f"precio_max[{col}]", "valor": round(v, 4)}
            for col, v in self.precio_max.items()
        ]
        rows += [{"estadistico": "edad_bins", "valor": str([round(e, 1) for e in self.edad_bins])}]
        rows += [
            {"estadistico": "ingreso_bins", "valor": str([round(e, 0) for e in self.ingreso_bins])}
        ]
        rows += [
            {"estadistico": "filas usadas en el ajuste", "valor": self.n_filas_ajuste},
            {"estadistico": "ultimo dia usado en el ajuste", "valor": self.dia_max_ajuste},
        ]
        return pd.DataFrame(rows)


def fit_reference(train: pd.DataFrame, n_bins: int = 5) -> FeatureReference:
    """Fit the reference statistics on the train split and nothing else.

    Pass the train frame, never the full one. :func:`build_features` calls this
    on the train portion by default so that the safe behaviour is also the
    default behaviour.
    """
    return FeatureReference(
        precio_medio={col: float(train[col].mean()) for col in cfg.PRICE_COLS},
        precio_max={col: float(train[col].max()) for col in cfg.PRICE_COLS},
        edad_bins=_bin_edges(train["edad"], n_bins),
        ingreso_bins=_bin_edges(train["ingreso_anual"], n_bins),
        n_filas_ajuste=len(train),
        dia_max_ajuste=int(train[cfg.KEY_TIME].max()),
        n_bins=n_bins,
    )


def _bin_edges(serie: pd.Series, n_bins: int) -> list[float]:
    """Interior quantile edges of ``serie``, deduplicated."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return [float(e) for e in np.unique(np.quantile(serie.dropna().to_numpy(), quantiles))]


def _apply_bins(serie: pd.Series, edges: list[float]) -> pd.Series:
    """Assign bin indices using train-fitted edges.

    A test value outside the train range falls into the first or last bin
    rather than creating a new one, which keeps the encoding stable.
    """
    if not edges:
        return pd.Series(0.0, index=serie.index, dtype="float64")
    return pd.Series(
        np.digitize(serie.to_numpy(), edges).astype("float64"), index=serie.index
    )


def _resolve_reference(
    df: pd.DataFrame, reference: FeatureReference | None, n_bins: int = 5
) -> FeatureReference:
    """Return the given reference, or fit one on the train portion of ``df``.

    Defaulting to the train portion -- not to ``df`` itself -- is what makes a
    plain ``build_features(df)`` call leak-free.
    """
    if reference is not None:
        return reference
    train = df[df[cfg.KEY_TIME] <= cfg.SPLIT_DAY]
    return fit_reference(train if len(train) else df, n_bins=n_bins)


# --------------------------------------------------------------------------
# 3.1 Competitive price features
# --------------------------------------------------------------------------


def add_price_features(
    df: pd.DataFrame, reference: FeatureReference | None = None
) -> pd.DataFrame:
    """Competitive position of each brand within the day's price set.

    These are the features through which the price decision variable
    ``y_jtk`` reaches the incidence and brand-choice models.

    ``reference`` carries the train-fitted historical mean of each price. Left
    at ``None`` it is fitted on the train portion of ``df`` (control E10).
    """
    out = df.copy()
    reference = _resolve_reference(out, reference)
    prices = out[cfg.PRICE_COLS]

    out["precio_min_dia"] = prices.min(axis=1)
    out["precio_medio_dia"] = prices.mean(axis=1)
    out["precio_max_dia"] = prices.max(axis=1)
    out["rango_precios_dia"] = out["precio_max_dia"] - out["precio_min_dia"]

    for j in cfg.BRANDS:
        col = f"precio_marca_{j}"
        # Position of the brand against the day's average: < 1 means cheap today.
        out[f"precio_relativo_{j}"] = out[col] / out["precio_medio_dia"]
        # Position against the brand's own history: < 0 means cheap for itself.
        # The mean comes from train only; computing it here over train+test
        # would let the test price distribution shape a train feature.
        hist_mean = reference.precio_medio[col]
        out[f"desviacion_precio_{j}"] = (out[col] - hist_mean) / hist_mean

    # Incentive to switch away from the habitual brand.
    last_brand_price = _price_of_brand(out, out["ultima_marca_comprada"])
    out["gap_vs_ultima_marca"] = (last_brand_price - out["precio_min_dia"]).fillna(0.0)
    return out


def _price_of_brand(df: pd.DataFrame, brand_series: pd.Series) -> pd.Series:
    """Price of the brand named per row; ``NaN`` where the brand is 0 (no history)."""
    matrix = df[cfg.PRICE_COLS].to_numpy()
    idx = brand_series.to_numpy() - 1
    valid = idx >= 0
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[valid] = matrix[valid, idx[valid]]
    return out


# --------------------------------------------------------------------------
# 3.2 Promotion features
# --------------------------------------------------------------------------


def add_promo_features(
    df: pd.DataFrame, reference: FeatureReference | None = None
) -> pd.DataFrame:
    """Promotional context of the visit -- the channel for decision ``z_jt``.

    ``reference`` carries the train-fitted list price (the historical maximum)
    each implicit discount is measured against.
    """
    out = df.copy()
    reference = _resolve_reference(out, reference)
    out["n_promos_dia"] = out[cfg.PROMO_COLS].sum(axis=1)

    promo_matrix = out[cfg.PROMO_COLS].to_numpy()
    idx = out["ultima_marca_comprada"].to_numpy() - 1
    valid = idx >= 0
    flag = np.zeros(len(out), dtype=float)
    flag[valid] = promo_matrix[valid, idx[valid]]
    out["promo_en_ultima_marca"] = flag

    for j in cfg.BRANDS:
        col = f"precio_marca_{j}"
        # List price = highest price seen in train. Taking the maximum over the
        # full frame would import a future price into a past feature.
        out[f"descuento_implicito_{j}"] = 1 - out[col] / reference.precio_max[col]
    return out


# --------------------------------------------------------------------------
# 3.3 Client-history features -- strictly backward looking (control E6)
# --------------------------------------------------------------------------


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural history of the client, computed only from earlier visits.

    Every expression here is either a ``shift`` or an expanding window closed
    before the current row. Nothing from the current visit enters, so these
    features are safe for the incidence model.
    """
    out = df.sort_values([cfg.KEY_CLIENT, cfg.KEY_TIME]).copy()
    g = out.groupby(cfg.KEY_CLIENT)

    out["n_visitas_previas"] = g.cumcount()
    out["n_compras_previas"] = (
        g[cfg.TARGET_INCIDENCE].cumsum() - out[cfg.TARGET_INCIDENCE]
    )
    out["tasa_compra_historica"] = (
        out["n_compras_previas"] / out["n_visitas_previas"].replace(0, np.nan)
    )

    # Days since the client's last purchase.
    day_if_bought = out[cfg.KEY_TIME].where(out[cfg.TARGET_INCIDENCE] == 1)
    last_purchase_day = day_if_bought.groupby(out[cfg.KEY_CLIENT]).shift(1).groupby(
        out[cfg.KEY_CLIENT]
    ).ffill()
    out["dias_desde_ultima_compra"] = out[cfg.KEY_TIME] - last_purchase_day

    # Consecutive previous visits without a purchase.
    out["racha_sin_comprar"] = _no_purchase_streak(out)

    # Previous quantity and spend, expanding over past purchases only.
    qty_prev = out[cfg.TARGET_QUANTITY].where(out[cfg.TARGET_INCIDENCE] == 1)
    out["cantidad_media_historica"] = (
        qty_prev.groupby(out[cfg.KEY_CLIENT]).apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    if "gasto" in out.columns:
        spend_prev = out["gasto"].where(out[cfg.TARGET_INCIDENCE] == 1)
        out["ticket_medio_historico"] = (
            spend_prev.groupby(out[cfg.KEY_CLIENT]).apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        )
    else:
        out["ticket_medio_historico"] = np.nan

    # Brand loyalty: historical share of each brand in the client's purchases.
    for j in cfg.BRANDS:
        bought_j = (out[cfg.TARGET_BRAND] == j).astype(float)
        prev_j = bought_j.groupby(out[cfg.KEY_CLIENT]).cumsum() - bought_j
        out[f"lealtad_marca_{j}"] = prev_j / out["n_compras_previas"].replace(0, np.nan)

    distinct = out.groupby(cfg.KEY_CLIENT)[cfg.TARGET_BRAND].apply(
        lambda s: s.shift(1).expanding().apply(lambda w: len(set(w) - {0}))
    ).reset_index(level=0, drop=True)
    out["n_marcas_distintas_previas"] = distinct

    return out


def _no_purchase_streak(df: pd.DataFrame) -> pd.Series:
    """Consecutive prior visits without a purchase, per client."""
    prev = df.groupby(cfg.KEY_CLIENT)[cfg.TARGET_INCIDENCE].shift(1).fillna(0)
    streak = np.zeros(len(df), dtype=float)
    counter = 0.0
    last_client = None
    clients = df[cfg.KEY_CLIENT].to_numpy()
    prev_values = prev.to_numpy()
    for i in range(len(df)):
        if clients[i] != last_client:
            counter = 0.0
            last_client = clients[i]
        streak[i] = counter
        counter = 0.0 if prev_values[i] == 1 else counter + 1
    return pd.Series(streak, index=df.index)


# --------------------------------------------------------------------------
# 3.4 Temporal features
# --------------------------------------------------------------------------


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar structure derived from ``dia_visita`` plus visit-density windows."""
    out = df.copy()
    out["dia_semana"] = out[cfg.KEY_TIME] % 7
    out["semana"] = (out[cfg.KEY_TIME] - 1) // 7 + 1
    out["mes"] = (out[cfg.KEY_TIME] - 1) // 30 + 1

    for window in (7, 30):
        counts = []
        for _, grp in out.groupby(cfg.KEY_CLIENT, sort=False):
            days = grp[cfg.KEY_TIME].to_numpy()
            # Prior visits strictly inside the window: searchsorted on sorted days.
            left = np.searchsorted(days, days - window, side="left")
            idx = np.arange(len(days))
            counts.append(pd.Series(idx - left, index=grp.index))
        out[f"visitas_ultimos_{window}d"] = pd.concat(counts).reindex(out.index)
    return out


# --------------------------------------------------------------------------
# 3.5 Demographic transformations
# --------------------------------------------------------------------------


def add_demographic_features(
    df: pd.DataFrame, reference: FeatureReference | None = None, n_bins: int = 5
) -> pd.DataFrame:
    """Binned age and income; education stays ordinal, occupation stays nominal.

    The bin edges are train-fitted quantiles, not ``pd.qcut`` over the frame
    being transformed: the cut points are a learned parameter (control E10).
    """
    out = df.copy()
    reference = _resolve_reference(out, reference, n_bins=n_bins)
    out["edad_bin"] = _apply_bins(out["edad"], reference.edad_bins)
    out["ingreso_bin"] = _apply_bins(out["ingreso_anual"], reference.ingreso_bins)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame, reference: FeatureReference | None = None
) -> pd.DataFrame:
    """Apply every family in order and return the enriched visit-level frame.

    The order matters: price features need the lag brand, history features need
    ``gasto``, so :func:`ferreyros.data.add_paid_price_and_spend` should have
    run first.

    Call it on the **full** frame. History features must see the whole timeline
    to accumulate a client's past across the split boundary, which is legitimate
    (control E6); the fitted statistics are what must stay inside train, and
    ``reference`` is where they live. Left at ``None``, it is fitted on the
    train portion of ``df``, so the default call is already leak-free. The
    returned frame carries the reference in ``attrs['reference']``.
    """
    out = df.copy()
    reference = _resolve_reference(out, reference)
    out = add_price_features(out, reference)
    out = add_promo_features(out, reference)
    out = add_history_features(out)
    out = add_temporal_features(out)
    out = add_demographic_features(out, reference)
    out.attrs["reference"] = reference
    return out


def feature_matrix(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Assemble the admissible feature matrix for model M1.

    Excludes, by construction: the three targets, the post-outcome columns
    (controls E2, E3), ``tamanio_ciudad`` (control E1), the keys, and the
    intermediate ``precio_pagado`` / ``gasto`` columns, which are outcomes too.
    """
    engineered = [c for c in all_feature_names() if c in df.columns]
    raw_admissible = [
        *cfg.PRICE_COLS,
        *cfg.PROMO_COLS,
        *cfg.DEMOGRAPHIC_COLS,
        *cfg.LAG_COLS,
    ]
    cols = engineered + raw_admissible + (extra or [])

    forbidden = {
        cfg.TARGET_INCIDENCE,
        *cfg.POST_OUTCOME_COLS,
        *cfg.EXCLUDED_COLS,
        cfg.KEY_CLIENT,
        "precio_pagado",
        "gasto",
    }
    cols = [c for c in dict.fromkeys(cols) if c not in forbidden]
    return df[cols]


def verify_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Control E6: prove the history features carry no information from the current row.

    Three concrete assertions, not a declaration of intent:

    1. On a client's **first** visit no history can exist, so every history
       feature must be zero or missing there.
    2. ``n_compras_previas`` must equal the cumulative purchase count *excluding*
       the current visit.
    3. ``lealtad_marca_j`` must not react to the brand bought in the current
       visit: among first-time buyers, prior loyalty must be missing.
    """
    d = df.sort_values([cfg.KEY_CLIENT, cfg.KEY_TIME])
    first_visit = d.groupby(cfg.KEY_CLIENT).cumcount() == 0
    rows = []

    history = [c for c in FEATURE_FAMILIES["historial"] if c in d.columns]
    offenders = []
    for col in history:
        vals = d.loc[first_visit, col]
        if not bool(((vals == 0) | vals.isna()).all()):
            offenders.append(col)
    rows.append(
        {
            "verificacion": "Primera visita sin historial (features = 0 o NaN)",
            "features_evaluadas": len(history),
            "violaciones": len(offenders),
            "detalle": ", ".join(offenders) if offenders else "ninguna",
        }
    )

    expected = (
        d.groupby(cfg.KEY_CLIENT)[cfg.TARGET_INCIDENCE].cumsum() - d[cfg.TARGET_INCIDENCE]
    )
    mismatches = int((d["n_compras_previas"] != expected).sum()) if "n_compras_previas" in d else -1
    rows.append(
        {
            "verificacion": "n_compras_previas excluye la compra de la fila actual",
            "features_evaluadas": 1,
            "violaciones": mismatches,
            "detalle": "cumsum() - fila actual" if mismatches == 0 else "revisar construccion",
        }
    )

    loyalty = [f"lealtad_marca_{j}" for j in cfg.BRANDS if f"lealtad_marca_{j}" in d.columns]
    first_buy = first_visit & (d[cfg.TARGET_INCIDENCE] == 1)
    leak = 0
    if first_buy.any() and loyalty:
        leak = int((d.loc[first_buy, loyalty].fillna(0) > 0).sum().sum())
    rows.append(
        {
            "verificacion": "lealtad_marca no reacciona a la compra actual",
            "features_evaluadas": len(loyalty),
            "violaciones": leak,
            "detalle": f"{int(first_buy.sum())} primeras visitas con compra evaluadas",
        }
    )

    out = pd.DataFrame(rows)
    out["estado"] = out["violaciones"].map(lambda n: "OK" if n == 0 else "FALLA")
    out.attrs["sin_fuga"] = bool((out["violaciones"] == 0).all())
    return out


def verify_reference_fit(
    df: pd.DataFrame, reference: FeatureReference | None = None
) -> pd.DataFrame:
    """Control E10, feature side: prove the fitted statistics come only from train.

    Two assertions, and the second is the one that matters:

    1. Every statistic in ``reference`` equals the statistic recomputed on the
       train split alone.
    2. At least one of them **differs** from the same statistic computed over
       train+test. Without this check the first one would also pass on a
       reference that happens to be fitted on everything, and the control would
       be cosmetic.
    """
    reference = reference or df.attrs.get("reference") or _resolve_reference(df, None)
    train = df[df[cfg.KEY_TIME] <= cfg.SPLIT_DAY]
    on_train = fit_reference(train, n_bins=reference.n_bins)
    on_full = fit_reference(df, n_bins=reference.n_bins)

    rows = []
    differs_from_full = 0
    for group, getter in (
        ("precio_medio", lambda r: r.precio_medio),
        ("precio_max", lambda r: r.precio_max),
    ):
        for col in cfg.PRICE_COLS:
            got, exp, full = getter(reference)[col], getter(on_train)[col], getter(on_full)[col]
            differs_from_full += int(abs(exp - full) > 1e-12)
            rows.append(
                {
                    "estadistico": f"{group}[{col}]",
                    "ajustado": round(got, 6),
                    "solo_train": round(exp, 6),
                    "train+test": round(full, 6),
                    "coincide_con_train": bool(abs(got - exp) < 1e-9),
                }
            )
    for group, getter in (
        ("edad_bins", lambda r: r.edad_bins),
        ("ingreso_bins", lambda r: r.ingreso_bins),
    ):
        got, exp, full = getter(reference), getter(on_train), getter(on_full)
        differs_from_full += int(exp != full)
        rows.append(
            {
                "estadistico": group,
                "ajustado": str([round(e, 1) for e in got]),
                "solo_train": str([round(e, 1) for e in exp]),
                "train+test": str([round(e, 1) for e in full]),
                "coincide_con_train": bool(got == exp),
            }
        )

    out = pd.DataFrame(rows)
    out["estado"] = out["coincide_con_train"].map({True: "OK", False: "FALLA"})
    all_match = bool(out["coincide_con_train"].all())
    out.attrs["ajustado_solo_en_train"] = all_match
    out.attrs["n_estadisticos_que_difieren_del_frame_completo"] = differs_from_full
    out.attrs["distincion_material"] = differs_from_full > 0
    out.attrs["e10_features_ok"] = all_match and differs_from_full > 0
    out.attrs["lectura"] = (
        f"{len(out)} estadisticos ajustados solo en train (dias 1-{reference.dia_max_ajuste}, "
        f"{reference.n_filas_ajuste:,} filas); {differs_from_full} difieren del valor que "
        "tendrian ajustados sobre train+test, asi que la distincion no es cosmetica."
    )
    return out


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per engineered feature: family, coverage and range.

    ``pct_nulos`` is expected to be non-zero for history features on a client's
    first visits -- that is absence of history, not a defect.
    """
    rows = []
    for family, names in FEATURE_FAMILIES.items():
        for name in names:
            if name not in df.columns:
                continue
            serie = df[name]
            rows.append(
                {
                    "familia": family,
                    "feature": name,
                    "pct_nulos": round(100 * float(serie.isna().mean()), 1),
                    "media": round(float(serie.mean()), 3),
                    "min": round(float(serie.min()), 3),
                    "max": round(float(serie.max()), 3),
                }
            )
    return pd.DataFrame(rows)
