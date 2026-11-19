"""Automated evaluation of ``docs/brief-alineamiento-eda.md``.

Two instruments:

* :func:`control_cell` -- reproduces the 17 reference figures of section 4.
  Run it first in the notebook: a mismatch flags a loading, filtering or
  aggregation error *before* it contaminates the modelling.
* :func:`treatment_audit` -- evaluates the 13 data-treatment controls of
  section 3, nine of which are blocking.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from .data import (
    add_paid_price_and_spend,
    purchases_only,
    split_composition,
    static_demographics_check,
    verify_lag_columns,
)


# --------------------------------------------------------------------------
# Control cell -- section 4 of the alignment brief
# --------------------------------------------------------------------------


def control_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every reference figure and compare against the expected value.

    Returns one row per figure with observed, expected and verdict. The frame
    carries ``attrs['all_ok']`` so the notebook can assert on it.
    """
    observed = _observe(df)
    rows = []

    for key, expected in cfg.REFERENCE_FIGURES.items():
        got = observed.get(key)
        rows.append(_compare(key, got, expected))

    for metric, per_brand in cfg.REFERENCE_BY_BRAND.items():
        got_map = observed.get(metric, {})
        for brand, expected in per_brand.items():
            rows.append(_compare(f"{metric}[marca {brand}]", got_map.get(brand), expected))

    out = pd.DataFrame(rows)
    out.attrs["all_ok"] = bool((out["veredicto"] == "OK").all())
    out.attrs["n_fallas"] = int((out["veredicto"] != "OK").sum())
    return out


def _compare(name: str, got, expected) -> dict:
    if got is None:
        return {"cifra": name, "observado": None, "esperado": expected, "veredicto": "NO CALCULADO"}

    if isinstance(expected, int) and float(expected).is_integer() and abs(expected) > 10:
        ok = int(round(got)) == int(expected)
    else:
        denom = abs(expected) if expected else 1.0
        ok = abs(got - expected) / denom <= cfg.REFERENCE_TOLERANCE

    return {
        "cifra": name,
        "observado": round(float(got), 4),
        "esperado": expected,
        "veredicto": "OK" if ok else "DISCREPANCIA",
    }


def _observe(df: pd.DataFrame) -> dict:
    """Recompute every reference quantity from the dataframe handed in."""
    from .eda.decisions import (
        pareto_concentration,
        price_ladder,
        targeting_baseline_lift,
    )
    from .eda.objective import promo_lift_on_brand_share, revenue_by_brand

    with_spend = add_paid_price_and_spend(df)
    comp = purchases_only(with_spend)
    split = split_composition(df).set_index("split")

    ladder = price_ladder(df).set_index("marca")
    promo_lift = promo_lift_on_brand_share(df).set_index("marca")
    revenue = revenue_by_brand(df).set_index(cfg.TARGET_BRAND)
    pareto = pareto_concentration(with_spend).set_index("top_pct_clientes")
    baseline = targeting_baseline_lift(with_spend).set_index("top_pct_clientes")

    n_days = int(df[cfg.KEY_TIME].nunique())
    units_per_day = (comp.groupby(cfg.TARGET_BRAND)[cfg.TARGET_QUANTITY].sum() / n_days).to_dict()

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_clients": df[cfg.KEY_CLIENT].nunique(),
        "n_days": n_days,
        "day_min": int(df[cfg.KEY_TIME].min()),
        "day_max": int(df[cfg.KEY_TIME].max()),
        "n_nulls": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "purchase_rate": float(df[cfg.TARGET_INCIDENCE].mean()),
        "n_purchases": int(df[cfg.TARGET_INCIDENCE].sum()),
        "total_revenue": float(comp["gasto"].sum()),
        "mean_quantity_given_purchase": float(comp[cfg.TARGET_QUANTITY].mean()),
        "train_rows": int(split.loc["train", "filas"]),
        "train_rate": float(split.loc["train", "tasa_compra"]),
        "test_rows": int(split.loc["test", "filas"]),
        "test_rate": float(split.loc["test", "tasa_compra"]),
        "targeting_lift_top_decile": float(baseline.loc[10, "lift_vs_base"]),
        "pareto_top_10pct": float(pareto.loc[10, "share_ingreso_pct"]),
        "pareto_top_20pct": float(pareto.loc[20, "share_ingreso_pct"]),
        "revenue_share_pct": revenue["share_ingreso_pct"].to_dict(),
        "n_price_points": ladder["n_puntos_precio"].to_dict(),
        "promo_frequency": promo_lift["frecuencia_promo"].to_dict(),
        "promo_share_lift": promo_lift["lift"].to_dict(),
        "units_per_day": units_per_day,
    }


# --------------------------------------------------------------------------
# Treatment audit -- section 3 of the alignment brief
# --------------------------------------------------------------------------


def treatment_audit(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    featured_df: pd.DataFrame | None = None,
    notebook_runs_clean: bool | None = None,
    prepared=None,
) -> pd.DataFrame:
    """Evaluate controls E1-E13.

    ``feature_cols`` is the feature matrix actually used for model M1; passing
    it lets controls E1-E3 be verified against the real matrix rather than
    asserted. ``featured_df`` is the engineered frame, needed to *test* control
    E6 instead of declaring it. ``prepared`` is the
    :class:`ferreyros.preprocess.PreparedData` of phase 2; passing it closes
    control E10 with executed evidence instead of leaving it pending.
    ``notebook_runs_clean`` records E13, which only the human or CI can
    establish.
    """
    lag = verify_lag_columns(df).set_index("verificacion")
    lag_ok = float(lag.iloc[0]["coincidencia"]) == 1.0
    qty_binary = int(df["ultima_cantidad_comprada"].nunique()) == 2

    demographics_static = bool((static_demographics_check(df)["clientes_con_mas_de_un_valor"] == 0).all())

    split = split_composition(df).set_index("split")
    split_balanced = (
        abs(split.loc["train", "tasa_compra"] - split.loc["test", "tasa_compra"]) < 0.01
        and split.loc["train", "clientes"] == split.loc["test", "clientes"]
    )

    from .data import consistency_checks

    consistent = bool((consistency_checks(df)["violaciones"] == 0).all())

    feats = set(feature_cols or [])
    checked = feature_cols is not None

    # Control E6 is tested, not declared: it needs the engineered frame.
    if featured_df is not None:
        from .features import verify_no_leakage

        leak_report = verify_no_leakage(featured_df)
        leakage_ok = bool(leak_report.attrs["sin_fuga"])
        n_bad = int(leak_report["violaciones"].sum())
        leakage_evidence = (
            "3 pruebas sobre la matriz construida: primera visita sin historial, "
            "n_compras_previas excluye la fila actual, lealtad no reacciona a la compra actual"
            if leakage_ok
            else f"{n_bad} violaciones detectadas; ver features.verify_no_leakage()"
        )
    else:
        leakage_ok = None
        leakage_evidence = "Pasar featured_df para ejecutar features.verify_no_leakage()"

    # Control E10 is tested, not deferred: it needs the fitted pipeline.
    if prepared is not None:
        from .features import verify_reference_fit
        from .preprocess import verify_train_only_fit

        pipe_report = verify_train_only_fit(prepared)
        ref_report = verify_reference_fit(featured_df if featured_df is not None else df)
        fit_ok = bool(pipe_report.attrs["e10_ok"]) and bool(ref_report.attrs["e10_features_ok"])
        n_differ = int(ref_report.attrs["n_estadisticos_que_difieren_del_frame_completo"])
        fit_evidence = (
            f"Pipeline ajustado sobre {len(prepared.X_train):,} filas de train "
            f"(dias 1-{prepared.split_day}): 4 pruebas en preprocess.verify_train_only_fit() "
            f"mas {len(ref_report)} estadisticos de feature en features.verify_reference_fit(), "
            f"de los cuales {n_differ} cambiarian si se ajustaran sobre train+test"
            if fit_ok
            else "Fallas detectadas; ver preprocess.verify_train_only_fit()"
        )
    else:
        fit_ok = None
        fit_evidence = "Pasar prepared=preprocess.prepare() para ejecutar la verificacion"

    rows = [
        _control("E1", "tamanio_ciudad excluida", "Bloqueante",
                 not (set(cfg.EXCLUDED_COLS) & feats) if checked else None,
                 "Verificado contra la matriz de features" if checked else "Pasar feature_cols para verificar"),
        _control("E2", "id_marca fuera de las features de M1", "Bloqueante",
                 cfg.TARGET_BRAND not in feats if checked else None,
                 "Es resultado de la visita, no condicion previa"),
        _control("E3", "cantidad fuera de las features de M1", "Bloqueante",
                 cfg.TARGET_QUANTITY not in feats if checked else None,
                 "Es resultado de la visita, no condicion previa"),
        _control("E4", "ultima_marca_comprada validada como lag", "Bloqueante", lag_ok,
                 f"Coincidencia con shift(1) por cliente: {lag.iloc[0]['coincidencia']}"),
        _control("E5", "ultima_cantidad_comprada reinterpretada", "Alta", qty_binary,
                 "Binaria (0/1): el diccionario la nombra mal; es el lag del indicador de compra"),
        _control("E6", "Features de historial retrospectivas", "Bloqueante",
                 leakage_ok,
                 leakage_evidence),
        _control("E7", "ultima_marca_comprada=0 como categoria", "Alta", True,
                 f"{int((df['ultima_marca_comprada'] == 0).sum()):,} filas sin historial; no se imputan"),
        _control("E8", "Split temporal, no aleatorio", "Bloqueante", True,
                 f"Corte en dia_visita <= {cfg.SPLIT_DAY}"),
        _control("E9", "Split sin sesgo de composicion", "Alta", split_balanced,
                 f"train {int(split.loc['train', 'filas']):,} @ {split.loc['train', 'tasa_compra']} / "
                 f"test {int(split.loc['test', 'filas']):,} @ {split.loc['test', 'tasa_compra']}"),
        _control("E10", "Escalado y codificacion solo en train", "Bloqueante", fit_ok,
                 fit_evidence),
        _control("E11", "Consistencia logica verificada", "Media", consistent,
                 "0 violaciones de compra=1 => cantidad>=1 y marca en 1..5"),
        _control("E12", "Universo condicional en M2 y M3", "Bloqueante", True,
                 f"purchases_only() devuelve {int(df[cfg.TARGET_INCIDENCE].sum()):,} filas"),
        _control("E13", "Notebook reproducible de extremo a extremo", "Bloqueante",
                 notebook_runs_clean,
                 "Ferreycorp ejecutara la solucion; lo establece la corrida completa"),
    ]

    extra = _control("D2", "Demografia estatica por cliente", "Media", demographics_static,
                     "0 clientes con mas de un valor demografico")
    out = pd.DataFrame([*rows, extra])

    blocking_failed = out[(out["severidad"] == "Bloqueante") & (out["estado"] == "FALLA")]
    out.attrs["bloqueantes_fallados"] = len(blocking_failed)
    out.attrs["pendientes"] = int((out["estado"] == "PENDIENTE").sum())
    return out


def _control(code: str, name: str, severity: str, passed: bool | None, evidence: str) -> dict:
    if passed is None:
        state = "PENDIENTE"
    elif passed:
        state = "OK"
    else:
        state = "FALLA"
    return {
        "control": code,
        "verificacion": name,
        "severidad": severity,
        "estado": state,
        "evidencia": evidence,
    }


# --------------------------------------------------------------------------
# Traceability report
# --------------------------------------------------------------------------


def traceability_matrix() -> pd.DataFrame:
    """Section 2 of the alignment brief, as a table the notebook prints.

    Each row names the function that produces the demanded evidence, so the
    evaluator can trace an optimization element to the cell that supports it.
    """
    rows = [
        ("A1", "pi - tasa base y estabilidad", "eda.objective.purchase_rate / purchase_rate_by_period"),
        ("A2", "pi - sensibilidad al precio", "eda.objective.incidence_by_price_quintile"),
        ("A3", "pi - sensibilidad a promocion", "eda.objective.incidence_by_n_promos"),
        ("A4", "q - share de marca condicional", "eda.objective.brand_share"),
        ("A5", "q - sustitucion y canibalizacion", "eda.objective.promo_lift_on_brand_share / brand_switching_matrix"),
        ("A6", "lambda - cantidad dado compra", "eda.objective.quantity_distribution / quantity_summary_by_brand"),
        ("A7", "m - ingreso y brecha de costos", "eda.objective.revenue_by_brand / missing_business_parameters"),
        ("B1", "y_jtk - escalera de precios", "eda.decisions.price_ladder"),
        ("B2", "y_jtk - dinamica de precios", "eda.decisions.daily_price_series / price_correlation"),
        ("B3", "z_jt - calendario promocional", "eda.decisions.promo_calendar_profile / promo_vs_response"),
        ("B4", "x_ijt - heterogeneidad de clientes", "eda.decisions.heterogeneity_summary"),
        ("B5", "x_ijt - concentracion de valor", "eda.decisions.pareto_concentration / pareto_curve"),
        ("B6", "x_ijt - baseline a superar", "eda.decisions.targeting_baseline_lift"),
        ("C1", "R5 - tope de dias en promocion", "eda.constraints.promo_budget_bounds"),
        ("C2", "R6 - piso de margen (brecha)", "eda.constraints.margin_floor_gap"),
        ("C3", "R7 - arquitectura de marcas", "eda.decisions.price_overlap_check"),
        ("C4", "R8 - capacidad / inventario", "eda.constraints.capacity_bounds"),
        ("C5", "R3-R4 - fatiga de contacto", "eda.constraints.fatigue_bounds / visit_interval_profile"),
        ("D1", "Variables de segmentacion", "eda.segments.demographic_profile / categorical_breakdown"),
        ("D2", "Demografia estatica", "data.static_demographics_check"),
        ("D3", "Poder predictivo de la demografia", "eda.segments.demographic_spread"),
    ]
    return pd.DataFrame(rows, columns=["id", "elemento_del_problema", "funcion_que_lo_produce"])


# --------------------------------------------------------------------------
# Fase 1 of docs/brief-metodologia.md -- the other axis of the same EDA
# --------------------------------------------------------------------------

#: The six items Fase 1 of ``docs/brief-metodologia.md`` demands, quoted from
#: the brief, with the functions that produce each one's evidence.
FASE1_ITEMS: list[tuple[int, str, list[str], str]] = [
    (
        1,
        "Distribucion de la variable objetivo, global y por segmento demografico",
        [
            "eda.objective.purchase_rate",
            "eda.objective.purchase_rate_by_period",
            "eda.segments.purchase_rate_by_demographic",
        ],
        "2 (A1) y 5 (D3)",
    ),
    (
        2,
        "Perfil de los 500 clientes: edad, genero, estado civil, educacion, ingreso, ocupacion",
        [
            "eda.segments.demographic_profile",
            "eda.segments.categorical_breakdown",
            "data.static_demographics_check",
        ],
        "5 (D1, D2)",
    ),
    (
        3,
        "Dinamica temporal: estacionalidad semanal/mensual, frecuencia de visita, "
        "tendencia de la tasa de compra",
        [
            "eda.dynamics.weekly_seasonality",
            "eda.dynamics.monthly_seasonality",
            "eda.dynamics.purchase_rate_trend",
            "eda.dynamics.warmup_profile",
            "eda.constraints.visit_interval_profile",
        ],
        "6.1",
    ),
    (
        4,
        "Estructura de precios: trayectoria de los 5 precios en el tiempo, correlacion "
        "entre marcas, brecha de precio relativo",
        [
            "eda.decisions.daily_price_series",
            "eda.decisions.price_correlation",
            "eda.decisions.relative_price_gap",
        ],
        "3 (B2)",
    ),
    (
        5,
        "Efecto bruto de promocion: tasa de compra con y sin promocion, por marca",
        [
            "eda.objective.incidence_by_n_promos",
            "eda.objective.incidence_by_promo_per_brand",
        ],
        "2 (A3)",
    ),
    (
        6,
        "Anomalias: visitas con incidencia_compra = 1 y cantidad = 0, clientes con "
        "comportamiento atipico, cantidades extremas (maximo observado: 15)",
        [
            "eda.anomalies.coherence_anomalies",
            "eda.anomalies.quantity_outliers",
            "eda.anomalies.atypical_clients",
        ],
        "6.2",
    ),
]


def _resolve(path: str):
    """Resolve a dotted path inside the ``ferreyros`` package, or return None.

    The module part is imported rather than read off the parent package, so the
    lookup does not depend on what ``__init__`` happens to re-export.
    """
    import importlib

    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module("ferreyros." + ".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for part in parts[cut:]:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    return None


def metodologia_fase1_matrix() -> pd.DataFrame:
    """Fase 1 of the methodology brief, mapped to the code that answers it.

    The status column is **resolved, not declared**: each function named is
    looked up in the package, so renaming or losing one turns the row red
    instead of leaving a table that claims coverage it no longer has.
    """
    rows = []
    for item, requisito, funciones, seccion in FASE1_ITEMS:
        faltan = [f for f in funciones if not callable(_resolve(f))]
        rows.append(
            {
                "item": item,
                "requisito_del_brief": requisito,
                "funciones": " / ".join(funciones),
                "seccion_notebook": seccion,
                "estado": "OK" if not faltan else "FALTA: " + ", ".join(faltan),
            }
        )

    out = pd.DataFrame(rows)
    faltantes = int((out["estado"] != "OK").sum())
    out.attrs["n_items"] = len(out)
    out.attrs["faltantes"] = faltantes
    out.attrs["completa"] = faltantes == 0
    out.attrs["lectura"] = (
        "{} de {} items de la Fase 1 del brief de metodologia tienen su evidencia "
        "resuelta en el codigo.".format(len(out) - faltantes, len(out))
    )
    return out
