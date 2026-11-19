"""Export everything the dashboard and the deliverables need, in one pass.

``dashboard/index.html`` is a static page: it reads ``dashboard/data.json`` and
nothing else, so it opens offline with a double click. That means every
scenario the page lets you explore has to be **precomputed here** -- the budget
frontier, the cost-ratio grid, the three margin assumptions, the price plans
with and without R7, the propensity curve of each brand.

The page says so in a visible notice. Interactivity over precomputed scenarios
is honest; a static page that implied live scoring would not be.

    uv run python -c "import sys; sys.path.insert(0,'src'); from ferreyros import dashboard; dashboard.export_all()"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as data_mod
from . import framing
from . import models as mdl
from . import prescriptive as px
from . import segmentation as seg_mod
from .eda import anomalies, decisions, dynamics, objective, segments

DASHBOARD_DIR = cfg.PROJECT_ROOT / "dashboard"
FIGURES_DIR = cfg.PROJECT_ROOT / "docs" / "figuras"
ARTIFACTS_DIR = cfg.PROJECT_ROOT / "artifacts"


def _records(df: pd.DataFrame) -> list[dict]:
    """A DataFrame as JSON-safe records, index included when it carries meaning."""
    out = df.reset_index() if df.index.name else df
    return json.loads(out.to_json(orient="records", double_precision=6))


def _frozen(nombre: str) -> dict:
    ruta = ARTIFACTS_DIR / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


# --------------------------------------------------------------------------
# One computation pass, shared by the dashboard and the figures
# --------------------------------------------------------------------------


def compute(verbose: bool = True) -> dict:
    """Run every phase once and keep the tables both outputs need."""
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    R: dict = {}
    log("· cargando datos y verificando contratos")
    df = data_mod.add_paid_price_and_spend(data_mod.load_raw())
    modelo_m0 = seg_mod.fit(df, k=3)
    assert framing.verify_frozen(df).attrs["coincide"], "encuadre congelado no coincide"
    assert seg_mod.verify_frozen(modelo_m0).attrs["coincide"], "segmentos congelados no coinciden"
    R["df"], R["m0"] = df, modelo_m0

    log("· EDA")
    R["tasa"] = objective.purchase_rate(df)
    R["brand_share"] = objective.brand_share(df)
    R["promo_lift"] = objective.promo_lift_on_brand_share(df)
    R["promo_incidencia"] = objective.incidence_by_promo_per_brand(df)
    R["ladder"] = decisions.price_ladder(df)
    R["gap"] = decisions.relative_price_gap(df)
    R["baseline"] = decisions.targeting_baseline_lift(data_mod.add_paid_price_and_spend(df))
    R["temporal"] = dynamics.temporal_dynamics_summary(df)
    R["anomalias"] = anomalies.anomaly_summary(df)
    R["demografia"] = segments.purchase_rate_by_demographic(df)

    log("· segmentos (M0)")
    R["perfil_seg"] = seg_mod.segment_profile(modelo_m0, df)
    R["comportamiento_seg"] = seg_mod.segment_behaviour(modelo_m0, df)
    R["marcas_seg"] = seg_mod.segment_brand_share(modelo_m0, df)
    R["promo_seg"] = seg_mod.segment_promo_response(modelo_m0, df)
    R["pca"] = seg_mod.pca_projection(modelo_m0, df)
    R["seleccion_k"] = seg_mod.select_k(df)

    log("· modelos (M1/M2/M3)")
    prep = seg_mod.prepare_with_segment(modelo_m0, df)
    X_train, X_test = mdl.drop_features(prep.X_train, prep.X_test, mdl.TREND_COLS)
    pipe_m1, proba = mdl.fit_final(
        mdl.m1_candidates()["1 · logistica"], X_train, prep.y_train, X_test, {"C": 0.03}
    )
    R["m1_test"] = mdl.evaluate_binary(prep.y_test, proba, "logistica C=0.03")
    R["lift"] = mdl.lift_table(prep.y_test, proba)
    R["contra_baseline"] = mdl.baseline_policy_lift(X_test, prep.y_test, proba)
    R["umbral"] = mdl.threshold_by_cost(prep.y_test, proba)
    R["shap"] = mdl.shap_global(pipe_m1, X_test, top=12)
    R["curva_pr"] = _pr_curve(prep.y_test, proba)
    R["calibracion"] = _calibration(prep.y_test, proba)
    R["y_test"], R["proba_m1"] = prep.y_test, proba

    cond2 = mdl.prepare_conditional(
        cfg.TARGET_BRAND, df=seg_mod.attach_segment(df, modelo_m0), extra=["segmento"]
    )
    cond3 = mdl.prepare_conditional(
        cfg.TARGET_QUANTITY, df=seg_mod.attach_segment(df, modelo_m0), extra=["segmento"]
    )
    X2tr, X2te = mdl.drop_features(cond2.X_train, cond2.X_test, mdl.TREND_COLS)
    X3tr, X3te = mdl.drop_features(cond3.X_train, cond3.X_test, mdl.TREND_COLS)
    pipe_m2 = mdl.pipeline(mdl.m2_candidates()["2 · random forest"], X2tr).fit(X2tr, cond2.y_train)
    pipe_m3 = mdl.pipeline(mdl.m3_candidates()["1 · poisson"], X3tr).fit(X3tr, cond3.y_train)
    R["m2_test"] = mdl.evaluate_multiclass(cond2.y_test, pipe_m2.predict_proba(X2te), pipe_m2.classes_)
    R["m3_test"] = mdl.evaluate_count(
        cond3.y_test, pipe_m3.predict(X3te), float(cond3.y_train.mean())
    )
    R["elasticidades"] = mdl.cross_price_elasticities(cond2)
    R["diag_elasticidad"] = mdl.elasticity_diagnostics(cond2)
    valores = mdl.expected_value(
        proba, pipe_m2.predict_proba(X_test), pipe_m2.classes_, pipe_m3.predict(X_test),
        df, prep.X_test.index,
    )
    R["captura"] = mdl.value_capture(valores, X_test)

    log("· capa prescriptiva (P1/P2/P3)")
    bundle = px.fit_bundle(df, modelo_m0)
    supuesto = px.MarginAssumption()
    base = px.score_scenario(bundle, df, supuesto)
    cantidad = float(base["lambda"].mean())
    a = supuesto.costo_contacto(df, cantidad)
    d = supuesto.costo_promo_dia(df, cantidad)
    R["margen_base"] = float(base["margen_total"].sum())

    R["frontera"] = px.p1_frontier(
        base, a, fracciones=(0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50), fatiga=8
    )
    R["p1"] = px.solve_p1(base, a, int(0.20 * len(base)), fatiga=8)
    R["politicas"] = px.p1_vs_policies(base, X_test, a, int(0.20 * len(base)), fatiga=8)
    R["verif_p1"] = px.verify_p1_optimality(base, a, int(0.20 * len(base)), fatiga=8)

    valores_promo = px.promo_value_matrix(bundle, df, supuesto)
    calendario = px.observed_calendar(df)
    R["valor_promo"] = valores_promo
    R["p2"] = px.solve_p2(valores_promo, calendario, d, supuesto.tope_promo_por_marca)
    R["verif_p2"] = px.verify_p2_additivity(
        bundle, df, supuesto, R["p2"].attrs["calendario"], R["p2"].attrs["ganancia"], d
    )

    grilla = px.price_grid(df, supuesto, banda=0.10)
    R["respuesta_precio"] = px.price_response(bundle, df, supuesto, grilla)
    R["p3_sin_r7"] = px.solve_p3(R["respuesta_precio"], imponer_r7=False)
    R["p3_con_r7"] = px.solve_p3(R["respuesta_precio"], imponer_r7=True)
    R["verif_p3"] = px.verify_p3_plan(
        bundle, df, supuesto, R["p3_con_r7"].attrs["plan"],
        R["p3_con_r7"].attrs["ganancia_aditiva_total"],
    )
    R["curvas"] = {j: px.propensity_curve(bundle, df, supuesto, j) for j in (1, 3, 4, 5)}
    R["elasticidad_seg"] = px.elasticity_by_segment(bundle, df, supuesto, marca=5)
    R["sensibilidad"] = px.margin_sensitivity(bundle, df)
    R["kpis"] = px.kpi_table(
        R["politicas"], R["p1"], R["p2"], R["verif_p3"],
        float(R["contra_baseline"].attrs["lift_modelo_top10"]),
        float(R["contra_baseline"].attrs["lift_baseline_top10"]),
        R["margen_base"],
    )
    R["impacto"] = px.business_area_impact(
        R["p1"], R["p2"], R["elasticidad_seg"], R["p3_con_r7"], base
    )
    R["supuesto_tabla"] = supuesto.as_table(df, cantidad)
    R["margenes"] = supuesto.margenes(df)
    R["costos"] = {"contacto": a, "promo_dia": d, "cantidad_media": cantidad}
    log("· listo")
    return R


def _pr_curve(y, proba, puntos: int = 60) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_curve

    p, r, _ = precision_recall_curve(y, proba)
    idx = np.linspace(0, len(p) - 1, puntos).astype(int)
    return pd.DataFrame({"recall": r[idx].round(4), "precision": p[idx].round(4)})


def _calibration(y, proba, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"p": proba, "y": np.asarray(y)})
    d["bin"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(
        predicha=("p", "mean"), observada=("y", "mean"), n=("y", "size")
    )
    return g.reset_index(drop=True).round(4)


# --------------------------------------------------------------------------
# dashboard/data.json
# --------------------------------------------------------------------------


def build_payload(R: dict) -> dict:
    """Shape the computed tables into what ``index.html`` consumes."""
    encuadre, segs, modelos, presc = (
        _frozen("encuadre.json"), _frozen("segmentos.json"),
        _frozen("modelos.json"), _frozen("prescriptivo.json"),
    )
    df = R["df"]
    p1, p2, p3 = R["p1"], R["p2"], R["p3_con_r7"]
    politicas = R["politicas"]
    # La ganancia de P1 se mide contra la **politica vigente** (ordenar por tasa
    # historica), que es lo que la empresa hace hoy. Compararla contra ordenar
    # por propension responde otra pregunta -- de diseno, no de negocio -- y
    # mezclar las dos daria dos cifras distintas en la misma pagina.
    ganancia_p1 = float(
        politicas[politicas["politica"].str.startswith("P1")]["objetivo_neto"].iloc[0]
        - politicas[politicas["politica"].str.startswith("politica historica")]["objetivo_neto"].iloc[0]
    )

    return {
        "meta": {
            "proyecto": "Modelo de Propension de Compra · Ferreycorp",
            "generado": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
            "filas": int(len(df)),
            "columnas": int(df.shape[1]),
            "clientes": int(df[cfg.KEY_CLIENT].nunique()),
            "dias": int(df[cfg.KEY_TIME].max()),
            "split_day": cfg.SPLIT_DAY,
            "aviso": "La pagina no ejecuta los modelos en vivo: cada combinacion de los controles se calculo al generar data.js.",
            "huellas": {
                "encuadre_train": encuadre.get("split", {}).get("train", {}).get("fingerprint"),
                "encuadre_test": encuadre.get("split", {}).get("test", {}).get("fingerprint"),
                "segmentos": segs.get("fingerprint"),
                "modelos": modelos.get("fecha"),
                "prescriptivo": presc.get("fecha"),
            },
            "fases": [
                {"fase": "0 · Encuadre", "estado": "congelado", "detalle": "split dia 550 con huella SHA-256"},
                {"fase": "1 · EDA", "estado": "cerrado", "detalle": "20 figuras, 13/13 controles de tratamiento"},
                {"fase": "2 · Features", "estado": "cerrado", "detalle": "83 columnas, ajuste solo en train (E10)"},
                {"fase": "3 · Segmentacion M0", "estado": "congelado", "detalle": f"k=3, ARI 1.0, huella {str(segs.get('fingerprint'))[:8]}"},
                {"fase": "4 · M1/M2/M3", "estado": "cerrado", "detalle": f"PR-AUC {modelos.get('m1', {}).get('pr_auc_test')}"},
                {"fase": "5 · P1/P2/P3", "estado": "cerrado", "detalle": "optimalidad de P1 verificada contra el LP"},
                {"fase": "6 · Despliegue", "estado": "documentado", "detalle": "arquitectura multi-cloud + scoring ejecutable"},
            ],
        },
        "resumen": {
            "tasa_base": float(R["tasa"]["tasa_compra"]),
            "pr_auc": float(R["m1_test"]["pr_auc"]),
            "pr_auc_azar": float(R["m1_test"]["pr_auc_azar"]),
            "lift_top10": float(R["contra_baseline"].attrs["lift_modelo_top10"]),
            "lift_baseline": float(R["contra_baseline"].attrs["lift_baseline_top10"]),
            "margen_base": round(R["margen_base"], 2),
            "ganancia_p1": round(ganancia_p1, 2),
            "ganancia_p2": float(p2.attrs["ganancia"]),
            "ganancia_p3": float(R["verif_p3"].attrs["ganancia_real"]),
            "captura_top10": float(
                R["captura"][
                    (R["captura"]["presupuesto"] == "top 10%")
                    & (R["captura"]["politica"] == "valor esperado (M1xM2xM3)")
                ]["pct_del_ingreso"].iloc[0]
            ),
            "captura_historica_top10": float(
                R["captura"][
                    (R["captura"]["presupuesto"] == "top 10%")
                    & (R["captura"]["politica"] == "politica historica")
                ]["pct_del_ingreso"].iloc[0]
            ),
        },
        "eda": {
            "marcas": _records(
                R["brand_share"]
                .rename(columns={cfg.TARGET_BRAND: "marca"})
                .merge(R["promo_lift"][["marca", "frecuencia_promo", "lift", "respuesta_vs_inversion"]], on="marca")
                .merge(R["promo_incidencia"][["marca", "delta_pp", "z", "efecto"]], on="marca")
                .merge(R["ladder"][["marca", "n_puntos_precio", "precio_min", "precio_max"]], on="marca")
            ),
            "brecha_precio": _records(R["gap"]),
            "temporal": _records(R["temporal"]),
            "anomalias": _records(R["anomalias"]),
            "demografia": _records(R["demografia"]),
            "baseline": _records(R["baseline"]),
            "lecturas": {
                "temporal": R["temporal"].attrs["lectura"],
                "promo": R["promo_incidencia"].attrs["lectura"],
                "anomalias": R["anomalias"].attrs["lectura"],
                "brecha": R["gap"].attrs["lectura"],
            },
        },
        "segmentos": {
            "k": int(R["m0"].k),
            "silhouette": float(R["m0"].silhouette),
            "nombres": {str(k): v for k, v in R["m0"].nombres.items()},
            "perfil": _records(R["perfil_seg"]),
            "comportamiento": _records(R["comportamiento_seg"]),
            "marcas": _records(R["marcas_seg"]),
            "promo": _records(R["promo_seg"]),
            "seleccion_k": _records(R["seleccion_k"]),
            "pca": _records(R["pca"].reset_index()[["id", "pc1", "pc2", "segmento"]].round(3)),
            "lecturas": {
                "comportamiento": R["comportamiento_seg"].attrs["lectura"],
                "marcas": R["marcas_seg"].attrs["lectura"],
                "promo": R["promo_seg"].attrs["lectura"],
                "seleccion_k": R["seleccion_k"].attrs["lectura"],
            },
        },
        "modelos": {
            "m1": json.loads(R["m1_test"].to_json(double_precision=6)),
            "m2": json.loads(R["m2_test"].to_json(double_precision=6)),
            "m3": json.loads(R["m3_test"].to_json(double_precision=6)),
            "lift": _records(R["lift"]),
            "contra_baseline": _records(R["contra_baseline"]),
            "umbral": _records(R["umbral"]),
            "shap": _records(R["shap"]),
            "curva_pr": _records(R["curva_pr"]),
            "calibracion": _records(R["calibracion"]),
            "captura": _records(R["captura"]),
            "elasticidades": _records(R["elasticidades"].reset_index(names="share")),
            "diag_elasticidad": _records(R["diag_elasticidad"]),
            "lecturas": {
                "umbral": R["umbral"].attrs["lectura"],
                "shap": R["shap"].attrs["lectura"],
                "captura": R["captura"].attrs["lectura"],
                "elasticidades": R["elasticidades"].attrs["lectura"],
                "diag": R["diag_elasticidad"].attrs["lectura"],
                "contra_baseline": R["contra_baseline"].attrs["lectura"],
            },
        },
        "decisiones": {
            "supuestos": _records(R["supuesto_tabla"]),
            "margenes": _records(R["margenes"]),
            "costos": {k: round(float(v), 4) for k, v in R["costos"].items()},
            "p1": {
                "frontera": _records(R["frontera"]),
                "politicas": _records(R["politicas"]),
                "verificacion": _records(R["verif_p1"]),
                "contactos_utiles": int(R["frontera"].attrs["contactos_utiles"]),
                "pct_visitas_utiles": float(R["frontera"].attrs["pct_visitas_utiles"]),
                "marcas_ofrecidas": {str(k): int(v) for k, v in p1.attrs["marcas_ofrecidas"].items()},
                "lectura": politicas.attrs["lectura"],
            },
            "p2": {
                "reasignacion": _records(p2),
                "valor_observado": float(p2.attrs["valor_observado"]),
                "valor_optimo": float(p2.attrs["valor_optimo"]),
                "ganancia": float(p2.attrs["ganancia"]),
                "dias_presupuesto": int(p2.attrs["dias_gastados_observado"]),
                "error_aditivo": float(R["verif_p2"].attrs["error_relativo"]),
                "ganancia_verificada": float(R["verif_p2"].attrs["ganancia_real"]),
                "lectura": p2.attrs["lectura"],
            },
            "p3": {
                "sin_r7": _records(R["p3_sin_r7"]),
                "con_r7": _records(R["p3_con_r7"]),
                "costo_r7": round(
                    float(R["p3_sin_r7"].attrs["ganancia_aditiva_total"]
                          - R["p3_con_r7"].attrs["ganancia_aditiva_total"]), 2
                ),
                "respuesta": _records(R["respuesta_precio"]),
                "ganancia_verificada": float(R["verif_p3"].attrs["ganancia_real"]),
                "marcas_congeladas": list(px.MARCAS_SIN_ELASTICIDAD),
                "lectura": R["verif_p3"].attrs["lectura"],
            },
            "curvas": {
                str(j): {
                    "puntos": _records(c),
                    "identificada": bool(c.attrs["identificada"]),
                    "lectura": c.attrs["lectura"],
                }
                for j, c in R["curvas"].items()
            },
            "elasticidad_segmento": _records(R["elasticidad_seg"]),
            "sensibilidad": _records(R["sensibilidad"]),
            "kpis": _records(R["kpis"]),
            "impacto": _records(R["impacto"]),
            "lecturas": {
                "sensibilidad": R["sensibilidad"].attrs["lectura"],
                "kpis": R["kpis"].attrs["lectura"],
                "elasticidad_segmento": R["elasticidad_seg"].attrs["lectura"],
            },
        },
    }


def export_data(R: dict | None = None, path: Path | None = None) -> Path:
    """Write ``dashboard/data.json`` and its ``data.js`` twin.

    Both, on purpose. Opening ``index.html`` with a double click means the page
    runs on ``file://``, where ``fetch`` of a local JSON is blocked by CORS but
    a ``<script src>`` is not. So the page reads ``data.js``, and the ``.json``
    stays for anything else that wants the numbers.
    """
    R = R or compute()
    path = path or (DASHBOARD_DIR / "data.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(R)
    crudo = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    path.write_text(crudo, encoding="utf-8")
    (path.parent / "data.js").write_text(
        "// Generado por src/ferreyros/dashboard.py -- no editar a mano.\n"
        "window.FERREYROS_DATA = " + crudo + ";\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# docs/figuras/*.png -- shared by the documentation and the deck
# --------------------------------------------------------------------------


def export_figures(R: dict | None = None, directorio: Path | None = None) -> list[Path]:
    """Render the figures the PDF deliverables embed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import plots

    R = R or compute()
    directorio = directorio or FIGURES_DIR
    directorio.mkdir(parents=True, exist_ok=True)
    plots.set_style()

    tareas = [
        ("eda_promo_incidencia", (9, 4.2), lambda ax: plots.plot_promo_incidence_by_brand(R["promo_incidencia"], ax=ax)),
        ("eda_promo_lift", (9, 4.2), lambda ax: plots.plot_promo_lift(R["promo_lift"], ax=ax)),
        ("eda_baseline", (9, 4.2), lambda ax: plots.plot_targeting_baseline(R["baseline"], ax=ax)),
        ("m0_segmentos_pca", (7.5, 5.2), lambda ax: plots.plot_pca_segments(R["pca"], ax=ax)),
        ("m0_marcas_por_segmento", (8.5, 3.6), lambda ax: plots.plot_segment_brand_share(R["marcas_seg"], ax=ax)),
        ("m1_pr", (7.5, 4.8), lambda ax: plots.plot_pr_roc(R["y_test"], R["proba_m1"], ax=ax)),
        ("m1_umbral", (9, 4.2), lambda ax: plots.plot_threshold_value(R["y_test"], R["proba_m1"], ax=ax)),
        ("m1_shap", (8, 5), lambda ax: plots.plot_shap_global(R["shap"], ax=ax)),
        ("m1_captura", (8.5, 4.5), lambda ax: plots.plot_value_capture(R["captura"], ax=ax)),
        ("m2_elasticidades", (8, 4), lambda ax: plots.plot_elasticities(R["elasticidades"], ax=ax)),
        ("p1_frontera", (9, 4.2), lambda ax: plots.plot_p1_frontier(R["frontera"], ax=ax)),
        ("p2_reasignacion", (9, 4.2), lambda ax: plots.plot_promo_reallocation(R["p2"], ax=ax)),
        ("p3_respuesta_precio", (9, 4.2), lambda ax: plots.plot_price_response(R["respuesta_precio"], ax=ax)),
        ("p5_curvas_propension", (8.5, 4.5), lambda ax: plots.plot_propensity_curves(R["curvas"], ax=ax)),
        ("p5_waterfall", (8, 4.5), lambda ax: plots.plot_value_waterfall(
            R["margen_base"],
            {
                "P1 targeting": round(float(
                    R["politicas"][R["politicas"]["politica"].str.startswith("P1")]["objetivo_neto"].iloc[0]
                    - R["politicas"][R["politicas"]["politica"].str.startswith("politica historica")]["objetivo_neto"].iloc[0]
                ), 2),
                "P2 calendario": float(R["p2"].attrs["ganancia"]),
                "P3 pricing": float(R["verif_p3"].attrs["ganancia_real"]),
            },
            ax=ax,
        )),
    ]

    escritas = []
    for nombre, tamanio, dibuja in tareas:
        fig, ax = plt.subplots(figsize=tamanio)
        dibuja(ax)
        fig.tight_layout()
        ruta = directorio / f"{nombre}.png"
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        plt.close(fig)
        escritas.append(ruta)
    return escritas


def export_all(verbose: bool = True) -> dict:
    """One pass: compute, write ``data.json`` and render the figures."""
    R = compute(verbose=verbose)
    datos = export_data(R)
    figuras = export_figures(R)
    if verbose:
        print(f"· {datos.relative_to(cfg.PROJECT_ROOT)} ({datos.stat().st_size / 1024:.0f} KB)")
        print(f"· {len(figuras)} figuras en {FIGURES_DIR.relative_to(cfg.PROJECT_ROOT)}")
    return {"data": datos, "figuras": figuras}
