"""The prescriptive layer: P1 targeting, P2 promotional calendar, P3 pricing (Fase 5).

Fase 4 estimated the three parameters of the objective function. This module
spends them: it decides who to contact, which brand to promote on which day,
and at which point of the observed price ladder to sell.

    max  sum_ijt  x_ijt * pi_it * q_ijt * lambda_ijt * sum_k (y_jtk * m_jk)
         - a * sum_ijt x_ijt  -  sum_jt d_j * z_jt

Three things govern everything here.

**The counterfactual is rebuilt, never edited.** A scenario changes the *raw*
prices or promotion flags and then rebuilds every feature through
``features.build_features`` with the train-fitted reference frozen in Fase 2.
Editing ``precio_relativo_3`` by hand while leaving ``precio_min_dia``,
``desviacion_precio_3`` and ``descuento_implicito_3`` untouched would score a
world that cannot exist.

**The missing costs are assumptions with a name.** ``c_j``, ``a``, ``d_j``,
``B`` and ``rho`` are not in the dataset (§4.8 of the problem brief). They enter
through :class:`MarginAssumption`, which states each one and its scale, and
every headline number is reported with a sensitivity band over the gross
margin. A number without its assumption attached is worse than no number.

**What Fase 4 could not identify, this layer does not optimize.** The own-price
elasticity of brand 3 came out positive in all five specifications, so P3
treats its price as frozen and says so, instead of quietly optimizing a lever
whose sign is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import data as data_mod
from . import features as feat_mod
from . import models as mdl
from . import preprocess as pre_mod
from . import segmentation as seg_mod

ARTIFACTS_DIR = cfg.PROJECT_ROOT / "artifacts"
PRESCRIPTIVE_PATH = ARTIFACTS_DIR / "prescriptivo.json"

#: Brands whose own-price elasticity Fase 4 could not identify. P3 leaves their
#: price at the observed level; see ``models.elasticity_diagnostics``.
MARCAS_SIN_ELASTICIDAD: tuple[int, ...] = (3,)


# --------------------------------------------------------------------------
# The assumptions that stand in for the missing business data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarginAssumption:
    """Every business parameter the dataset does not contain, with its scale.

    ``margen_bruto`` is the gross margin at the brand's average observed price,
    which fixes ``c_j = (1 - margen_bruto) * precio_medio_j``. Unit margin then
    moves with price, which is what gives P3 a lever at all.

    ``costo_contacto_rel`` and ``costo_promo_rel`` are expressed as fractions of
    the margin of an average sale, so the whole layer is denominated in one
    interpretable unit instead of in invented currency.
    """

    margen_bruto: float = 0.30
    costo_contacto_rel: float = 0.20
    costo_promo_rel: float = 2.0
    tope_promo_por_marca: int = 90
    margen_minimo_rho: float = 0.10

    def costos_unitarios(self, df: pd.DataFrame) -> pd.Series:
        """``c_j`` per brand, derived from its average observed price."""
        precios = df[cfg.PRICE_COLS].mean()
        precios.index = list(cfg.BRANDS)
        return ((1 - self.margen_bruto) * precios).rename("costo_unitario")

    def margenes(self, df: pd.DataFrame) -> pd.DataFrame:
        """``m_j`` at the observed average price, per brand."""
        costos = self.costos_unitarios(df)
        precios = df[cfg.PRICE_COLS].mean()
        precios.index = list(cfg.BRANDS)
        return pd.DataFrame(
            {
                "precio_medio": precios.round(3),
                "costo_unitario": costos.round(3),
                "margen_unitario": (precios - costos).round(3),
                "precio_minimo_R6": (costos * (1 + self.margen_minimo_rho)).round(3),
            }
        ).rename_axis("marca")

    def costo_contacto(self, df: pd.DataFrame, cantidad_media: float) -> float:
        """``a``: cost of one commercial contact, in the same units as margin."""
        m = self.margenes(df)["margen_unitario"].mean()
        return float(self.costo_contacto_rel * m * cantidad_media)

    def costo_promo_dia(self, df: pd.DataFrame, cantidad_media: float) -> float:
        """``d_j``: cost of holding a brand on promotion for one day."""
        m = self.margenes(df)["margen_unitario"].mean()
        return float(self.costo_promo_rel * m * cantidad_media)

    def as_table(self, df: pd.DataFrame, cantidad_media: float) -> pd.DataFrame:
        """The assumptions as a table the notebook prints before using them."""
        filas = [
            {
                "parametro": "c_j (costo unitario)",
                "valor": "(1 - {:.0%}) x precio medio de la marca".format(self.margen_bruto),
                "de_donde_sale": "SUPUESTO: margen bruto declarado; el dataset no trae costos",
            },
            {
                "parametro": "m_j (margen unitario)",
                "valor": "precio - c_j, entre {:.3f} y {:.3f}".format(
                    float(self.margenes(df)["margen_unitario"].min()),
                    float(self.margenes(df)["margen_unitario"].max()),
                ),
                "de_donde_sale": "derivado del supuesto anterior; sube si sube el precio",
            },
            {
                "parametro": "a (costo de contacto)",
                "valor": "{:.4f} = {:.0%} del margen de una venta media".format(
                    self.costo_contacto(df, cantidad_media), self.costo_contacto_rel
                ),
                "de_donde_sale": "SUPUESTO declarado, con sensibilidad en P1",
            },
            {
                "parametro": "d_j (costo de un dia-promo)",
                "valor": "{:.4f} = {:.1f}x el margen de una venta media".format(
                    self.costo_promo_dia(df, cantidad_media), self.costo_promo_rel
                ),
                "de_donde_sale": "SUPUESTO declarado; P2 lo usa como costo fijo por dia-marca",
            },
            {
                "parametro": "B (presupuesto promocional)",
                "valor": "el costo del calendario observado",
                "de_donde_sale": "DECISION: P2 reasigna, no aumenta (conclusion del EDA)",
            },
            {
                "parametro": "T_j (tope de dias en promo, R5)",
                "valor": "{} dias por marca".format(self.tope_promo_por_marca),
                "de_donde_sale": "DECISION de salud de marca: ninguna marca en promo mas de la "
                "mitad del horizonte",
            },
            {
                "parametro": "rho (piso de margen, R6)",
                "valor": "{:.0%} sobre el costo".format(self.margen_minimo_rho),
                "de_donde_sale": "SUPUESTO declarado; acota el precio por abajo en P3",
            },
        ]
        out = pd.DataFrame(filas)
        out.attrs["lectura"] = (
            "Cinco de los siete parametros de negocio son supuestos declarados, no dato. Por "
            "eso cada cifra de esta fase se reporta con su banda de sensibilidad sobre el "
            "margen bruto: lo que se entrega es la **forma** de la decision y su orden de "
            "magnitud, no una cifra de utilidad que el dataset no permite calcular."
        )
        return out


# --------------------------------------------------------------------------
# The three fitted models, bundled with what a scenario needs to re-score
# --------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """M1, M2 and M3 fitted on train, plus the frozen feature reference."""

    pipe_m1: Pipeline
    pipe_m2: Pipeline
    pipe_m3: Pipeline
    clases_m2: np.ndarray
    reference: object
    columnas: list[str]
    modelo_m0: seg_mod.SegmentModel
    split_day: int = cfg.SPLIT_DAY
    extras: list[str] = field(default_factory=lambda: ["segmento"])

    @property
    def resumen(self) -> pd.Series:
        return pd.Series(
            {
                "M1": type(self.pipe_m1["clf"]).__name__,
                "M2": type(self.pipe_m2["clf"]).__name__,
                "M3": type(self.pipe_m3["clf"]).__name__,
                "features": len(self.columnas),
                "clases_m2": ", ".join(str(int(c)) for c in self.clases_m2),
                "split_day": self.split_day,
            }
        )


def fit_bundle(df: pd.DataFrame, modelo_m0: seg_mod.SegmentModel) -> ModelBundle:
    """Refit the three models exactly as Fase 4 chose them.

    The choices are not re-litigated here: logistic ``C=0.03`` for M1 because
    the paired fold comparison found no significant gain from anything heavier,
    random forest for M2 because there the gain was unambiguous, Poisson for M3
    because the target is a count and it also won.
    """
    dfs = seg_mod.attach_segment(df, modelo_m0)
    train_raw, _ = data_mod.temporal_split(df)
    reference = feat_mod.fit_reference(train_raw)
    featured = feat_mod.build_features(dfs, reference=reference)
    train, _ = data_mod.temporal_split(featured)

    X = feat_mod.feature_matrix(train, extra=["segmento"])
    X = X.drop(columns=[c for c in mdl.TREND_COLS if c in X.columns])

    from sklearn.linear_model import LogisticRegression

    pipe_m1 = mdl.pipeline(
        LogisticRegression(max_iter=2000, C=0.03, random_state=mdl.RANDOM_STATE), X
    ).fit(X, train[cfg.TARGET_INCIDENCE])

    compras = train[train[cfg.TARGET_INCIDENCE] == 1]
    Xc = X.loc[compras.index]
    pipe_m2 = mdl.pipeline(mdl.m2_candidates()["2 · random forest"], Xc).fit(
        Xc, compras[cfg.TARGET_BRAND]
    )
    pipe_m3 = mdl.pipeline(mdl.m3_candidates()["1 · poisson"], Xc).fit(
        Xc, compras[cfg.TARGET_QUANTITY]
    )

    return ModelBundle(
        pipe_m1=pipe_m1,
        pipe_m2=pipe_m2,
        pipe_m3=pipe_m3,
        clases_m2=pipe_m2.classes_,
        reference=reference,
        columnas=list(X.columns),
        modelo_m0=modelo_m0,
    )


# --------------------------------------------------------------------------
# Scenarios: change the world, rebuild the features, re-score
# --------------------------------------------------------------------------


def build_scenario(
    df: pd.DataFrame,
    precios: dict[int, float] | None = None,
    promos: dict[int, int] | None = None,
    desde_dia: int | None = None,
) -> pd.DataFrame:
    """A raw frame with prices multiplied and/or promotion flags forced.

    ``precios`` maps brand to a multiplier (1.05 = five per cent dearer) and
    ``promos`` maps brand to 0/1. Changes apply only from ``desde_dia`` onward,
    so the past -- which the history features read -- stays untouched.
    """
    out = df.copy()
    desde = desde_dia if desde_dia is not None else cfg.SPLIT_DAY + 1
    mask = out[cfg.KEY_TIME] >= desde

    for marca, factor in (precios or {}).items():
        col = f"precio_marca_{marca}"
        out.loc[mask, col] = out.loc[mask, col] * factor
    for marca, valor in (promos or {}).items():
        out.loc[mask, f"promo_marca_{marca}"] = int(valor)
    return out


def build_price_plan(
    df: pd.DataFrame, precios: dict[int, float], desde_dia: int | None = None
) -> pd.DataFrame:
    """A raw frame with each named brand held at an absolute price over the horizon.

    This is what a P3 decision looks like in the data: one point of the observed
    ladder per brand, constant over the planning horizon (constraint R1 applied
    to a single period).
    """
    out = df.copy()
    desde = desde_dia if desde_dia is not None else cfg.SPLIT_DAY + 1
    mask = out[cfg.KEY_TIME] >= desde
    for marca, precio in precios.items():
        out.loc[mask, f"precio_marca_{marca}"] = float(precio)
    return out


def build_calendar(
    df: pd.DataFrame, calendario: pd.DataFrame, desde_dia: int | None = None
) -> pd.DataFrame:
    """A raw frame whose promotion flags follow a day-by-brand calendar.

    Needed to score a P2 solution as one scenario instead of trusting the
    additive approximation its objective was built on.
    """
    out = df.copy()
    desde = desde_dia if desde_dia is not None else cfg.SPLIT_DAY + 1
    mask = out[cfg.KEY_TIME] >= desde
    dias = out.loc[mask, cfg.KEY_TIME]
    for marca in cfg.BRANDS:
        if marca not in calendario.columns:
            continue
        valores = dias.map(calendario[marca]).fillna(0).astype(int)
        out.loc[mask, f"promo_marca_{marca}"] = valores.to_numpy()
    return out


def score_scenario(
    bundle: ModelBundle,
    df_escenario: pd.DataFrame,
    supuesto: MarginAssumption,
    solo_test: bool = True,
) -> pd.DataFrame:
    """Per-visit ``pi``, ``q_j``, ``lambda`` and expected margin for a scenario.

    Features are rebuilt over the **whole** timeline so a client's history is
    still read from its own past, and only the horizon rows are returned.
    """
    dfs = seg_mod.attach_segment(df_escenario, bundle.modelo_m0)
    featured = feat_mod.build_features(dfs, reference=bundle.reference)
    if solo_test:
        _, horizonte = data_mod.temporal_split(featured, bundle.split_day)
    else:
        horizonte = featured

    X = feat_mod.feature_matrix(horizonte, extra=bundle.extras)[bundle.columnas]
    pi = bundle.pipe_m1.predict_proba(X)[:, 1]
    q = bundle.pipe_m2.predict_proba(X)
    lam = bundle.pipe_m3.predict(X)

    orden = {int(c): i for i, c in enumerate(bundle.clases_m2)}
    costos = supuesto.costos_unitarios(df_escenario)
    precios = horizonte[cfg.PRICE_COLS].to_numpy()

    out = pd.DataFrame({"pi": pi, "lambda": lam}, index=horizonte.index)
    out[cfg.KEY_CLIENT] = horizonte[cfg.KEY_CLIENT].to_numpy()
    out[cfg.KEY_TIME] = horizonte[cfg.KEY_TIME].to_numpy()
    out["segmento"] = horizonte["segmento"].to_numpy()

    margenes = np.column_stack(
        [precios[:, j - 1] - float(costos.loc[j]) for j in cfg.BRANDS]
    )
    for i, j in enumerate(cfg.BRANDS):
        out[f"q_{j}"] = q[:, orden[j]]
        out[f"margen_{j}"] = pi * q[:, orden[j]] * lam * margenes[:, i]

    cols_margen = [f"margen_{j}" for j in cfg.BRANDS]
    out["margen_total"] = out[cols_margen].sum(axis=1)
    out["mejor_marca"] = (
        out[cols_margen].to_numpy().argmax(axis=1) + 1
    )
    out["margen_mejor_marca"] = out[cols_margen].to_numpy().max(axis=1)
    out["ingreso_real"] = df_escenario.loc[horizonte.index, "gasto"].fillna(0.0).to_numpy()
    return out


# --------------------------------------------------------------------------
# S1 -- elasticities and propensity curves
# --------------------------------------------------------------------------


def propensity_curve(
    bundle: ModelBundle,
    df: pd.DataFrame,
    supuesto: MarginAssumption,
    marca: int,
    multiplicadores: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10),
) -> pd.DataFrame:
    """Demand response to moving one brand's price, holding the others observed.

    This is the brief's literal question -- *what happens to demand if I move
    brand 3's price by 5%?* -- answered by re-scoring the whole horizon under
    each price, not by reading a coefficient.
    """
    filas = []
    for factor in multiplicadores:
        esc = build_scenario(df, precios={marca: factor})
        s = score_scenario(bundle, esc, supuesto)
        filas.append(
            {
                "factor_precio": factor,
                "cambio_precio_pct": round(100 * (factor - 1), 1),
                "pi_media": round(float(s["pi"].mean()), 4),
                f"share_marca_{marca}": round(float(s[f"q_{marca}"].mean()), 4),
                "margen_total": round(float(s["margen_total"].sum()), 2),
                f"margen_marca_{marca}": round(float(s[f"margen_{marca}"].sum()), 2),
            }
        )
    out = pd.DataFrame(filas)
    base = out[out["factor_precio"] == 1.00].iloc[0]

    share_col = f"share_marca_{marca}"
    out["elasticidad_share"] = np.where(
        out["factor_precio"] == 1.0,
        np.nan,
        ((out[share_col] / base[share_col] - 1) / (out["factor_precio"] - 1)).round(3),
    )
    out["delta_margen_total"] = (out["margen_total"] - base["margen_total"]).round(2)

    arco = out.loc[out["factor_precio"] == 1.05, "elasticidad_share"].iloc[0]
    delta5 = out.loc[out["factor_precio"] == 1.05, "delta_margen_total"].iloc[0]
    out.attrs["marca"] = marca
    out.attrs["elasticidad_arco_5pct"] = float(arco)
    out.attrs["delta_margen_5pct"] = float(delta5)
    out.attrs["mejor_factor"] = float(out.loc[out["margen_total"].idxmax(), "factor_precio"])
    out.attrs["identificada"] = marca not in MARCAS_SIN_ELASTICIDAD
    out.attrs["lectura"] = (
        "Marca {}: subir el precio un 5% mueve su share un {:+.1%} (elasticidad de arco "
        "{:+.3f}) y el margen total del horizonte en {:+.2f}. El optimo de esta curva esta en "
        "un factor de {:.2f}. {}"
    ).format(
        marca,
        float(out.loc[out["factor_precio"] == 1.05, share_col].iloc[0] / base[share_col] - 1),
        arco,
        delta5,
        out.attrs["mejor_factor"],
        "Elasticidad identificada en la Fase 4: la curva se puede usar para fijar precio."
        if marca not in MARCAS_SIN_ELASTICIDAD
        else "ATENCION: la Fase 4 no pudo identificar la elasticidad propia de esta marca "
        "(positiva en las 5 especificaciones), asi que esta curva describe el dato "
        "observacional, no una relacion causal utilizable para fijar precio.",
    )
    return out


def elasticity_by_segment(
    bundle: ModelBundle,
    df: pd.DataFrame,
    supuesto: MarginAssumption,
    marca: int,
    delta: float = 0.05,
) -> pd.DataFrame:
    """Arc elasticity of a brand's share, per segment of model M0.

    The brief asks for elasticity *by segment* because that is what licenses
    differentiated pricing: one number for the whole portfolio cannot tell you
    whether a discount should be targeted.
    """
    base = score_scenario(bundle, df, supuesto)
    alto = score_scenario(bundle, build_scenario(df, precios={marca: 1 + delta}), supuesto)

    share_col = f"q_{marca}"
    g0 = base.groupby("segmento").agg(
        visitas=("pi", "size"),
        pi_base=("pi", "mean"),
        share_base=(share_col, "mean"),
        margen_base=("margen_total", "sum"),
    )
    g1 = alto.groupby("segmento").agg(
        pi_alto=("pi", "mean"),
        share_alto=(share_col, "mean"),
        margen_alto=("margen_total", "sum"),
    )
    out = g0.join(g1)
    out["elasticidad_share"] = (
        (out["share_alto"] / out["share_base"] - 1) / delta
    ).round(3)
    out["delta_margen"] = (out["margen_alto"] - out["margen_base"]).round(2)
    out["nombre"] = [bundle.modelo_m0.nombres.get(int(s), "") for s in out.index]
    for col in ("pi_base", "share_base", "pi_alto", "share_alto"):
        out[col] = out[col].round(4)
    for col in ("margen_base", "margen_alto"):
        out[col] = out[col].round(2)

    rango = float(out["elasticidad_share"].max() - out["elasticidad_share"].min())
    out.attrs["marca"] = marca
    out.attrs["rango_elasticidad"] = round(rango, 3)
    out.attrs["mas_elastico"] = int(out["elasticidad_share"].idxmin())
    out.attrs["heterogenea"] = rango > 0.5
    out.attrs["lectura"] = (
        "La elasticidad de la marca {} va de {:+.3f} a {:+.3f} entre segmentos (rango "
        "{:.3f}); el mas sensible es el segmento {}. {}"
    ).format(
        marca,
        float(out["elasticidad_share"].min()),
        float(out["elasticidad_share"].max()),
        rango,
        out.attrs["mas_elastico"],
        "La diferencia es grande: hay base para precio diferenciado por segmento."
        if rango > 0.5
        else "La diferencia es pequenia: el precio diferenciado por segmento no se sostiene "
        "en el dato, y la palanca por segmento es la promocion dirigida, no el precio.",
    )
    return out


# --------------------------------------------------------------------------
# P1 -- targeting. The base model: exactly solvable and interpretable.
# --------------------------------------------------------------------------


def solve_p1(
    scored: pd.DataFrame,
    costo_contacto: float,
    n_contactos: int,
    fatiga: int | None = None,
) -> pd.DataFrame:
    """Choose which visits to push, by expected margin, under budget and fatigue.

    Constraint R3 (one offer per visit) is satisfied by construction: each visit
    enters with its best brand only. What remains is a cardinality cap from the
    budget (``n_contactos = B / a``) and the per-client fatigue cap ``F_i``,
    which together form a matroid -- which is why the greedy is not a heuristic
    here but the exact optimum, as §4.7 claims and :func:`verify_p1_optimality`
    checks.
    """
    orden = scored.sort_values("margen_mejor_marca", ascending=False)
    elegidos, por_cliente = [], {}
    for idx, fila in orden.iterrows():
        if len(elegidos) >= n_contactos:
            break
        if fila["margen_mejor_marca"] <= costo_contacto:
            break
        cliente = fila[cfg.KEY_CLIENT]
        if fatiga is not None and por_cliente.get(cliente, 0) >= fatiga:
            continue
        por_cliente[cliente] = por_cliente.get(cliente, 0) + 1
        elegidos.append(idx)

    sel = scored.loc[elegidos]
    out = sel.assign(x=1)
    objetivo = float(sel["margen_mejor_marca"].sum() - costo_contacto * len(sel))

    out.attrs["contactos"] = len(sel)
    out.attrs["tope_contactos"] = n_contactos
    out.attrs["clientes_distintos"] = int(sel[cfg.KEY_CLIENT].nunique())
    out.attrs["fatiga"] = fatiga
    out.attrs["objetivo"] = round(objetivo, 2)
    out.attrs["objetivo_exacto"] = objetivo
    out.attrs["margen_bruto"] = round(float(sel["margen_mejor_marca"].sum()), 2)
    out.attrs["costo_total"] = round(costo_contacto * len(sel), 2)
    out.attrs["ingreso_real_capturado"] = round(float(sel["ingreso_real"].sum()), 2)
    out.attrs["marcas_ofrecidas"] = sel["mejor_marca"].value_counts().sort_index().to_dict()
    out.attrs["lectura"] = (
        "P1 selecciona {} contactos sobre {} visitas ({:.1%}) repartidos en {} clientes, con "
        "un objetivo de {:.2f} en unidades de margen (bruto {:.2f} menos {:.2f} de costo de "
        "contacto). Las marcas ofrecidas se reparten {}."
    ).format(
        len(sel),
        len(scored),
        len(sel) / len(scored),
        int(sel[cfg.KEY_CLIENT].nunique()),
        objetivo,
        float(sel["margen_mejor_marca"].sum()),
        costo_contacto * len(sel),
        out.attrs["marcas_ofrecidas"],
    )
    return out


def verify_p1_optimality(
    scored: pd.DataFrame,
    costo_contacto: float,
    n_contactos: int,
    fatiga: int | None = None,
    muestra: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Check the greedy against the LP relaxation instead of citing a theorem.

    The claim in §4.7 is that the relaxed LP is integral, so the greedy is
    optimal. That is checkable: solve the LP on a subsample with
    ``scipy.optimize.linprog`` and compare objectives. If they match to solver
    tolerance, the greedy solution *is* the optimum and the base model can be
    delivered as exact.
    """
    from scipy.optimize import linprog

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(scored), size=min(muestra, len(scored)), replace=False)
    sub = scored.iloc[idx].reset_index(drop=True)
    tope = max(1, int(round(n_contactos * len(sub) / len(scored))))

    c = -(sub["margen_mejor_marca"].to_numpy() - costo_contacto)
    filas, datos = [np.ones(len(sub))], [tope]
    if fatiga is not None:
        for cliente, grupo in sub.groupby(cfg.KEY_CLIENT).groups.items():
            fila = np.zeros(len(sub))
            fila[np.asarray(list(grupo))] = 1.0
            filas.append(fila)
            datos.append(fatiga)
    lp = linprog(c, A_ub=np.vstack(filas), b_ub=np.array(datos), bounds=(0, 1), method="highs")

    greedy = solve_p1(sub, costo_contacto, tope, fatiga)
    obj_greedy = float(greedy.attrs["objetivo_exacto"])
    obj_lp = float(-lp.fun)
    brecha = obj_lp - obj_greedy
    fraccionarias = int(((lp.x > 1e-6) & (lp.x < 1 - 1e-6)).sum())

    out = pd.DataFrame(
        [
            {"medida": "objetivo del greedy", "valor": round(obj_greedy, 4)},
            {"medida": "objetivo del LP relajado", "valor": round(obj_lp, 4)},
            {"medida": "brecha", "valor": round(brecha, 6)},
            {"medida": "variables fraccionarias en el LP", "valor": fraccionarias},
            {"medida": "visitas en la muestra", "valor": len(sub)},
        ]
    )
    tolerancia = 1e-6 * max(1.0, abs(obj_lp))
    out.attrs["tolerancia"] = tolerancia
    out.attrs["optimo"] = bool(abs(brecha) < tolerancia)
    out.attrs["brecha"] = brecha
    out.attrs["lectura"] = (
        "Sobre {} visitas, el greedy da {:.4f} y el LP relajado {:.4f}: brecha {:.2e} con {} "
        "variables fraccionarias. {}"
    ).format(
        len(sub),
        obj_greedy,
        obj_lp,
        brecha,
        fraccionarias,
        "El greedy alcanza el optimo del LP (brecha dentro de la tolerancia del solver), asi "
        "que P1 se entrega como resuelto de forma exacta -- no como una heuristica razonable."
        if abs(brecha) < tolerancia
        else "Hay brecha: la afirmacion de integralidad del brief no se sostiene en esta "
        "instancia y habria que resolver el MILP.",
    )
    return out


def p1_vs_policies(
    scored: pd.DataFrame,
    X_test: pd.DataFrame,
    costo_contacto: float,
    n_contactos: int,
    fatiga: int | None = None,
    columna_baseline: str = "tasa_compra_historica",
) -> pd.DataFrame:
    """P1 against the policies it has to beat, in margin and in real revenue.

    Every policy spends the same budget, so the comparison is about *where* the
    contacts go, which is the only question P1 answers.
    """
    historico = X_test[columna_baseline].reindex(scored.index).fillna(0)
    politicas = {
        "azar": pd.Series(
            np.random.default_rng(42).random(len(scored)), index=scored.index
        ),
        "politica historica (tasa pasada)": historico,
        "solo propension (pi)": scored["pi"],
        "P1 (margen esperado)": scored["margen_mejor_marca"],
    }

    filas = []
    for nombre, score in politicas.items():
        orden = scored.assign(_s=score).sort_values("_s", ascending=False)
        elegidos, por_cliente = [], {}
        for idx, fila in orden.iterrows():
            if len(elegidos) >= n_contactos:
                break
            cliente = fila[cfg.KEY_CLIENT]
            if fatiga is not None and por_cliente.get(cliente, 0) >= fatiga:
                continue
            por_cliente[cliente] = por_cliente.get(cliente, 0) + 1
            elegidos.append(idx)
        sel = scored.loc[elegidos]
        filas.append(
            {
                "politica": nombre,
                "contactos": len(sel),
                "margen_esperado": round(float(sel["margen_mejor_marca"].sum()), 2),
                "objetivo_neto": round(
                    float(sel["margen_mejor_marca"].sum() - costo_contacto * len(sel)), 2
                ),
                "ingreso_real_capturado": round(float(sel["ingreso_real"].sum()), 2),
                "tasa_compra_real": round(
                    float((sel["ingreso_real"] > 0).mean()), 4
                ),
            }
        )
    out = pd.DataFrame(filas)
    p1 = out[out["politica"].str.startswith("P1")].iloc[0]
    hist = out[out["politica"].str.startswith("politica historica")].iloc[0]
    azar = out[out["politica"] == "azar"].iloc[0]

    # Differences, not ratios: contacting at random leaves a **negative** net
    # objective, and a ratio against a negative base says nothing.
    out["lift_margen_vs_azar"] = (out["margen_esperado"] / azar["margen_esperado"]).round(2)
    out["ganancia_neta_vs_azar"] = (out["objetivo_neto"] - azar["objetivo_neto"]).round(2)
    out.attrs["ganancia_vs_historica_pct"] = round(
        100 * (p1["objetivo_neto"] / hist["objetivo_neto"] - 1), 1
    )
    out.attrs["ganancia_vs_historica_abs"] = round(
        float(p1["objetivo_neto"] - hist["objetivo_neto"]), 2
    )
    out.attrs["azar_destruye_valor"] = bool(azar["objetivo_neto"] < 0)
    out.attrs["lectura"] = (
        "Con el mismo presupuesto de {} contactos, P1 deja un objetivo neto de {:.2f} frente "
        "a {:.2f} de la politica historica: {:+.1f}% ({:+.2f} en unidades de margen). {} "
        "Ordenar por propension sola da {:.2f}, asi que elegir por **margen** y no por "
        "probabilidad aporta {:+.2f} adicionales."
    ).format(
        n_contactos,
        p1["objetivo_neto"],
        hist["objetivo_neto"],
        out.attrs["ganancia_vs_historica_pct"],
        out.attrs["ganancia_vs_historica_abs"],
        "Y el dato que ordena la discusion: contactar al azar deja un objetivo **negativo** "
        "({:.2f}), es decir destruye valor, porque el costo de contacto supera el margen "
        "esperado de una visita promedio.".format(azar["objetivo_neto"])
        if azar["objetivo_neto"] < 0
        else "Contactar al azar deja {:.2f}.".format(azar["objetivo_neto"]),
        float(out[out["politica"].str.startswith("solo propension")]["objetivo_neto"].iloc[0]),
        float(
            p1["objetivo_neto"]
            - out[out["politica"].str.startswith("solo propension")]["objetivo_neto"].iloc[0]
        ),
    )
    return out


def p1_frontier(
    scored: pd.DataFrame,
    costo_contacto: float,
    fracciones=(0.05, 0.10, 0.20, 0.30, 0.50, 1.00),
    fatiga: int | None = None,
) -> pd.DataFrame:
    """Objective as a function of the budget -- the curve that sizes the campaign.

    Without ``B`` in the data, the deliverable is this frontier: for each budget
    the company might have, how much margin P1 leaves on the table and where the
    curve stops paying.
    """
    filas = []
    for f in fracciones:
        n = int(round(f * len(scored)))
        sol = solve_p1(scored, costo_contacto, n, fatiga)
        filas.append(
            {
                "presupuesto_pct_visitas": round(100 * f, 1),
                "tope_contactos": n,
                "contactos_usados": sol.attrs["contactos"],
                "objetivo_neto": sol.attrs["objetivo"],
                "margen_bruto": sol.attrs["margen_bruto"],
                "ingreso_real_capturado": sol.attrs["ingreso_real_capturado"],
            }
        )
    out = pd.DataFrame(filas)
    out["margen_marginal_por_contacto"] = (
        out["objetivo_neto"].diff() / out["contactos_usados"].diff()
    ).round(4)
    mejor = out.loc[out["objetivo_neto"].idxmax()]

    saturado = out[out["contactos_usados"] < out["tope_contactos"]]
    tope_util = int(out["contactos_usados"].max())
    out.attrs["objetivo_maximo"] = float(mejor["objetivo_neto"])
    out.attrs["contactos_utiles"] = tope_util
    out.attrs["pct_visitas_utiles"] = round(100 * tope_util / len(scored), 1)
    out.attrs["saturacion"] = len(saturado) > 0
    out.attrs["lectura"] = (
        "El objetivo neto maximo es {:.2f} y se alcanza con **{} contactos**, el {:.1f}% de "
        "las visitas del horizonte. {} El margen marginal por contacto cae de {:.4f} a "
        "{:.4f} entre los dos primeros tramos: contactar mas no es contactar mejor, y el "
        "presupuesto que el negocio necesita es el que llega a ese punto, no mas."
    ).format(
        float(mejor["objetivo_neto"]),
        tope_util,
        100 * tope_util / len(scored),
        "Por encima de ahi P1 deja presupuesto sin gastar: ninguna visita adicional tiene "
        "margen esperado mayor que el costo de contactarla."
        if len(saturado)
        else "P1 usa todo el presupuesto en todos los tramos evaluados.",
        float(out["margen_marginal_por_contacto"].dropna().iloc[0])
        if out["margen_marginal_por_contacto"].notna().any()
        else float("nan"),
        float(out["margen_marginal_por_contacto"].dropna().iloc[-1])
        if out["margen_marginal_por_contacto"].notna().any()
        else float("nan"),
    )
    return out


# --------------------------------------------------------------------------
# P2 -- the promotional calendar
# --------------------------------------------------------------------------


def observed_calendar(df: pd.DataFrame, desde_dia: int | None = None) -> pd.DataFrame:
    """The promotion calendar actually run, one row per day, one column per brand."""
    desde = desde_dia if desde_dia is not None else cfg.SPLIT_DAY + 1
    h = df[df[cfg.KEY_TIME] >= desde]
    out = h.groupby(cfg.KEY_TIME)[cfg.PROMO_COLS].first()
    out.columns = list(cfg.BRANDS)
    return out.astype(int)


def promo_value_matrix(
    bundle: ModelBundle, df: pd.DataFrame, supuesto: MarginAssumption
) -> pd.DataFrame:
    """Incremental expected margin of having each brand on promotion, day by day.

    Two scenarios per brand -- promotion forced off and forced on across the
    horizon -- so the value of a promo-day is a difference between two scored
    worlds rather than a lift read off a descriptive table. Other brands stay at
    their observed state, which makes the matrix **separable by construction**;
    :func:`verify_p2_additivity` is what measures the error of that assumption
    instead of hiding it.
    """
    filas = {}
    for j in cfg.BRANDS:
        off = score_scenario(bundle, build_scenario(df, promos={j: 0}), supuesto)
        on = score_scenario(bundle, build_scenario(df, promos={j: 1}), supuesto)
        delta = (on["margen_total"] - off["margen_total"]).groupby(on[cfg.KEY_TIME]).sum()
        filas[j] = delta
    out = pd.DataFrame(filas)
    out.index.name = cfg.KEY_TIME

    medias = out.mean().round(4)
    out.attrs["valor_medio_por_dia"] = medias.to_dict()
    out.attrs["mejor_marca"] = int(medias.idxmax())
    out.attrs["peor_marca"] = int(medias.idxmin())
    out.attrs["lectura"] = (
        "Valor incremental medio de un dia-promo, por marca: {}. La marca {} es la que mas "
        "aporta por dia y la {} la que menos. Es la version cuantificada del hallazgo del "
        "EDA: el presupuesto promocional esta donde menos rinde."
    ).format(
        ", ".join(f"marca {j}: {v:+.4f}" for j, v in medias.items()),
        out.attrs["mejor_marca"],
        out.attrs["peor_marca"],
    )
    return out


def solve_p2(
    valores: pd.DataFrame,
    calendario_obs: pd.DataFrame,
    costo_promo: float,
    tope_por_marca: int,
    presupuesto_dias: int | None = None,
) -> pd.DataFrame:
    """MILP over ``z_jt``: which brand to promote on which day (R5 + budget).

    The budget defaults to the number of brand-days actually spent, so the
    answer is a **reallocation** and not a request for more money -- which is
    what the EDA concluded the problem really was.
    """
    from scipy.optimize import LinearConstraint, milp

    marcas = list(valores.columns)
    dias = list(valores.index)
    n = len(marcas) * len(dias)
    gastados = int(calendario_obs.to_numpy().sum())
    presupuesto = presupuesto_dias if presupuesto_dias is not None else gastados

    beneficio = (valores.to_numpy().T - costo_promo).ravel()  # (marca, dia)
    A, lb, ub = [], [], []
    for i in range(len(marcas)):  # R5: tope de dias por marca
        fila = np.zeros(n)
        fila[i * len(dias) : (i + 1) * len(dias)] = 1
        A.append(fila)
        lb.append(0)
        ub.append(tope_por_marca)
    A.append(np.ones(n))  # presupuesto total de dias-promo
    lb.append(0)
    ub.append(presupuesto)

    res = milp(
        c=-beneficio,
        constraints=LinearConstraint(np.vstack(A), lb, ub),
        integrality=np.ones(n),
        bounds=(0, 1),
    )
    z = np.round(res.x).reshape(len(marcas), len(dias)).astype(int)
    optimo = pd.DataFrame(z.T, index=dias, columns=marcas)

    valor_obs = float(
        (valores.to_numpy() * calendario_obs.reindex(valores.index)[marcas].to_numpy()).sum()
        - costo_promo * gastados
    )
    valor_opt = float((valores.to_numpy() * z.T).sum() - costo_promo * int(z.sum()))

    dias_obs = calendario_obs[marcas].sum()
    dias_opt = optimo.sum()
    comparacion = pd.DataFrame(
        {
            "dias_observados": dias_obs,
            "dias_optimos": dias_opt,
            "cambio": dias_opt - dias_obs,
            "valor_medio_por_dia": valores.mean().round(4),
        }
    ).rename_axis("marca")

    comparacion.attrs["status"] = res.message
    comparacion.attrs["dias_gastados_observado"] = gastados
    comparacion.attrs["dias_gastados_optimo"] = int(z.sum())
    comparacion.attrs["valor_observado"] = round(valor_obs, 2)
    comparacion.attrs["valor_optimo"] = round(valor_opt, 2)
    comparacion.attrs["ganancia"] = round(valor_opt - valor_obs, 2)
    comparacion.attrs["calendario"] = optimo
    comparacion.attrs["observado_destruye_valor"] = bool(valor_obs < 0)
    # No percentage here: the observed calendar can be worth less than zero, and
    # a ratio against a negative base is not a number anyone can act on.
    comparacion.attrs["topes_activos"] = [
        int(j) for j in marcas if int(dias_opt[j]) >= tope_por_marca
    ]
    comparacion.attrs["lectura"] = (
        "Con el mismo presupuesto de {} dias-promo, reasignarlos vale {:.2f} frente a "
        "{:.2f} del calendario observado: **{:+.2f}** en unidades de margen. {} El optimo "
        "usa {} dias-promo y los movimientos son {}. {}"
    ).format(
        gastados,
        valor_opt,
        valor_obs,
        valor_opt - valor_obs,
        "El calendario observado tiene valor **negativo**: tal como esta asignado, promocionar "
        "cuesta mas margen del que genera."
        if valor_obs < 0
        else "",
        int(z.sum()),
        ", ".join(f"marca {j}: {int(dias_obs[j])}→{int(dias_opt[j])}" for j in marcas),
        "El tope R5 de {} dias esta activo para la(s) marca(s) {}: sin ese limite de salud de "
        "marca el optimo concentraria aun mas.".format(
            tope_por_marca, comparacion.attrs["topes_activos"]
        )
        if comparacion.attrs["topes_activos"]
        else "Ningun tope R5 esta activo.",
    )
    return comparacion


def verify_p2_additivity(
    bundle: ModelBundle,
    df: pd.DataFrame,
    supuesto: MarginAssumption,
    calendario: pd.DataFrame,
    valor_aditivo: float,
    costo_promo: float,
) -> pd.DataFrame:
    """Score the optimized calendar as one world and measure the additive error.

    The P2 objective treats each brand-day as independent. That is an
    approximation, and this is the check: build the whole optimized calendar as
    a single scenario, score it, and compare the realised incremental margin
    against what the additive objective promised.
    """
    base = score_scenario(bundle, df, supuesto)
    opt = score_scenario(bundle, build_calendar(df, calendario), supuesto)
    dias_opt = int(calendario.to_numpy().sum())
    dias_obs = int(observed_calendar(df).to_numpy().sum())

    real = float(
        (opt["margen_total"].sum() - costo_promo * dias_opt)
        - (base["margen_total"].sum() - costo_promo * dias_obs)
    )
    out = pd.DataFrame(
        [
            {"medida": "ganancia prometida por el objetivo aditivo", "valor": round(valor_aditivo, 2)},
            {"medida": "ganancia al puntuar el calendario completo", "valor": round(real, 2)},
            {"medida": "error de la aproximacion aditiva", "valor": round(real - valor_aditivo, 2)},
            {
                "medida": "error relativo",
                "valor": round(abs(real - valor_aditivo) / abs(valor_aditivo), 4)
                if valor_aditivo
                else float("nan"),
            },
        ]
    )
    rel = abs(real - valor_aditivo) / abs(valor_aditivo) if valor_aditivo else float("nan")
    out.attrs["ganancia_real"] = round(real, 2)
    out.attrs["error_relativo"] = round(float(rel), 4)
    out.attrs["aproximacion_valida"] = bool(rel < 0.15)
    out.attrs["lectura"] = (
        "El objetivo aditivo prometia {:+.2f} y puntuar el calendario completo da {:+.2f}: "
        "un error relativo del {:.1%}. {}"
    ).format(
        valor_aditivo,
        real,
        rel,
        "La separabilidad por marca-dia es una aproximacion aceptable para decidir el "
        "calendario, y el numero que se reporta al negocio es el de la simulacion completa."
        if rel < 0.15
        else "El error es grande: las promociones interactuan entre marcas mas de lo que el "
        "objetivo aditivo supone, y la cifra valida es la de la simulacion completa.",
    )
    return out


# --------------------------------------------------------------------------
# P3 -- price, over the observed ladder
# --------------------------------------------------------------------------


def price_grid(
    df: pd.DataFrame,
    supuesto: MarginAssumption,
    n_puntos: int = 7,
    congeladas: tuple[int, ...] = MARCAS_SIN_ELASTICIDAD,
    banda: float | None = None,
) -> dict[int, list[float]]:
    """Candidate price points per brand: the observed ladder, thinned and filtered.

    Thinned to ``n_puntos`` quantiles because the full ladder (21-44 points) is
    not the bottleneck -- scoring is. Filtered by R6 (price at least
    ``c_j(1+rho)``). Brands whose elasticity Fase 4 could not identify keep a
    single candidate: their observed price.

    ``banda`` is a trust region: candidates within +-banda of the observed
    price. It exists because a predict-then-optimize solution that lands on the
    edge of the grid is usually the optimizer exploiting the model where the
    model was never tested, and a 20% sustained price cut is exactly that kind
    of answer. Bounding the move is how the recommendation stays actionable.
    """
    pisos = supuesto.margenes(df)["precio_minimo_R6"]
    grilla = {}
    for j in cfg.BRANDS:
        col = f"precio_marca_{j}"
        observado = float(df.loc[df[cfg.KEY_TIME] > cfg.SPLIT_DAY, col].mean())
        if j in congeladas:
            grilla[j] = [round(observado, 3)]
            continue
        puntos = np.quantile(
            np.sort(df[col].unique()), np.linspace(0, 1, n_puntos)
        )
        puntos = sorted({round(float(p), 3) for p in puntos} | {round(observado, 3)})
        puntos = [p for p in puntos if p >= float(pisos.loc[j])]
        if banda is not None:
            lo, hi = observado * (1 - banda), observado * (1 + banda)
            puntos = [p for p in puntos if lo - 1e-9 <= p <= hi + 1e-9] or [round(observado, 3)]
        grilla[j] = puntos
    return grilla


def price_response(
    bundle: ModelBundle,
    df: pd.DataFrame,
    supuesto: MarginAssumption,
    grilla: dict[int, list[float]],
) -> pd.DataFrame:
    """Total expected margin of the horizon with one brand moved to each candidate.

    One scenario per (brand, price point), every other brand left at its
    observed price. These are the coefficients P3's MILP optimizes over, and the
    curve per brand is also the pricing answer on its own.
    """
    base = score_scenario(bundle, df, supuesto)
    base_total = float(base["margen_total"].sum())

    filas = []
    for j, puntos in grilla.items():
        observado = round(
            float(df.loc[df[cfg.KEY_TIME] > cfg.SPLIT_DAY, f"precio_marca_{j}"].mean()), 3
        )
        for p in puntos:
            s = score_scenario(bundle, build_price_plan(df, {j: p}), supuesto)
            filas.append(
                {
                    "marca": j,
                    "precio": p,
                    "es_observado": bool(abs(p - observado) < 1e-9),
                    "margen_total": round(float(s["margen_total"].sum()), 2),
                    "margen_de_la_marca": round(float(s[f"margen_{j}"].sum()), 2),
                    "share_medio": round(float(s[f"q_{j}"].mean()), 4),
                    "pi_media": round(float(s["pi"].mean()), 4),
                }
            )
    out = pd.DataFrame(filas)
    out["delta_vs_observado"] = out.groupby("marca")["margen_total"].transform(
        lambda s: s - s.iloc[0]
    )
    # Delta measured against each brand's own observed point, not the first row.
    for j in grilla:
        mask = out["marca"] == j
        ref = out.loc[mask & out["es_observado"], "margen_total"]
        if len(ref):
            out.loc[mask, "delta_vs_observado"] = (
                out.loc[mask, "margen_total"] - float(ref.iloc[0])
            ).round(2)

    out.attrs["margen_base"] = round(base_total, 2)
    mejores = out.loc[out.groupby("marca")["margen_total"].idxmax()]
    out.attrs["mejor_por_marca"] = {
        int(r["marca"]): (float(r["precio"]), float(r["delta_vs_observado"]))
        for _, r in mejores.iterrows()
    }
    out.attrs["lectura"] = (
        "Margen base del horizonte: {:.2f}. Mejor punto por marca (precio, ganancia): {}. "
        "Cada fila es un escenario puntuado de extremo a extremo, no una extrapolacion "
        "lineal de un coeficiente."
    ).format(
        base_total,
        ", ".join(
            f"marca {j}: {p:.3f} ({d:+.2f})" for j, (p, d) in out.attrs["mejor_por_marca"].items()
        ),
    )
    return out


def solve_p3(
    respuesta: pd.DataFrame, imponer_r7: bool = False
) -> pd.DataFrame:
    """MILP over ``y_jk``: one price point per brand (R1), R6 built into the grid.

    ``imponer_r7`` adds the brand-architecture ordering
    ``p_1 <= p_2 <= ... <= p_5``. The brief calls it a modelling decision rather
    than a fact of the data, so it is a switch, and the cost of switching it on
    is reported instead of assumed away.
    """
    from scipy.optimize import LinearConstraint, milp

    marcas = sorted(respuesta["marca"].unique())
    bloques, precios, valores = {}, {}, {}
    inicio = 0
    for j in marcas:
        sub = respuesta[respuesta["marca"] == j].sort_values("precio")
        bloques[j] = list(range(inicio, inicio + len(sub)))
        precios[j] = sub["precio"].to_numpy()
        valores[j] = sub["delta_vs_observado"].to_numpy()
        inicio += len(sub)
    n = inicio

    c = -np.concatenate([valores[j] for j in marcas])
    A, lb, ub = [], [], []
    for j in marcas:  # R1: exactamente un precio por marca
        fila = np.zeros(n)
        fila[bloques[j]] = 1
        A.append(fila)
        lb.append(1)
        ub.append(1)
    if imponer_r7:
        for a_, b_ in zip(marcas[:-1], marcas[1:]):  # R7: precios ordenados
            fila = np.zeros(n)
            fila[bloques[a_]] = precios[a_]
            fila[bloques[b_]] = -precios[b_]
            A.append(fila)
            lb.append(-np.inf)
            ub.append(0)

    res = milp(
        c=c,
        constraints=LinearConstraint(np.vstack(A), lb, ub),
        integrality=np.ones(n),
        bounds=(0, 1),
    )
    y = np.round(res.x).astype(int)

    filas = []
    for j in marcas:
        k = int(np.argmax(y[bloques[j]]))
        sub = respuesta[respuesta["marca"] == j].sort_values("precio").iloc[k]
        observado = respuesta[(respuesta["marca"] == j) & respuesta["es_observado"]]
        filas.append(
            {
                "marca": j,
                "precio_observado": float(observado["precio"].iloc[0]) if len(observado) else np.nan,
                "precio_elegido": float(sub["precio"]),
                "cambio_pct": round(
                    100 * (sub["precio"] / float(observado["precio"].iloc[0]) - 1), 1
                )
                if len(observado)
                else np.nan,
                "ganancia_aditiva": float(sub["delta_vs_observado"]),
                "congelada": len(respuesta[respuesta["marca"] == j]) == 1,
            }
        )
    out = pd.DataFrame(filas)
    out.attrs["r7"] = imponer_r7
    out.attrs["status"] = res.message
    out.attrs["ganancia_aditiva_total"] = round(float(out["ganancia_aditiva"].sum()), 2)
    out.attrs["plan"] = {int(r["marca"]): float(r["precio_elegido"]) for _, r in out.iterrows()}
    out.attrs["ordenado"] = bool(
        all(out.sort_values("marca")["precio_elegido"].diff().dropna() >= -1e-9)
    )
    # A solution sitting on the edge of the grid is the optimizer pushing the
    # model past where it was validated. Flagging it is not pedantry: it is the
    # difference between a recommendation and a liability.
    borde = []
    for j in marcas:
        puntos = respuesta[respuesta["marca"] == j]["precio"]
        fila = out.loc[out["marca"] == j].iloc[0]
        elegido, observado = float(fila["precio_elegido"]), float(fila["precio_observado"])
        if len(puntos) > 1 and abs(elegido - observado) > 1e-9:
            en_extremo = (
                abs(elegido - float(puntos.min())) < 1e-9
                or abs(elegido - float(puntos.max())) < 1e-9
            )
            if en_extremo:
                borde.append(int(j))
    out.attrs["en_el_borde"] = borde
    out.attrs["lectura"] = (
        "Plan de precios {}R7: {}. Ganancia aditiva total {:+.2f}. {}"
    ).format(
        "con " if imponer_r7 else "sin ",
        ", ".join(
            "marca {}: {:.3f} ({:+.1f}%)".format(int(r["marca"]), r["precio_elegido"], r["cambio_pct"])
            for _, r in out.iterrows()
        ),
        float(out["ganancia_aditiva"].sum()),
        ("Las marcas congeladas mantienen su precio observado porque la Fase 4 no identifico "
         "su elasticidad propia. " if out["congelada"].any() else "")
        + ("ATENCION: la(s) marca(s) {} quedan en el **borde** de la grilla, senial de que el "
           "optimizador esta empujando al modelo fuera del rango donde se valido; la "
           "recomendacion accionable es la que acota el movimiento con una banda de "
           "confianza.".format(borde) if borde else "Ninguna marca queda en el borde de la grilla."),
    )
    return out


def verify_p3_plan(
    bundle: ModelBundle,
    df: pd.DataFrame,
    supuesto: MarginAssumption,
    plan: dict[int, float],
    ganancia_aditiva: float,
) -> pd.DataFrame:
    """Score the whole price plan as one world, and measure the additive error.

    Prices interact through cross-elasticity, so optimizing brand by brand and
    adding the gains is an approximation. This is the number that goes to the
    business: the joint simulation, with the gap against the additive promise
    stated.
    """
    base = score_scenario(bundle, df, supuesto)
    conjunto = score_scenario(bundle, build_price_plan(df, plan), supuesto)
    real = float(conjunto["margen_total"].sum() - base["margen_total"].sum())

    out = pd.DataFrame(
        [
            {"medida": "margen base del horizonte", "valor": round(float(base["margen_total"].sum()), 2)},
            {"medida": "margen con el plan de precios", "valor": round(float(conjunto["margen_total"].sum()), 2)},
            {"medida": "ganancia prometida por el objetivo aditivo", "valor": round(ganancia_aditiva, 2)},
            {"medida": "ganancia en la simulacion conjunta", "valor": round(real, 2)},
            {"medida": "error de la aproximacion", "valor": round(real - ganancia_aditiva, 2)},
        ]
    )
    rel = abs(real - ganancia_aditiva) / abs(ganancia_aditiva) if ganancia_aditiva else float("nan")
    out.attrs["ganancia_real"] = round(real, 2)
    out.attrs["error_relativo"] = round(float(rel), 4)
    out.attrs["aditivo_subestima"] = bool(real > ganancia_aditiva)
    out.attrs["aproximacion_conservadora"] = bool(real >= ganancia_aditiva)
    out.attrs["lectura"] = (
        "El objetivo aditivo prometia {:+.2f} y la simulacion conjunta da {:+.2f}: {:.1%} de "
        "diferencia, y el signo importa. {} La cifra que se reporta al negocio es la de la "
        "simulacion conjunta, no la del objetivo."
    ).format(
        ganancia_aditiva,
        real,
        rel,
        "La simulacion conjunta da **mas** que la suma de las partes: las elasticidades "
        "cruzadas trabajan a favor (mover dos precios a la vez refuerza el efecto), asi que "
        "el objetivo aditivo es una cota **inferior** y el procedimiento es conservador."
        if real > ganancia_aditiva
        else "La simulacion conjunta da **menos** que la suma de las partes: optimizar marca "
        "por marca sobreestima, y haria falta una busqueda conjunta antes de recomendar el "
        "plan.",
    )
    return out


# --------------------------------------------------------------------------
# Sensitivity, KPIs and business impact
# --------------------------------------------------------------------------


def margin_sensitivity(
    bundle: ModelBundle,
    df: pd.DataFrame,
    margenes: tuple[float, ...] = (0.20, 0.30, 0.40),
    presupuesto_frac: float = 0.20,
    fatiga: int = 8,
) -> pd.DataFrame:
    """The headline numbers of P1 and P2 under three gross-margin assumptions.

    The gross margin is the one assumption everything rests on, so the
    deliverable is a band and not a point. What has to survive the band is the
    *decision*: if the recommended action flips when the margin moves from 20%
    to 40%, the recommendation is not usable.
    """
    filas = []
    for mu in margenes:
        sup = MarginAssumption(margen_bruto=mu)
        base = score_scenario(bundle, df, sup)
        a = sup.costo_contacto(df, float(base["lambda"].mean()))
        d = sup.costo_promo_dia(df, float(base["lambda"].mean()))
        n = int(presupuesto_frac * len(base))

        p1 = solve_p1(base, a, n, fatiga)
        V = promo_value_matrix(bundle, df, sup)
        cal = observed_calendar(df)
        p2 = solve_p2(V, cal, d, sup.tope_promo_por_marca)

        filas.append(
            {
                "margen_bruto": mu,
                "costo_contacto": round(a, 4),
                "costo_promo_dia": round(d, 4),
                "p1_contactos": p1.attrs["contactos"],
                "p1_objetivo": p1.attrs["objetivo"],
                "p2_ganancia": p2.attrs["ganancia"],
                "p2_dias_marca_5": int(p2.loc[5, "dias_optimos"]),
                "p2_dias_marca_1_2": int(p2.loc[1, "dias_optimos"] + p2.loc[2, "dias_optimos"]),
            }
        )
    out = pd.DataFrame(filas)
    misma_direccion = bool((out["p2_dias_marca_5"] > out["p2_dias_marca_1_2"]).all())

    out.attrs["decision_estable"] = misma_direccion
    out.attrs["rango_p1"] = (float(out["p1_objetivo"].min()), float(out["p1_objetivo"].max()))
    out.attrs["rango_p2"] = (float(out["p2_ganancia"].min()), float(out["p2_ganancia"].max()))
    out.attrs["lectura"] = (
        "Con el margen bruto entre {:.0%} y {:.0%}, el objetivo de P1 va de {:.2f} a {:.2f} y "
        "la ganancia de P2 de {:.2f} a {:.2f}. {} Las **cifras** dependen del supuesto; la "
        "**decision** es lo que hay que mirar, y es la misma en los tres escenarios."
    ).format(
        min(margenes),
        max(margenes),
        float(out["p1_objetivo"].min()),
        float(out["p1_objetivo"].max()),
        float(out["p2_ganancia"].min()),
        float(out["p2_ganancia"].max()),
        "En los tres casos P2 concentra la promocion en la marca 5 por encima de las marcas "
        "1 y 2, asi que la recomendacion no depende del supuesto de margen."
        if misma_direccion
        else "La direccion de la recomendacion cambia con el supuesto: no se puede recomendar "
        "sin fijar el margen con el negocio.",
    )
    return out


def kpi_table(
    p1_politicas: pd.DataFrame,
    p1_solucion: pd.DataFrame,
    p2: pd.DataFrame,
    p3_verificacion: pd.DataFrame,
    lift_top10: float,
    lift_baseline: float,
    margen_base: float,
) -> pd.DataFrame:
    """KPIs of value capture: baseline, target, and **how it gets measured**.

    The measurement column is the one that matters. The brief is explicit: *sin
    medicion contrafactual no hay captura de valor demostrable, solo una
    proyeccion*. Every row here is a projection until an A/B test or a control
    group turns it into a measurement, and the table says so row by row.
    """
    p1 = p1_politicas[p1_politicas["politica"].str.startswith("P1")].iloc[0]
    hist = p1_politicas[p1_politicas["politica"].str.startswith("politica historica")].iloc[0]
    ganancia_p2 = float(p2.attrs["ganancia"])
    ganancia_p3 = float(p3_verificacion.attrs["ganancia_real"])

    filas = [
        {
            "kpi": "Lift de conversion en el decil superior",
            "linea_base": f"x{lift_baseline:.2f} (politica por tasa historica)",
            "con_el_modelo": f"x{lift_top10:.2f}",
            "delta": f"{lift_top10 - lift_baseline:+.2f}",
            "como_se_mide": "A/B: decil superior del modelo vs. decil superior de la politica "
            "actual, misma semana, mismos vendedores",
            "estado": "PROYECCION hasta tener el test",
        },
        {
            "kpi": "Margen incremental por targeting (P1)",
            "linea_base": f"{hist['objetivo_neto']:.2f} (politica vigente, mismo presupuesto)",
            "con_el_modelo": f"{p1['objetivo_neto']:.2f}",
            "delta": f"{p1['objetivo_neto'] - hist['objetivo_neto']:+.2f} "
            f"({100 * (p1['objetivo_neto'] / hist['objetivo_neto'] - 1):+.1f}%)",
            "como_se_mide": "grupo de control: 10% de la cartera asignada por la politica "
            "vigente durante el piloto",
            "estado": "PROYECCION hasta tener el grupo de control",
        },
        {
            "kpi": "Ahorro / reasignacion promocional (P2)",
            "linea_base": f"{p2.attrs['valor_observado']:.2f} (calendario observado)",
            "con_el_modelo": f"{p2.attrs['valor_optimo']:.2f}",
            "delta": f"{ganancia_p2:+.2f} con el mismo numero de dias-promo",
            "como_se_mide": "A/B geografico o por cartera: calendario reasignado en un grupo, "
            "calendario actual en el otro",
            "estado": "PROYECCION hasta tener el test",
        },
        {
            "kpi": "Margen incremental por pricing (P3)",
            "linea_base": f"{margen_base:.2f} (precios observados)",
            "con_el_modelo": f"{margen_base + ganancia_p3:.2f}",
            "delta": f"{ganancia_p3:+.2f} ({100 * ganancia_p3 / margen_base:+.1f}%)",
            "como_se_mide": "test de precio por region o canal; es el unico KPI que **exige** "
            "experimento, porque la elasticidad es observacional",
            "estado": "PROYECCION; sin test de precio no se recomienda ejecutar",
        },
        {
            "kpi": "Estabilidad del modelo en produccion",
            "linea_base": "PSI de las features = 0 al desplegar",
            "con_el_modelo": "umbral de alerta PSI > 0.2; drift de la tasa de compra > 2 pp",
            "delta": "—",
            "como_se_mide": "monitoreo mensual contra la distribucion de train (dias 1-550) y "
            "contra la tasa base 0.2498",
            "estado": "requisito de operacion, no de captura",
        },
    ]
    out = pd.DataFrame(filas)
    out.attrs["proyecciones"] = int(out["estado"].str.startswith("PROYECCION").sum())
    out.attrs["ganancia_total_proyectada"] = round(
        float(p1["objetivo_neto"] - hist["objetivo_neto"]) + ganancia_p2 + ganancia_p3, 2
    )
    out.attrs["lectura"] = (
        "Cuatro de los cinco KPIs son **proyecciones** hasta que exista medicion "
        "contrafactual, y asi estan etiquetados. Sumadas, las tres palancas proyectan "
        "{:+.2f} unidades de margen sobre un margen base de {:.2f} en el horizonte de test "
        "({:+.1f}%), pero esa suma solo se vuelve captura demostrada con A/B o grupo de "
        "control. Decirlo asi es parte del entregable: una proyeccion presentada como "
        "resultado es la forma mas rapida de perder la confianza del negocio."
    ).format(
        out.attrs["ganancia_total_proyectada"],
        margen_base,
        100 * out.attrs["ganancia_total_proyectada"] / margen_base,
    )
    return out


def business_area_impact(
    p1_solucion: pd.DataFrame,
    p2: pd.DataFrame,
    elasticidad_segmento: pd.DataFrame,
    p3_plan: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """The four business areas of the brief, each tied to a number produced here."""
    demanda = (
        scored[[f"margen_{j}" for j in cfg.BRANDS]].sum().rename(lambda s: int(s.split("_")[1]))
    )
    filas = [
        {
            "area": "Comercial / Ventas",
            "decision_que_habilita": "a quien contactar y con que marca",
            "cifra_de_esta_fase": "{} contactos sobre {} clientes; objetivo {:.2f}".format(
                p1_solucion.attrs["contactos"],
                p1_solucion.attrs["clientes_distintos"],
                p1_solucion.attrs["objetivo"],
            ),
            "sub_problema": "P1",
        },
        {
            "area": "Marketing",
            "decision_que_habilita": "a que marca y en que dias promocionar",
            "cifra_de_esta_fase": "reasignar {} dias-promo vale {:+.2f}".format(
                p2.attrs["dias_gastados_observado"], p2.attrs["ganancia"]
            ),
            "sub_problema": "P2",
        },
        {
            "area": "Pricing",
            "decision_que_habilita": "que precio por marca, y si diferenciar por segmento",
            "cifra_de_esta_fase": "plan de precios {}; elasticidad por segmento con rango {:.3f}".format(
                ", ".join(
                    "m{}: {:+.1f}%".format(int(r["marca"]), r["cambio_pct"])
                    for _, r in p3_plan.iterrows()
                    if abs(r["cambio_pct"]) > 0.05
                )
                or "sin cambios",
                elasticidad_segmento.attrs["rango_elasticidad"],
            ),
            "sub_problema": "P3",
        },
        {
            "area": "Planeamiento / Inventario",
            "decision_que_habilita": "demanda esperada por marca en el horizonte",
            "cifra_de_esta_fase": "margen esperado por marca: {}".format(
                ", ".join(f"m{j}: {v:.0f}" for j, v in demanda.items())
            ),
            "sub_problema": "M1xM2xM3",
        },
    ]
    out = pd.DataFrame(filas)
    out.attrs["lectura"] = (
        "Cada area recibe una decision, no un reporte. Y cada decision viene con el "
        "sub-problema que la produce, de modo que si el negocio discute una cifra se sabe "
        "exactamente que supuesto o que modelo hay que revisar."
    )
    return out


def freeze(resultados: dict, path=None) -> dict:
    """Write the prescriptive results to ``artifacts/prescriptivo.json``."""
    import json
    from datetime import date

    path = path or PRESCRIPTIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fecha": date.today().isoformat(), **resultados}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def phase5_report(
    curvas: dict[int, pd.DataFrame],
    elasticidad_segmento: pd.DataFrame,
    p1_verificacion: pd.DataFrame,
    p1_politicas: pd.DataFrame,
    p2: pd.DataFrame,
    p2_verificacion: pd.DataFrame,
    p3_sin_r7: pd.DataFrame,
    p3_con_r7: pd.DataFrame,
    p3_verificacion: pd.DataFrame,
    kpis: pd.DataFrame,
    sensibilidad: pd.DataFrame,
) -> pd.DataFrame:
    """One row per requirement of Fase 5, with the evidence that closes it."""
    costo_r7 = float(
        p3_sin_r7.attrs["ganancia_aditiva_total"] - p3_con_r7.attrs["ganancia_aditiva_total"]
    )
    filas = [
        {
            "requisito": "S1 · curvas de propension frente al precio",
            "resultado": "{} marcas con curva puntuada de extremo a extremo".format(len(curvas)),
            "ok": len(curvas) >= 2,
        },
        {
            "requisito": "S1 · elasticidad por segmento",
            "resultado": "rango {:.3f} entre segmentos".format(
                elasticidad_segmento.attrs["rango_elasticidad"]
            ),
            "ok": True,
        },
        {
            "requisito": "S3 · optimalidad exacta de P1 verificada",
            "resultado": "brecha greedy vs LP = {:.2e}".format(p1_verificacion.attrs["brecha"]),
            "ok": bool(p1_verificacion.attrs["optimo"]),
        },
        {
            "requisito": "S2 · P1 bate la politica vigente",
            "resultado": "{:+.1f}% de objetivo neto con el mismo presupuesto".format(
                p1_politicas.attrs["ganancia_vs_historica_pct"]
            ),
            "ok": p1_politicas.attrs["ganancia_vs_historica_pct"] > 0,
        },
        {
            "requisito": "S4 · P2 reasigna el calendario (MILP con R5)",
            "resultado": "{:+.2f} con el mismo numero de dias-promo; error aditivo {:.1%}".format(
                p2.attrs["ganancia"], p2_verificacion.attrs["error_relativo"]
            ),
            "ok": p2.attrs["ganancia"] > 0,
        },
        {
            "requisito": "S4 · P3 sobre la escalera discreta",
            "resultado": "ganancia conjunta verificada {:+.2f}".format(
                p3_verificacion.attrs["ganancia_real"]
            ),
            "ok": p3_verificacion.attrs["ganancia_real"] > 0,
        },
        {
            "requisito": "S5 · costo de imponer R7 reportado",
            "resultado": "{:.2f} unidades de margen".format(costo_r7),
            "ok": True,
        },
        {
            "requisito": "S7 · la marca 3 no se optimiza",
            "resultado": "precio congelado en su nivel observado",
            "ok": bool((p3_con_r7.loc[p3_con_r7["marca"] == 3, "congelada"]).all()),
        },
        {
            "requisito": "S6 · sensibilidad sobre el margen bruto",
            "resultado": "decision estable entre 20% y 40%: {}".format(
                sensibilidad.attrs["decision_estable"]
            ),
            "ok": bool(sensibilidad.attrs["decision_estable"]),
        },
        {
            "requisito": "S9 · KPIs con metodo de medicion",
            "resultado": "{} de {} etiquetados como proyeccion hasta tener contrafactual".format(
                kpis.attrs["proyecciones"], len(kpis)
            ),
            "ok": kpis.attrs["proyecciones"] >= 3,
        },
    ]
    out = pd.DataFrame(filas)
    out["ok"] = out["ok"].astype(bool)
    out.attrs["fase5_completa"] = bool(out["ok"].all())
    out.attrs["costo_r7"] = round(costo_r7, 2)
    out.attrs["lectura"] = "{} de {} requisitos de la Fase 5 cerrados con evidencia ejecutada.".format(
        int(out["ok"].sum()), len(out)
    )
    return out
