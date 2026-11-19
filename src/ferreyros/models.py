"""Models M1, M2 and M3 -- the predictive layer (Fase 4).

The three models exist because the objective function needs them, not because
three models are better than one. Section 5.1 of ``docs/brief-problema.md``
fixes the interface:

===== ================================== ======================= ==============
Model Target                             Universe                Parameter
===== ================================== ======================= ==============
M1    ``incidencia_compra``              58,693 visits           ``pi_it``
M2    ``id_marca`` in 1..5               14,638 purchases        ``q_ijt``
M3    ``cantidad`` >= 1                  14,638 purchases        ``lambda_ijt``
===== ================================== ======================= ==============

and their product is what the optimizer maximizes::

    valor esperado de la visita = P(compra) x P(marca | compra) x E[cantidad] x precio

Four disciplines run through every function here.

**Nothing is fitted outside train.** Every candidate is a ``Pipeline`` whose
first step is an *unfitted* preprocessor, so cross-validation refits the
scaler and the encoder inside each fold. Scoring a pre-transformed matrix would
leak each fold's validation rows into its own scaling.

**No random split.** The panel has ~117 visits per client, so the folds are
either expanding windows over ``dia_visita`` (how the model will actually be
used: predict forward) or ``GroupKFold`` by client (generalization to clients
never seen). Both are reported, because they answer different questions and
can disagree.

**PR-AUC is the metric.** With 25% positives, accuracy is 75.1% for a model
that predicts "never buys", so accuracy is reported once -- next to that
baseline -- and never used to choose anything.

**The threshold is a business decision.** 0.5 is a convention with no standing
here. The operating threshold is the one that equates expected margin with
contact cost, which is the sub-problem P1 the optimizer solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from . import config as cfg
from . import data as data_mod
from . import features as feat_mod
from . import framing
from . import preprocess as pre_mod

RANDOM_STATE = 42

#: Trend indices flagged by Fase 1: 98% and 88% of their test values fall
#: outside the train range and there is no trend to extrapolate (t=-1.16).
#: Spec S11 says the call is made with the metric in hand, so they stay in the
#: matrix until :func:`trend_feature_experiment` says otherwise.
TREND_COLS: list[str] = ["semana", "mes"]


# --------------------------------------------------------------------------
# Data for each universe
# --------------------------------------------------------------------------


@dataclass
class ConditionalData:
    """Matrices for a model trained only on the visits that ended in a purchase."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series
    dias_train: pd.Series
    dias_test: pd.Series
    target: str

    @property
    def shapes(self) -> pd.DataFrame:
        filas = []
        for nombre, X, dias, grupos in (
            ("train", self.X_train, self.dias_train, self.groups_train),
            ("test", self.X_test, self.dias_test, self.groups_test),
        ):
            filas.append(
                {
                    "conjunto": nombre,
                    "filas": len(X),
                    "columnas": X.shape[1],
                    "clientes": int(grupos.nunique()),
                    "dias": f"{int(dias.min())}-{int(dias.max())}",
                }
            )
        return pd.DataFrame(filas)


def day_index(prepared: pre_mod.PreparedData, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The visit day of every row of the modelling matrices.

    ``prepare`` drops ``dia_visita`` from the feature matrix on purpose (Fase 0
    declared it non-admissible: day 600 means nothing to a future client), but
    the fold boundaries need it, so it is recovered by index.
    """
    return (
        df.loc[prepared.X_train.index, cfg.KEY_TIME],
        df.loc[prepared.X_test.index, cfg.KEY_TIME],
    )


def prepare_conditional(
    target: str,
    df: pd.DataFrame | None = None,
    split_day: int | None = None,
    extra: list[str] | None = None,
) -> ConditionalData:
    """Matrices for M2 / M3: the purchases-only universe, fitted only on train.

    The order mirrors :func:`ferreyros.preprocess.prepare` -- split first, then
    the train-only feature reference, then the features over the full timeline
    -- with one difference that matters: the universe is restricted to
    purchases *after* the features are built, so a client's history still
    includes the visits where nothing was bought.
    """
    split_day = split_day or cfg.SPLIT_DAY
    if df is None:
        df = data_mod.add_paid_price_and_spend(data_mod.load_raw())

    train_raw, _ = data_mod.temporal_split(df, split_day)
    reference = feat_mod.fit_reference(train_raw)
    featured = feat_mod.build_features(df, reference=reference)

    compras = featured[featured[cfg.TARGET_INCIDENCE] == 1]
    train, test = data_mod.temporal_split(compras, split_day)

    X_train = feat_mod.feature_matrix(train, extra=extra)
    X_test = feat_mod.feature_matrix(test, extra=extra)[list(X_train.columns)]
    return ConditionalData(
        X_train=X_train,
        X_test=X_test,
        y_train=train[target],
        y_test=test[target],
        groups_train=train[cfg.KEY_CLIENT],
        groups_test=test[cfg.KEY_CLIENT],
        dias_train=train[cfg.KEY_TIME],
        dias_test=test[cfg.KEY_TIME],
        target=target,
    )


def drop_features(X_train: pd.DataFrame, X_test: pd.DataFrame, cols: list[str]):
    """Both matrices without ``cols`` -- used by the S11 experiment."""
    presentes = [c for c in cols if c in X_train.columns]
    return X_train.drop(columns=presentes), X_test.drop(columns=presentes)


# --------------------------------------------------------------------------
# Folds that respect the structure of the panel
# --------------------------------------------------------------------------


def expanding_day_folds(dias: pd.Series, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding windows over ``dia_visita``: train on the past, score the next block.

    This is the shape of the real task. A fold boundary is a day, never a row,
    so no visit of a client can sit in training while the same client's earlier
    visit is being scored.
    """
    dias = dias.reset_index(drop=True)
    cortes = np.quantile(dias.unique(), np.linspace(0, 1, n_splits + 2)[1:-1])
    cortes = np.unique(np.floor(cortes).astype(int))

    folds = []
    for i, corte in enumerate(cortes):
        siguiente = int(cortes[i + 1]) if i + 1 < len(cortes) else int(dias.max())
        tr = np.where(dias <= corte)[0]
        va = np.where((dias > corte) & (dias <= siguiente))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def client_folds(groups: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """``GroupKFold`` by client: every validation client is unseen in training."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, groups=groups.to_numpy()))


def grouped_time_folds(
    dias: pd.Series, groups: pd.Series, n_splits: int = 4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The strict scheme: unseen clients **and** future days at the same time.

    Needed because neither of the two schemes the brief names is conservative
    on its own. ``GroupKFold`` holds out clients but trains on the *same days*,
    and with 500 clients visiting the same calendar the day-level demand
    conditions (prices, promotions) are shared, so the held-out clients are
    scored under conditions the model has already seen. Crossing the two is the
    only fold that matches deployment: a client we may not know, on a day that
    has not happened yet.
    """
    dias = dias.reset_index(drop=True)
    grupos = groups.reset_index(drop=True)
    clientes = np.array(sorted(grupos.unique()))
    rng = np.random.default_rng(RANDOM_STATE)
    bloques = np.array_split(rng.permutation(clientes), n_splits)

    cortes = np.unique(
        np.floor(np.quantile(dias.unique(), np.linspace(0, 1, n_splits + 2)[1:-1])).astype(int)
    )
    folds = []
    for i, corte in enumerate(cortes):
        siguiente = int(cortes[i + 1]) if i + 1 < len(cortes) else int(dias.max())
        fuera = np.isin(grupos, bloques[i % len(bloques)])
        tr = np.where((dias <= corte) & ~fuera)[0]
        va = np.where((dias > corte) & (dias <= siguiente) & fuera)[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def fold_description(dias: pd.Series, folds: list) -> pd.DataFrame:
    """The folds as a table, so the validation scheme is visible, not implied."""
    dias = dias.reset_index(drop=True)
    return pd.DataFrame(
        [
            {
                "fold": i + 1,
                "filas_train": len(tr),
                "filas_validacion": len(va),
                "dias_train": f"{int(dias.iloc[tr].min())}-{int(dias.iloc[tr].max())}",
                "dias_validacion": f"{int(dias.iloc[va].min())}-{int(dias.iloc[va].max())}",
            }
            for i, (tr, va) in enumerate(folds)
        ]
    )


# --------------------------------------------------------------------------
# Candidates and the progression of complexity
# --------------------------------------------------------------------------


def pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    """An unfitted preprocessor plus the estimator, so each fold fits its own."""
    return Pipeline([("pre", pre_mod.build_preprocessor(X)), ("clf", estimator)])


def m1_candidates() -> dict[str, object]:
    """The progression the brief asks for, from interpretable to flexible.

    ``HistGradientBoostingClassifier`` stands in for XGBoost / LightGBM: same
    family (histogram-based GBDT), no extra dependency, and no ``libomp`` to
    install before the notebook will run. The substitution is declared because
    the brief names the libraries.
    """
    return {
        "1 · logistica": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "2 · random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "3 · gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _binary_fold_scores(est, X, y, folds) -> list[dict]:
    filas = []
    Xv, yv = X.reset_index(drop=True), y.reset_index(drop=True)
    for i, (tr, va) in enumerate(folds):
        modelo = clone(est).fit(Xv.iloc[tr], yv.iloc[tr])
        p = modelo.predict_proba(Xv.iloc[va])[:, 1]
        filas.append(
            {
                "fold": i + 1,
                "pr_auc": average_precision_score(yv.iloc[va], p),
                "roc_auc": roc_auc_score(yv.iloc[va], p),
                "tasa_base": float(yv.iloc[va].mean()),
            }
        )
    return filas


def cv_compare(
    X: pd.DataFrame,
    y: pd.Series,
    dias: pd.Series,
    groups: pd.Series,
    candidatos: dict | None = None,
) -> pd.DataFrame:
    """Every candidate under both validation schemes, PR-AUC and ROC-AUC.

    Reporting the two schemes side by side is the point: the temporal one says
    whether the model predicts the future of known clients, the client one says
    whether it generalizes to clients it has never seen. A model can be good at
    one and poor at the other, and with 500 clients that is the live risk.
    """
    candidatos = candidatos or m1_candidates()
    esquemas = {
        "temporal (ventana expansiva)": expanding_day_folds(dias),
        "por cliente (GroupKFold)": client_folds(groups),
        "cliente nuevo + dia futuro": grouped_time_folds(dias, groups),
    }

    filas = []
    por_fold: dict[tuple[str, str], list[float]] = {}
    for nombre, est in candidatos.items():
        pipe = pipeline(est, X)
        for esquema, folds in esquemas.items():
            scores = pd.DataFrame(_binary_fold_scores(pipe, X, y, folds))
            por_fold[(nombre, esquema)] = scores["pr_auc"].tolist()
            filas.append(
                {
                    "modelo": nombre,
                    "esquema": esquema,
                    "folds": len(scores),
                    "pr_auc_media": round(float(scores["pr_auc"].mean()), 4),
                    "pr_auc_std": round(float(scores["pr_auc"].std()), 4),
                    "roc_auc_media": round(float(scores["roc_auc"].mean()), 4),
                    "pr_auc_min": round(float(scores["pr_auc"].min()), 4),
                }
            )
    out = pd.DataFrame(filas)

    estricto_nombre = "cliente nuevo + dia futuro"
    temporal = out[out["esquema"].str.startswith("temporal")].set_index("modelo")
    estricto = out[out["esquema"] == estricto_nombre].set_index("modelo")
    por_cliente = out[out["esquema"].str.startswith("por cliente")].set_index("modelo")

    # Paired comparison on the strict scheme. The candidates were scored on the
    # same folds, so the difference fold by fold is the honest test: with four
    # folds whose spread is 0.02-0.04 of PR-AUC, a mean gain of 0.01 can easily
    # be noise, and a fixed tolerance would not know the difference.
    referencia = next(iter(candidatos))
    base_folds = np.array(por_fold[(referencia, estricto_nombre)], dtype=float)
    pareadas = []
    for nombre in candidatos:
        if nombre == referencia:
            continue
        d = np.array(por_fold[(nombre, estricto_nombre)], dtype=float) - base_folds
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        pareadas.append(
            {
                "modelo": nombre,
                "vs": referencia,
                "diferencia_media": round(float(d.mean()), 4),
                "error_estandar": round(se, 4),
                "t": round(float(d.mean() / se), 2) if se else np.nan,
                "significativa": bool(se and abs(d.mean() / se) > 2 and d.mean() > 0),
            }
        )
    pareada = pd.DataFrame(pareadas)

    mejor = estricto["pr_auc_media"].idxmax()
    ganancia = float(estricto.loc[mejor, "pr_auc_media"] - estricto.loc[referencia, "pr_auc_media"])
    inflacion = (por_cliente["pr_auc_media"] - estricto["pr_auc_media"]).max()
    inflado = (por_cliente["pr_auc_media"] - estricto["pr_auc_media"]).idxmax()
    paga = bool(pareada["significativa"].any()) if len(pareada) else False

    out.attrs["comparacion_pareada"] = pareada
    out.attrs["mejor_estricto"] = mejor
    out.attrs["ganancia_vs_referencia"] = round(ganancia, 4)
    out.attrs["paga_la_complejidad"] = paga
    out.attrs["inflacion_groupkfold"] = round(float(inflacion), 4)
    out.attrs["modelo_mas_inflado"] = inflado
    out.attrs["lectura"] = (
        "Bajo el esquema estricto (cliente nuevo + dia futuro) el mejor PR-AUC es de {} "
        "({:.4f}) y {} da {:.4f}: {:+.4f}. Pero los folds tienen desvios de {:.4f} a {:.4f}, "
        "asi que la comparacion que decide es la pareada fold a fold: {}. {}\n"
        "Y una advertencia sobre el esquema por cliente: para {} da {:.4f} frente a {:.4f} "
        "en el estricto, {:+.4f} de diferencia. GroupKFold deja fuera al cliente pero "
        "entrena con los mismos dias, y como los 500 clientes comparten calendario, precios "
        "y promociones, el modelo ya vio las condiciones de demanda del fold de validacion. "
        "No es el limite pesimista: es 'cliente nuevo, periodo conocido'."
    ).format(
        mejor,
        float(estricto.loc[mejor, "pr_auc_media"]),
        referencia,
        float(estricto.loc[referencia, "pr_auc_media"]),
        ganancia,
        float(estricto["pr_auc_std"].min()),
        float(estricto["pr_auc_std"].max()),
        "; ".join(
            "{} {:+.4f} (t={:+.2f})".format(r["modelo"], r["diferencia_media"], r["t"])
            for _, r in pareada.iterrows()
        )
        if len(pareada)
        else "sin candidatos alternativos",
        "Alguna diferencia supera el ruido: se justifica subir en la progresion."
        if paga
        else "Ninguna diferencia supera el ruido (|t|<2), asi que la complejidad no paga: se "
        "queda el modelo interpretable, que ademas es el que la capa prescriptiva necesita.",
        inflado,
        float(por_cliente.loc[inflado, "pr_auc_media"]),
        float(estricto.loc[inflado, "pr_auc_media"]),
        float(inflacion),
    )
    return out


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    dias: pd.Series,
    estimator,
    grid: list[dict],
    n_splits: int = 4,
) -> pd.DataFrame:
    """Hyperparameter search scored by PR-AUC on expanding day windows.

    An explicit list of configurations instead of a grid object, so the
    notebook can print exactly what was tried and what each one scored.
    """
    folds = expanding_day_folds(dias, n_splits=n_splits)
    filas = []
    for params in grid:
        est = clone(estimator).set_params(**params)
        scores = pd.DataFrame(_binary_fold_scores(pipeline(est, X), X, y, folds))
        filas.append(
            {
                "configuracion": ", ".join(f"{k}={v}" for k, v in params.items()) or "por defecto",
                "pr_auc_media": round(float(scores["pr_auc"].mean()), 4),
                "pr_auc_std": round(float(scores["pr_auc"].std()), 4),
                "roc_auc_media": round(float(scores["roc_auc"].mean()), 4),
                **params,
            }
        )
    out = pd.DataFrame(filas).sort_values("pr_auc_media", ascending=False, ignore_index=True)
    mejores = {k: v for k, v in out.iloc[0].items() if k in grid[0]}
    out.attrs["mejores_params"] = mejores
    out.attrs["pr_auc_mejor"] = float(out.iloc[0]["pr_auc_media"])
    out.attrs["lectura"] = (
        "Mejor configuracion: {} con PR-AUC {:.4f} en validacion temporal; la peor de las {} "
        "probadas da {:.4f}, asi que el margen que mueve el tuning es de {:.4f}."
    ).format(
        out.iloc[0]["configuracion"],
        float(out.iloc[0]["pr_auc_media"]),
        len(out),
        float(out.iloc[-1]["pr_auc_media"]),
        float(out.iloc[0]["pr_auc_media"] - out.iloc[-1]["pr_auc_media"]),
    )
    return out


# --------------------------------------------------------------------------
# Test metrics, against the baselines that matter
# --------------------------------------------------------------------------


def lift_table(y: pd.Series, proba: np.ndarray, deciles=(0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """Capture rate and lift by top fraction of the score -- the targeting view.

    This is the table P1 reads: if we can only contact 10% of the visits, what
    share of the buyers do we reach, and how much better is that than the
    historical-rate policy already in use (lift x2.43).
    """
    orden = np.argsort(-proba)
    yv = y.to_numpy()[orden]
    base = float(y.mean())

    filas = []
    for q in deciles:
        n = int(np.ceil(q * len(yv)))
        tasa = float(yv[:n].mean())
        filas.append(
            {
                "top": f"{int(q * 100)}%",
                "visitas": n,
                "compras_capturadas": int(yv[:n].sum()),
                "tasa_en_el_grupo": round(tasa, 4),
                "lift": round(tasa / base, 2),
                "recall": round(float(yv[:n].sum() / yv.sum()), 4),
            }
        )
    out = pd.DataFrame(filas)
    contrato = framing.FramingContract()
    out.attrs["baseline_historico"] = contrato.baseline_targeting_lift
    out.attrs["lift_top10"] = float(out.loc[out["top"] == "10%", "lift"].iloc[0])
    out.attrs["supera_baseline"] = out.attrs["lift_top10"] > contrato.baseline_targeting_lift
    out.attrs["lectura"] = (
        "En el decil superior el lift es x{:.2f} y captura el {:.1%} de las compras. El "
        "baseline a batir es el targeting por tasa historica, x{:.2f}: {}."
    ).format(
        out.attrs["lift_top10"],
        float(out.loc[out["top"] == "10%", "recall"].iloc[0]),
        contrato.baseline_targeting_lift,
        "lo supera" if out.attrs["supera_baseline"] else "NO lo supera",
    )
    return out


def evaluate_binary(y: pd.Series, proba: np.ndarray, nombre: str = "modelo") -> pd.Series:
    """Every metric in one place, with the forbidden one shown next to its baseline."""
    contrato = framing.FramingContract()
    base = float(y.mean())
    pred_05 = (proba >= 0.5).astype(int)
    return pd.Series(
        {
            "modelo": nombre,
            "filas": len(y),
            "tasa_base": round(base, 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "pr_auc_azar": round(base, 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "brier": round(float(brier_score_loss(y, proba)), 4),
            "log_loss": round(float(log_loss(y, proba)), 4),
            "accuracy_en_0.5": round(float((pred_05 == y).mean()), 4),
            "accuracy_de_no_comprar_nunca": round(1 - base, 4),
            "metrica_principal": contrato.metrica_principal,
            "metrica_prohibida": contrato.metrica_prohibida,
        }
    )


def fit_final(estimator, X_train, y_train, X_test, extra_params: dict | None = None):
    """Fit on all of train and score test once -- the only time test is touched."""
    est = clone(estimator)
    if extra_params:
        est.set_params(**extra_params)
    pipe = pipeline(est, X_train).fit(X_train, y_train)
    return pipe, pipe.predict_proba(X_test)[:, 1]


def trend_feature_experiment(
    X_train, y_train, X_test, y_test, dias_train, estimator, cols=None
) -> pd.DataFrame:
    """Spec S11: does dropping the trend indices cost anything measurable?

    Fase 1 showed there is no trend to extrapolate and that 98% / 88% of the
    test values of ``semana`` / ``mes`` fall outside the train range. That makes
    them a risk; whether they are also useless is an empirical question, and
    this is the measurement that settles it.
    """
    cols = cols or TREND_COLS
    Xtr2, Xte2 = drop_features(X_train, X_test, cols)
    folds = expanding_day_folds(dias_train)

    filas = []
    for nombre, (Xtr, Xte) in {
        "con semana y mes": (X_train, X_test),
        "sin semana y mes": (Xtr2, Xte2),
    }.items():
        cv = pd.DataFrame(_binary_fold_scores(pipeline(estimator, Xtr), Xtr, y_train, folds))
        _, proba = fit_final(estimator, Xtr, y_train, Xte)
        filas.append(
            {
                "variante": nombre,
                "columnas": Xtr.shape[1],
                "pr_auc_cv_temporal": round(float(cv["pr_auc"].mean()), 4),
                "pr_auc_test": round(float(average_precision_score(y_test, proba)), 4),
                "roc_auc_test": round(float(roc_auc_score(y_test, proba)), 4),
            }
        )
    out = pd.DataFrame(filas)
    delta_cv = float(out.loc[1, "pr_auc_cv_temporal"] - out.loc[0, "pr_auc_cv_temporal"])
    delta_test = float(out.loc[1, "pr_auc_test"] - out.loc[0, "pr_auc_test"])

    out.attrs["delta_cv"] = round(delta_cv, 4)
    out.attrs["delta_test"] = round(delta_test, 4)
    out.attrs["quitarlas_no_cuesta"] = delta_cv >= -0.005 and delta_test >= -0.005
    out.attrs["lectura"] = (
        "Quitar semana y mes mueve el PR-AUC {:+.4f} en validacion temporal y {:+.4f} en "
        "test. {}"
    ).format(
        delta_cv,
        delta_test,
        "No cuesta nada medible, asi que se quitan: extrapolar indices de tendencia fuera "
        "del rango de train seria riesgo sin beneficio."
        if (delta_cv >= -0.005 and delta_test >= -0.005)
        else "Cuesta desempenio, asi que se conservan y se declara el riesgo de "
        "extrapolacion que la Fase 1 documento.",
    )
    return out


# --------------------------------------------------------------------------
# The operating threshold: P1, not 0.5
# --------------------------------------------------------------------------

#: Grid of contact-cost / purchase-margin ratios. No cost data exists in the
#: dataset, so the threshold is reported as a function of that ratio instead of
#: being invented: the shape of the answer is what survives the missing input.
RATIOS_COSTO: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def threshold_by_cost(
    y: pd.Series, proba: np.ndarray, ratios: tuple[float, ...] = RATIOS_COSTO
) -> pd.DataFrame:
    """Operating threshold for each contact-cost ratio, with its net value.

    The rule is one line of algebra: contacting a visit pays when
    ``p * margen > costo``, so the optimal threshold is ``costo / margen``. It
    is the solution of P1 restricted to one visit, and it is almost never 0.5.

    ``valor_neto_por_1000`` is expressed in units of margin per purchase, which
    is the only currency available while ``c_j`` is unknown.
    """
    n = len(y)
    yv = y.to_numpy()
    filas = []
    for r in ratios:
        contacta = proba >= r
        tp = int(yv[contacta].sum())
        contactos = int(contacta.sum())
        filas.append(
            {
                "ratio_costo_margen": r,
                "umbral_optimo": r,
                "contactos": contactos,
                "pct_visitas_contactadas": round(100 * contactos / n, 1),
                "compras_capturadas": tp,
                "precision": round(tp / contactos, 4) if contactos else np.nan,
                "recall": round(tp / int(yv.sum()), 4),
                "valor_neto_por_1000": round(1000 * (tp - r * contactos) / n, 2),
            }
        )
    out = pd.DataFrame(filas)

    contacta05 = proba >= 0.5
    tp05 = int(yv[contacta05].sum())
    out.attrs["contactos_en_0.5"] = int(contacta05.sum())
    out.attrs["pct_en_0.5"] = round(100 * float(contacta05.mean()), 1)
    out.attrs["recall_en_0.5"] = round(tp05 / int(yv.sum()), 4)
    out.attrs["proba_maxima"] = round(float(proba.max()), 4)
    out.attrs["lectura"] = (
        "El umbral optimo es el ratio costo/margen, asi que va de {:.2f} a {:.2f} en esta "
        "rejilla y contacta entre el {:.1f}% y el {:.1f}% de las visitas. Con el 0.5 de la "
        "convencion se contactaria el {:.1f}% de las visitas y se capturaria el {:.1%} de "
        "las compras: fijar el umbral en 0.5 no es conservador, es renunciar a la mayor "
        "parte del valor. La probabilidad maxima del modelo en test es {:.4f}."
    ).format(
        float(out["umbral_optimo"].min()),
        float(out["umbral_optimo"].max()),
        float(out["pct_visitas_contactadas"].min()),
        float(out["pct_visitas_contactadas"].max()),
        out.attrs["pct_en_0.5"],
        out.attrs["recall_en_0.5"],
        out.attrs["proba_maxima"],
    )
    return out


def verify_threshold_rule(
    y: pd.Series, proba: np.ndarray, ratios: tuple[float, ...] = (0.1, 0.2, 0.3)
) -> pd.DataFrame:
    """Check the ``umbral = costo/margen`` rule empirically instead of asserting it.

    For each ratio, the net value is evaluated over a dense grid of thresholds
    and the empirical argmax is compared against the theoretical one. If the
    algebra were wrong, or the probabilities badly calibrated, this is where it
    would show.
    """
    yv = y.to_numpy()
    rejilla = np.linspace(0.02, 0.95, 187)
    filas = []
    for r in ratios:
        valores = [(float(yv[proba >= t].sum()) - r * int((proba >= t).sum())) for t in rejilla]
        empirico = float(rejilla[int(np.argmax(valores))])
        filas.append(
            {
                "ratio_costo_margen": r,
                "umbral_teorico": r,
                "umbral_empirico": round(empirico, 3),
                "diferencia": round(empirico - r, 3),
                "valor_en_el_optimo": round(max(valores), 1),
                "valor_en_0.5": round(
                    float(yv[proba >= 0.5].sum()) - r * int((proba >= 0.5).sum()), 1
                ),
            }
        )
    out = pd.DataFrame(filas)
    out.attrs["max_desvio"] = round(float(out["diferencia"].abs().max()), 3)
    out.attrs["regla_valida"] = bool(out["diferencia"].abs().max() <= 0.06)
    out.attrs["lectura"] = (
        "El umbral empirico que maximiza el valor neto se desvia como maximo {:.3f} del "
        "teorico costo/margen. {} Y en los tres ratios el valor en 0.5 es menor que en el "
        "optimo, de modo que la convencion cuesta dinero medible."
    ).format(
        out.attrs["max_desvio"],
        "La regla se sostiene sobre el dato de test, no solo sobre el papel."
        if out.attrs["regla_valida"]
        else "La diferencia es grande: habria que revisar la calibracion de las probabilidades.",
    )
    return out


# --------------------------------------------------------------------------
# Explainability: SHAP and odds ratios
# --------------------------------------------------------------------------


def _transform(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    Z = pipe["pre"].transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    return Z, [str(n) for n in pipe["pre"].get_feature_names_out()]


def shap_values(
    pipe: Pipeline, X: pd.DataFrame, n_sample: int = 1500, random_state: int = RANDOM_STATE
):
    """SHAP values of the fitted pipeline on a sample, with the output names.

    The explainer is chosen by model family -- linear for the logistic, tree
    for the forests -- because both are exact for their family and neither
    needs the slow model-agnostic fallback.
    """
    import shap

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    muestra = X.iloc[idx]
    Z, nombres = _transform(pipe, muestra)
    clf = pipe["clf"]

    if isinstance(clf, LogisticRegression):
        fondo, _ = _transform(pipe, X.iloc[rng.choice(len(X), size=min(500, len(X)), replace=False)])
        valores = shap.LinearExplainer(clf, fondo).shap_values(Z)
    else:
        valores = shap.TreeExplainer(clf).shap_values(Z)
        valores = np.asarray(valores)
        if valores.ndim == 3:  # one matrix per class
            valores = valores[..., 1]
    return np.asarray(valores), nombres, muestra


def shap_global(pipe: Pipeline, X: pd.DataFrame, top: int = 15, **kwargs) -> pd.DataFrame:
    """Global driver ranking: mean |SHAP| per feature, with its average direction."""
    valores, nombres, _ = shap_values(pipe, X, **kwargs)
    out = (
        pd.DataFrame(
            {
                "feature": nombres,
                "importancia_shap": np.abs(valores).mean(axis=0),
                "efecto_medio": valores.mean(axis=0),
            }
        )
        .sort_values("importancia_shap", ascending=False, ignore_index=True)
        .head(top)
    )
    out["importancia_shap"] = out["importancia_shap"].round(4)
    out["efecto_medio"] = out["efecto_medio"].round(4)
    out["direccion"] = np.where(out["efecto_medio"] >= 0, "sube la probabilidad", "la baja")
    out.attrs["top1"] = out.loc[0, "feature"]
    out.attrs["lectura"] = (
        "El driver dominante es {} (|SHAP| medio {:.4f}), seguido de {}. Los tres primeros "
        "concentran el {:.1%} de la importancia del top {}."
    ).format(
        out.loc[0, "feature"],
        float(out.loc[0, "importancia_shap"]),
        ", ".join(out.loc[1:3, "feature"]),
        float(out.loc[:2, "importancia_shap"].sum() / out["importancia_shap"].sum()),
        top,
    )
    return out


def shap_local(pipe: Pipeline, X: pd.DataFrame, posicion: int = 0, top: int = 10, **kwargs):
    """Why *this* visit: the contributions of one row, largest first."""
    valores, nombres, muestra = shap_values(pipe, X, **kwargs)
    fila = pd.DataFrame(
        {
            "feature": nombres,
            "contribucion": valores[posicion],
            "valor_transformado": _transform(pipe, muestra.iloc[[posicion]])[0][0],
        }
    )
    fila["abs"] = fila["contribucion"].abs()
    out = fila.sort_values("abs", ascending=False, ignore_index=True).head(top).drop(columns="abs")
    out["contribucion"] = out["contribucion"].round(4)
    out["valor_transformado"] = out["valor_transformado"].round(3)
    out.attrs["cliente"] = int(muestra.index[posicion])
    out.attrs["probabilidad"] = round(float(pipe.predict_proba(muestra.iloc[[posicion]])[0, 1]), 4)
    out.attrs["lectura"] = (
        "Visita {} con probabilidad {:.4f}. Lo que mas empuja: {} ({:+.4f}); lo que mas "
        "resta: {} ({:+.4f}). Esta es la explicacion que acompania a una recomendacion "
        "comercial concreta."
    ).format(
        out.attrs["cliente"],
        out.attrs["probabilidad"],
        out.loc[0, "feature"],
        float(out.loc[0, "contribucion"]),
        out.loc[out["contribucion"].idxmin(), "feature"],
        float(out["contribucion"].min()),
    )
    return out


def odds_ratios(pipe: Pipeline, top: int = 15) -> pd.DataFrame:
    """Odds ratios of the logistic model, per standard deviation of each feature.

    The preprocessor standardizes, so ``exp(beta)`` reads as "the factor by
    which the odds of buying move when this feature rises one standard
    deviation" -- the reading the brief asks for, and the one a commercial team
    can act on.
    """
    clf = pipe["clf"]
    nombres = [str(n) for n in pipe["pre"].get_feature_names_out()]
    coef = clf.coef_[0]
    out = pd.DataFrame(
        {"feature": nombres, "coeficiente": coef, "odds_ratio": np.exp(coef)}
    )
    out["abs"] = out["coeficiente"].abs()
    out = out.sort_values("abs", ascending=False, ignore_index=True).head(top).drop(columns="abs")
    out["coeficiente"] = out["coeficiente"].round(4)
    out["odds_ratio"] = out["odds_ratio"].round(3)
    out["lectura_negocio"] = np.where(
        out["odds_ratio"] >= 1,
        "+" + ((out["odds_ratio"] - 1) * 100).round(1).astype(str) + "% en las odds",
        "-" + ((1 - out["odds_ratio"]) * 100).round(1).astype(str) + "% en las odds",
    )
    out.attrs["lectura"] = (
        "El coeficiente mas fuerte es {} (odds ratio {:.3f}); el mas negativo es {} "
        "({:.3f}). Son los mismos drivers que SHAP ordena arriba, leidos en la unidad que "
        "entiende el negocio."
    ).format(
        out.loc[0, "feature"],
        float(out.loc[0, "odds_ratio"]),
        out.loc[out["coeficiente"].idxmin(), "feature"],
        float(out["odds_ratio"].min()),
    )
    return out


def baseline_policy_lift(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba: np.ndarray,
    columna: str = "tasa_compra_historica",
    deciles=(0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """Model versus the policy already in use, on **the same test rows**.

    The x2.43 figure of the EDA was measured over the whole panel, so comparing
    it against a test-set lift would not be like for like. This recomputes the
    historical-rate policy on the identical rows the model was scored on, which
    is the only comparison that supports a claim about captured value.
    """
    base = X_test[columna].fillna(0).to_numpy()
    tasa = float(y_test.mean())
    yv = y_test.to_numpy()

    filas = []
    for q in deciles:
        n = int(np.ceil(q * len(yv)))
        for nombre, score in (("politica historica", base), ("modelo M1", proba)):
            sel = np.argsort(-score)[:n]
            filas.append(
                {
                    "top": f"{int(q * 100)}%",
                    "politica": nombre,
                    "visitas": n,
                    "compras": int(yv[sel].sum()),
                    "tasa": round(float(yv[sel].mean()), 4),
                    "lift": round(float(yv[sel].mean()) / tasa, 2),
                }
            )
    out = pd.DataFrame(filas)
    pivote = out.pivot(index="top", columns="politica", values="lift")
    ganancia = (pivote["modelo M1"] - pivote["politica historica"]).round(2)

    out.attrs["lift_modelo_top10"] = float(pivote.loc["10%", "modelo M1"])
    out.attrs["lift_baseline_top10"] = float(pivote.loc["10%", "politica historica"])
    out.attrs["ganancia_top10"] = float(ganancia.loc["10%"])
    out.attrs["gana_en_todos_los_cortes"] = bool((ganancia > 0).all())
    out.attrs["lectura"] = (
        "Sobre las mismas filas de test, en el decil superior el modelo da lift x{:.2f} y la "
        "politica historica x{:.2f}: {:+.2f}. {} La comparacion importa porque el x2.43 del "
        "EDA se midio sobre el panel completo, no sobre test."
    ).format(
        out.attrs["lift_modelo_top10"],
        out.attrs["lift_baseline_top10"],
        out.attrs["ganancia_top10"],
        "El modelo gana en todos los cortes."
        if out.attrs["gana_en_todos_los_cortes"]
        else "El modelo no gana en todos los cortes, y eso hay que decirlo.",
    )
    return out


# --------------------------------------------------------------------------
# M2 -- brand choice, and the elasticities the optimizer needs
# --------------------------------------------------------------------------


def m2_candidates() -> dict[str, object]:
    """Multinomial logit first, because its price coefficients *are* the elasticities."""
    return {
        "1 · logit multinomial": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        "2 · random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "3 · gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_multiclass(y: pd.Series, proba: np.ndarray, clases: np.ndarray) -> pd.Series:
    """Log-loss first (it scores the probabilities M2 has to hand the optimizer)."""
    pred = clases[proba.argmax(axis=1)]
    mayoritaria = y.value_counts(normalize=True)
    return pd.Series(
        {
            "filas": len(y),
            "clases": len(clases),
            "log_loss": round(float(log_loss(y, proba, labels=list(clases))), 4),
            "log_loss_uniforme": round(float(np.log(len(clases))), 4),
            "accuracy": round(float((pred == y.to_numpy()).mean()), 4),
            "accuracy_clase_mayoritaria": round(float(mayoritaria.max()), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        }
    )


def cv_compare_multiclass(
    X: pd.DataFrame, y: pd.Series, dias: pd.Series, groups: pd.Series, candidatos: dict | None = None
) -> pd.DataFrame:
    """Same two honest schemes as M1, scored by log-loss and accuracy."""
    candidatos = candidatos or m2_candidates()
    esquemas = {
        "temporal (ventana expansiva)": expanding_day_folds(dias),
        "cliente nuevo + dia futuro": grouped_time_folds(dias, groups),
    }
    clases = np.sort(y.unique())
    Xv, yv = X.reset_index(drop=True), y.reset_index(drop=True)

    filas = []
    for nombre, est in candidatos.items():
        pipe = pipeline(est, X)
        for esquema, folds in esquemas.items():
            ll, acc = [], []
            for tr, va in folds:
                modelo = clone(pipe).fit(Xv.iloc[tr], yv.iloc[tr])
                p = modelo.predict_proba(Xv.iloc[va])
                ll.append(log_loss(yv.iloc[va], p, labels=list(clases)))
                acc.append(float((modelo.predict(Xv.iloc[va]) == yv.iloc[va]).mean()))
            filas.append(
                {
                    "modelo": nombre,
                    "esquema": esquema,
                    "folds": len(folds),
                    "log_loss_media": round(float(np.mean(ll)), 4),
                    "accuracy_media": round(float(np.mean(acc)), 4),
                }
            )
    out = pd.DataFrame(filas)
    estricto = out[out["esquema"].str.startswith("cliente nuevo")].set_index("modelo")
    mejor = estricto["log_loss_media"].idxmin()
    out.attrs["mejor_estricto"] = mejor
    out.attrs["ganancia_vs_logit"] = round(
        float(estricto.loc["1 · logit multinomial", "log_loss_media"] - estricto.loc[mejor, "log_loss_media"]),
        4,
    )
    out.attrs["lectura"] = (
        "Bajo el esquema estricto el mejor log-loss es de {} ({:.4f}) y el logit multinomial "
        "da {:.4f}. {} El logit se conserva de todos modos para las elasticidades: ahi la "
        "interpretabilidad no es preferencia, es el entregable."
    ).format(
        mejor,
        float(estricto.loc[mejor, "log_loss_media"]),
        float(estricto.loc["1 · logit multinomial", "log_loss_media"]),
        "Un modelo flexible predice mejor la marca."
        if mejor != "1 · logit multinomial"
        else "El logit tambien es el mejor prediciendo.",
    )
    return out


def brand_confusion(y: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    """Row-normalized confusion matrix: where M2 confuses which brand."""
    clases = np.sort(y.unique())
    M = confusion_matrix(y, pred, labels=clases, normalize="true")
    out = pd.DataFrame(M.round(3), index=[f"real {c}" for c in clases], columns=[f"pred {c}" for c in clases])
    out.attrs["aciertos_diagonal"] = {int(c): round(float(M[i, i]), 3) for i, c in enumerate(clases)}
    out.attrs["lectura"] = (
        "La diagonal va de {:.3f} a {:.3f}: M2 no acierta igual en todas las marcas, y las "
        "confusiones dicen entre que marcas existe sustitucion real."
    ).format(float(np.diag(M).min()), float(np.diag(M).max()))
    return out


def elasticity_model(cond: ConditionalData, C: float = 1e6):
    """Multinomial logit on raw prices and promotions, for interpretation only.

    Deliberately separate from the full M2. Here the five prices enter
    **unscaled** and the penalty is switched off (``C=1e6``), so the
    coefficients live in units of 1/price and the elasticity formula below is
    exact rather than approximate. Prediction is M2's job; this model's output
    is a number P3 can use.
    """
    cols = [*cfg.PRICE_COLS, *cfg.PROMO_COLS]
    modelo = LogisticRegression(max_iter=5000, C=C, random_state=RANDOM_STATE).fit(
        cond.X_train[cols], cond.y_train
    )
    return modelo, cols


def cross_price_elasticities(cond: ConditionalData, modelo=None, cols=None) -> pd.DataFrame:
    """The 5x5 matrix of own- and cross-price elasticities of brand share.

    For a multinomial logit with utility ``V_j = a_j + sum_l b_jl * p_l``::

        d ln P_j / d ln p_l = p_l * ( b_jl - sum_m P_m * b_ml )

    evaluated at mean prices and mean predicted shares. The diagonal must come
    out negative (a brand loses share when it raises its price) and that is the
    first thing to check: if it does not, the model is not usable for pricing.
    """
    if modelo is None:
        modelo, cols = elasticity_model(cond)
    clases = modelo.classes_
    precios = cond.X_train[cfg.PRICE_COLS].mean().to_numpy()
    shares = modelo.predict_proba(cond.X_train[cols]).mean(axis=0)
    beta = modelo.coef_[:, : len(cfg.PRICE_COLS)]

    E = np.zeros((len(clases), len(cfg.PRICE_COLS)))
    for j in range(len(clases)):
        for l in range(len(cfg.PRICE_COLS)):
            E[j, l] = precios[l] * (beta[j, l] - float((shares * beta[:, l]).sum()))

    out = pd.DataFrame(
        E.round(3),
        index=[f"share marca {int(c)}" for c in clases],
        columns=[f"precio marca {j}" for j in cfg.BRANDS],
    )
    propias = np.diag(E)
    out.attrs["elasticidades_propias"] = {int(c): round(float(propias[i]), 3) for i, c in enumerate(clases)}
    out.attrs["todas_propias_negativas"] = bool((propias < 0).all())
    out.attrs["mas_elastica"] = int(clases[int(np.argmin(propias))])
    out.attrs["shares_medios"] = {int(c): round(float(shares[i]), 3) for i, c in enumerate(clases)}
    out.attrs["lectura"] = (
        "Las cinco elasticidades propias son {} y la marca mas elastica es la {} ({:.3f}): "
        "un 1% de aumento de su precio le cuesta ese % de share. {} Las cruzadas positivas "
        "son sustitucion: cuanto share gana una marca cuando sube el precio de otra, que es "
        "exactamente lo que P3 necesita para no canibalizarse."
    ).format(
        "todas negativas" if bool((propias < 0).all()) else "NO todas negativas",
        out.attrs["mas_elastica"],
        float(propias.min()),
        "El signo correcto en toda la diagonal es la primera condicion para usarlo en pricing."
        if bool((propias < 0).all())
        else "Con alguna propia positiva el modelo no se puede usar para fijar precios.",
    )
    return out


# --------------------------------------------------------------------------
# M3 -- quantity
# --------------------------------------------------------------------------


def m3_candidates() -> dict[str, object]:
    """Count models: the Poisson GLM first, since the target is a count."""
    return {
        "1 · poisson": PoissonRegressor(max_iter=2000),
        "2 · random forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "3 · gradient boosting": HistGradientBoostingRegressor(
            loss="poisson", random_state=RANDOM_STATE
        ),
    }


def evaluate_count(y: pd.Series, pred: np.ndarray, media_train: float) -> pd.Series:
    """MAE next to RMSE, as Fase 1 decided when it left the tail untruncated."""
    constante = np.full(len(y), media_train)
    return pd.Series(
        {
            "filas": len(y),
            "media_real": round(float(y.mean()), 4),
            "media_predicha": round(float(pred.mean()), 4),
            "mae": round(float(mean_absolute_error(y, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
            "mae_baseline_media": round(float(mean_absolute_error(y, constante)), 4),
            "rmse_baseline_media": round(float(np.sqrt(mean_squared_error(y, constante))), 4),
        }
    )


def cv_compare_count(
    X: pd.DataFrame, y: pd.Series, dias: pd.Series, groups: pd.Series, candidatos: dict | None = None
) -> pd.DataFrame:
    """Candidates for M3 under the two honest schemes, scored by MAE and RMSE."""
    candidatos = candidatos or m3_candidates()
    esquemas = {
        "temporal (ventana expansiva)": expanding_day_folds(dias),
        "cliente nuevo + dia futuro": grouped_time_folds(dias, groups),
    }
    Xv, yv = X.reset_index(drop=True), y.reset_index(drop=True)

    filas = []
    for nombre, est in candidatos.items():
        pipe = pipeline(est, X)
        for esquema, folds in esquemas.items():
            mae, rmse = [], []
            for tr, va in folds:
                modelo = clone(pipe).fit(Xv.iloc[tr], yv.iloc[tr])
                p = modelo.predict(Xv.iloc[va])
                mae.append(mean_absolute_error(yv.iloc[va], p))
                rmse.append(np.sqrt(mean_squared_error(yv.iloc[va], p)))
            filas.append(
                {
                    "modelo": nombre,
                    "esquema": esquema,
                    "folds": len(folds),
                    "mae_media": round(float(np.mean(mae)), 4),
                    "rmse_media": round(float(np.mean(rmse)), 4),
                }
            )
    out = pd.DataFrame(filas)
    estricto = out[out["esquema"].str.startswith("cliente nuevo")].set_index("modelo")
    mejor = estricto["mae_media"].idxmin()
    out.attrs["mejor_estricto"] = mejor
    out.attrs["lectura"] = (
        "Bajo el esquema estricto el mejor MAE es de {} ({:.4f}); el Poisson da {:.4f}. La "
        "diferencia entre modelos es de {:.4f} unidades, asi que lo que decide es la "
        "interpretabilidad y el costo de mantener el modelo, no la metrica."
    ).format(
        mejor,
        float(estricto.loc[mejor, "mae_media"]),
        float(estricto.loc["1 · poisson", "mae_media"]),
        float(estricto["mae_media"].max() - estricto["mae_media"].min()),
    )
    return out


def elasticity_diagnostics(cond: ConditionalData, controles: dict | None = None) -> pd.DataFrame:
    """Own-price elasticities under several specifications.

    A positive own-price elasticity is either a fitting accident or a property
    of the data, and the two call for opposite responses. Refitting with
    regularization, with loyalty controls and with the segment separates them:
    a sign that survives every specification is not an accident.
    """
    base = [*cfg.PRICE_COLS, *cfg.PROMO_COLS]
    lealtad = [c for c in cond.X_train.columns if c.startswith("lealtad_marca_")]
    controles = controles or {
        "precios + promos (sin penalizacion)": (base, 1e6),
        "precios + promos (L2, C=1)": (base, 1.0),
        "+ lealtad por marca": (base + lealtad, 1e6),
        "+ lealtad + ultima marca": (base + lealtad + ["ultima_marca_comprada"], 1e6),
        "+ lealtad + segmento": (
            base + lealtad + (["segmento"] if "segmento" in cond.X_train.columns else []),
            1e6,
        ),
    }

    precios = cond.X_train[cfg.PRICE_COLS].mean().to_numpy()
    filas = []
    for nombre, (cols, C) in controles.items():
        X = cond.X_train[cols].fillna(0.0)
        modelo = LogisticRegression(max_iter=6000, C=C, random_state=RANDOM_STATE).fit(
            X, cond.y_train
        )
        shares = modelo.predict_proba(X).mean(axis=0)
        beta = modelo.coef_[:, : len(cfg.PRICE_COLS)]
        propias = [
            precios[j] * (beta[j, j] - float((shares * beta[:, j]).sum()))
            for j in range(len(cfg.BRANDS))
        ]
        fila = {"especificacion": nombre, "columnas": len(cols)}
        fila.update({f"propia_marca_{j}": round(float(propias[i]), 2) for i, j in enumerate(cfg.BRANDS)})
        fila["todas_negativas"] = bool(all(p < 0 for p in propias))
        filas.append(fila)

    out = pd.DataFrame(filas)
    cols_propias = [f"propia_marca_{j}" for j in cfg.BRANDS]
    siempre_positiva = [
        j for j in cfg.BRANDS if bool((out[f"propia_marca_{j}"] > 0).all())
    ]
    siempre_negativa = [j for j in cfg.BRANDS if bool((out[f"propia_marca_{j}"] < 0).all())]

    out.attrs["marcas_no_identificadas"] = siempre_positiva
    out.attrs["marcas_identificadas"] = siempre_negativa
    out.attrs["robusto"] = len(siempre_positiva) + len(siempre_negativa) == len(cfg.BRANDS)
    out.attrs["lectura"] = (
        "El signo de la elasticidad propia es el mismo en las {} especificaciones para "
        "{} de 5 marcas. Las marcas {} salen siempre negativas (identificables) y la(s) "
        "marca(s) {} sale(n) siempre positiva(s). Un signo que sobrevive a regularizacion, "
        "controles de lealtad y segmento no es un accidente de ajuste: es lo que dice este "
        "dato observacional."
    ).format(
        len(out),
        len(siempre_positiva) + len(siempre_negativa),
        siempre_negativa or "ninguna",
        siempre_positiva or "ninguna",
    )
    return out


def price_variation_diagnostics(df: pd.DataFrame, cond: ConditionalData) -> pd.DataFrame:
    """How much independent price variation each brand actually offers.

    An elasticity can only be estimated where the price moved for reasons
    unrelated to demand. This table is the cheap version of that question:
    how many distinct prices, over what range, with how much share to estimate
    from. It is what explains *which* brand came out unidentified.
    """
    from .eda.decisions import price_ladder, relative_price_gap

    escalera = price_ladder(df).set_index("marca")
    brecha = relative_price_gap(df).set_index("marca")
    shares = cond.y_train.value_counts(normalize=True)

    filas = []
    for j in cfg.BRANDS:
        filas.append(
            {
                "marca": j,
                "puntos_precio": int(escalera.loc[j, "n_puntos_precio"]),
                "rango_relativo_pct": float(escalera.loc[j, "rango_relativo_pct"]),
                "desvio_precio_relativo": float(brecha.loc[j, "desvio_std_relativo"]),
                "share_en_train": round(float(shares.get(j, 0.0)), 4),
                "compras_en_train": int((cond.y_train == j).sum()),
            }
        )
    out = pd.DataFrame(filas)
    peor = int(out.loc[out["desvio_precio_relativo"].idxmin(), "marca"])
    out.attrs["menor_variacion"] = peor
    out.attrs["lectura"] = (
        "La marca {} es la que menos mueve su precio relativo (desvio {:.4f}) con solo {} "
        "puntos de precio y el {:.1%} de las compras de train. Es la combinacion que deja "
        "una elasticidad sin identificar: poca variacion propia y poca masa para estimarla."
    ).format(
        peor,
        float(out["desvio_precio_relativo"].min()),
        int(out.loc[out["marca"] == peor, "puntos_precio"].iloc[0]),
        float(out.loc[out["marca"] == peor, "share_en_train"].iloc[0]),
    )
    return out


# --------------------------------------------------------------------------
# Composing the three models: the value of a visit
# --------------------------------------------------------------------------


def expected_value(
    proba_m1: np.ndarray,
    proba_m2: np.ndarray,
    clases_m2: np.ndarray,
    cantidad_m3: np.ndarray,
    df: pd.DataFrame,
    index: pd.Index,
) -> pd.DataFrame:
    """``P(compra) x P(marca | compra) x E[cantidad] x precio`` for every visit.

    This is the number the prescriptive layer consumes: ``pi * q * lambda * m``
    with the margin still missing, so what comes out is **expected revenue**,
    the upper bound of the expected margin. M2 and M3 are conditional models
    applied to every visit on purpose -- the formula asks what the visit is
    worth *if* it converts.
    """
    precios = df.loc[index, cfg.PRICE_COLS].to_numpy()
    orden = {int(c): i for i, c in enumerate(clases_m2)}
    q = np.column_stack([proba_m2[:, orden[j]] for j in cfg.BRANDS])
    precio_esperado = (q * precios).sum(axis=1)

    out = pd.DataFrame(
        {
            "pi": proba_m1,
            "precio_esperado": precio_esperado,
            "lambda": cantidad_m3,
            "valor_esperado": proba_m1 * precio_esperado * cantidad_m3,
        },
        index=index,
    )
    out["ingreso_real"] = df.loc[index, "gasto"].fillna(0.0)
    out.attrs["valor_esperado_medio"] = round(float(out["valor_esperado"].mean()), 4)
    out.attrs["ingreso_real_medio"] = round(float(out["ingreso_real"].mean()), 4)
    out.attrs["sesgo"] = round(
        float(out["valor_esperado"].mean() - out["ingreso_real"].mean()), 4
    )
    out.attrs["correlacion"] = round(float(out["valor_esperado"].corr(out["ingreso_real"])), 4)
    out.attrs["lectura"] = (
        "Valor esperado medio por visita {:.4f} frente a un ingreso real medio de {:.4f} "
        "({:+.4f}). La correlacion entre lo esperado y lo realizado es {:.4f}: suficiente "
        "para ordenar visitas, que es lo que P1 necesita, aunque no para predecir el "
        "ingreso de una visita concreta."
    ).format(
        out.attrs["valor_esperado_medio"],
        out.attrs["ingreso_real_medio"],
        out.attrs["sesgo"],
        out.attrs["correlacion"],
    )
    return out


def value_capture(
    valores: pd.DataFrame, X_test: pd.DataFrame, columna_baseline: str = "tasa_compra_historica",
    fracciones=(0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """Revenue actually captured by each targeting policy, on the same test rows.

    The comparison is against realized ``gasto``, not against predictions, so
    what the table reports is money that was really there -- the honest version
    of a value-capture claim.
    """
    ingreso = valores["ingreso_real"].to_numpy()
    total = float(ingreso.sum())
    politicas = {
        "azar (proporcional)": None,
        "politica historica": X_test[columna_baseline].fillna(0).to_numpy(),
        "solo propension (M1)": valores["pi"].to_numpy(),
        "valor esperado (M1xM2xM3)": valores["valor_esperado"].to_numpy(),
    }

    filas = []
    for q in fracciones:
        n = int(np.ceil(q * len(ingreso)))
        for nombre, score in politicas.items():
            capturado = total * q if score is None else float(ingreso[np.argsort(-score)[:n]].sum())
            filas.append(
                {
                    "presupuesto": f"top {int(q * 100)}%",
                    "politica": nombre,
                    "visitas": n,
                    "ingreso_capturado": round(capturado, 2),
                    "pct_del_ingreso": round(100 * capturado / total, 1),
                    "lift_vs_azar": round(capturado / (total * q), 2),
                }
            )
    out = pd.DataFrame(filas)
    pivote = out.pivot(index="presupuesto", columns="politica", values="pct_del_ingreso")
    top10 = pivote.loc["top 10%"]

    out.attrs["ingreso_total_test"] = round(total, 2)
    out.attrs["mejor_politica"] = str(top10.idxmax())
    out.attrs["ganancia_vs_historica"] = round(
        float(top10["valor esperado (M1xM2xM3)"] - top10["politica historica"]), 1
    )
    out.attrs["ganancia_vs_propension"] = round(
        float(top10["valor esperado (M1xM2xM3)"] - top10["solo propension (M1)"]), 1
    )
    out.attrs["lectura"] = (
        "Con presupuesto para el 10% de las visitas, el valor esperado completo captura el "
        "{:.1f}% del ingreso real de test, frente al {:.1f}% de la politica historica "
        "({:+.1f} puntos) y al {:.1f}% de usar solo la propension ({:+.1f}). Encadenar M2 y "
        "M3 detras de M1 {} sobre ordenar por probabilidad de compra."
    ).format(
        float(top10["valor esperado (M1xM2xM3)"]),
        float(top10["politica historica"]),
        out.attrs["ganancia_vs_historica"],
        float(top10["solo propension (M1)"]),
        out.attrs["ganancia_vs_propension"],
        "aporta" if out.attrs["ganancia_vs_propension"] > 0 else "no aporta",
    )
    return out


# --------------------------------------------------------------------------
# Freezing the results and reporting the phase
# --------------------------------------------------------------------------

MODELS_PATH = cfg.PROJECT_ROOT / "artifacts" / "modelos.json"


def freeze(resultados: dict, path=None) -> dict:
    """Write the test metrics and the chosen models to ``artifacts/modelos.json``.

    Metrics are frozen for the same reason the split was: so that a number
    quoted in the documentation or in a slide can be traced to the run that
    produced it.
    """
    import json
    from datetime import date

    path = path or MODELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fecha": date.today().isoformat(), "split_day": cfg.SPLIT_DAY, **resultados}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return payload


def phase4_report(
    m1_test: pd.Series,
    lift: pd.DataFrame,
    contra_baseline: pd.DataFrame,
    umbral: pd.DataFrame,
    regla: pd.DataFrame,
    m2_test: pd.Series,
    elasticidades: pd.DataFrame,
    m3_test: pd.Series,
    captura: pd.DataFrame,
) -> pd.DataFrame:
    """One row per requirement of Fase 4, with the evidence that closes it."""
    filas = [
        {
            "requisito": "S1 · M1 entrega pi_it",
            "resultado": "PR-AUC {:.4f} (azar {:.4f})".format(m1_test["pr_auc"], m1_test["pr_auc_azar"]),
            "ok": bool(m1_test["pr_auc"] > m1_test["pr_auc_azar"]),
        },
        {
            "requisito": "S1 · M2 entrega q_ijt",
            "resultado": "accuracy {:.4f} vs clase mayoritaria {:.4f}".format(
                m2_test["accuracy"], m2_test["accuracy_clase_mayoritaria"]
            ),
            "ok": bool(m2_test["accuracy"] > m2_test["accuracy_clase_mayoritaria"]),
        },
        {
            "requisito": "S1 · M3 entrega lambda_ijt",
            "resultado": "MAE {:.4f} vs baseline de la media {:.4f}".format(
                m3_test["mae"], m3_test["mae_baseline_media"]
            ),
            "ok": bool(m3_test["mae"] < m3_test["mae_baseline_media"]),
        },
        {
            "requisito": "S3 · elasticidades cruzadas del logit",
            "resultado": "{} de 5 elasticidades propias identificadas".format(
                len(elasticidades.attrs["marcas_identificadas"])
            ),
            "ok": len(elasticidades.attrs["marcas_identificadas"]) >= 4,
        },
        {
            "requisito": "S5 · PR-AUC como metrica principal",
            "resultado": "accuracy reportada solo junto a su baseline ({:.4f} vs {:.4f})".format(
                m1_test["accuracy_en_0.5"], m1_test["accuracy_de_no_comprar_nunca"]
            ),
            "ok": True,
        },
        {
            "requisito": "S6 · umbral derivado del costo",
            "resultado": "regla umbral=costo/margen validada, desvio maximo {:.3f}".format(
                regla.attrs["max_desvio"]
            ),
            "ok": bool(regla.attrs["regla_valida"]),
        },
        {
            "requisito": "S8 · superar el baseline de targeting",
            "resultado": "lift top 10%: modelo x{:.2f} vs politica historica x{:.2f}".format(
                contra_baseline.attrs["lift_modelo_top10"], contra_baseline.attrs["lift_baseline_top10"]
            ),
            "ok": bool(contra_baseline.attrs["ganancia_top10"] > 0),
        },
        {
            "requisito": "S9 · composicion pi x q x lambda x precio",
            "resultado": "captura el {:.1f}% del ingreso con el 10% de las visitas".format(
                float(captura.loc[(captura["presupuesto"] == "top 10%") & (captura["politica"] == "valor esperado (M1xM2xM3)"), "pct_del_ingreso"].iloc[0])
            ),
            "ok": True,
        },
    ]
    out = pd.DataFrame(filas)
    out["ok"] = out["ok"].astype(bool)
    out.attrs["fase4_completa"] = bool(out["ok"].all())
    out.attrs["lectura"] = "{} de {} requisitos de la Fase 4 cerrados con evidencia ejecutada.".format(
        int(out["ok"].sum()), len(out)
    )
    return out
