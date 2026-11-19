"""Encoding and scaling inside a ``Pipeline`` fitted on train only.

Phase 2 of ``docs/brief-metodologia.md`` ends here: *"Codificación y escalado
dentro de un `Pipeline` de scikit-learn, ajustado **solo** sobre train."* This
module is also the one that closes control **E10** of
``docs/brief-alineamiento-eda.md``, the only blocking control the EDA notebook
left pending, because E10 cannot be verified before the pipeline exists.

Two leaks are possible at this stage and they are different:

* **Fitted feature statistics** -- the price mean a brand is compared against,
  the quantile edges of the demographic bins. Those live in
  :class:`ferreyros.features.FeatureReference`.
* **Fitted preprocessing statistics** -- the scaler's mean and variance, the
  encoder's category list. Those live here.

Both must come from train and nothing else, and :func:`verify_train_only_fit`
proves it instead of declaring it.

Missing values are *not* imputed with the mean. Per control E7, a client's
first visits have no history, and that absence is signal: an 8.5% block of
``NaN`` in the loyalty and recency features means "no purchase yet", not "value
unknown". The imputer fills a placeholder and ``add_indicator=True`` keeps the
absence itself as an explicit feature, so the model can learn from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from . import data as data_mod
from . import features as feat_mod

# --------------------------------------------------------------------------
# Column roles -- declared, not guessed from dtype
# --------------------------------------------------------------------------

#: Unordered categories. One-hot encoded; ``ultima_marca_comprada`` includes
#: the ``0`` level, which is absence of history and a category of its own (E7).
#: ``segmento`` is the label of model M0 (Fase 3). It only appears when the
#: caller attached it with ``segmentation.attach_segment``; declaring it here is
#: what makes that attachment the single step needed to use it as a feature.
NOMINAL_COLS: list[str] = ["ocupacion", "ultima_marca_comprada", "dia_semana", "segmento"]

#: Ordered categories. Kept as a single number -- the order carries meaning,
#: so one-hot encoding would throw information away.
ORDINAL_COLS: list[str] = ["nivel_educacion", "edad_bin", "ingreso_bin"]

#: Already 0/1. Scaling them would only make the coefficients harder to read.
BINARY_COLS: list[str] = [
    "genero",
    "estado_civil",
    "ultima_cantidad_comprada",
    "promo_en_ultima_marca",
    *cfg.PROMO_COLS,
]

#: History features that are legitimately missing on a client's first visits.
#: They get the placeholder-plus-indicator treatment described in the module
#: docstring, never mean imputation.
HISTORY_GAP_COLS: list[str] = [
    "tasa_compra_historica",
    "dias_desde_ultima_compra",
    "cantidad_media_historica",
    "ticket_medio_historico",
    "n_marcas_distintas_previas",
    *[f"lealtad_marca_{j}" for j in cfg.BRANDS],
]

#: Placeholder for an absent history. The indicator column is what the model
#: reads; this value only has to be a constant the indicator makes identifiable.
HISTORY_FILL_VALUE = 0.0


def classify_columns(X: pd.DataFrame) -> dict[str, list[str]]:
    """Split the feature matrix into the four preprocessing roles.

    Anything not explicitly declared above and not a history gap is treated as
    a complete numeric feature. Returning the classification -- rather than
    hiding it inside the transformer -- lets the notebook show it as a table.
    """
    present = [c for c in X.columns]
    nominal = [c for c in NOMINAL_COLS if c in present]
    ordinal = [c for c in ORDINAL_COLS if c in present]
    binary = [c for c in BINARY_COLS if c in present]
    history = [c for c in HISTORY_GAP_COLS if c in present]
    assigned = {*nominal, *ordinal, *binary, *history}
    numeric = [c for c in present if c not in assigned]
    return {
        "nominal": nominal,
        "ordinal": ordinal,
        "binaria": binary,
        "historial_con_vacios": history,
        "numerica": numeric,
    }


def column_roles(X: pd.DataFrame) -> pd.DataFrame:
    """The classification as a table: role, treatment and reason."""
    groups = classify_columns(X)
    treatment = {
        "nominal": ("OneHotEncoder(handle_unknown='ignore')", "categoria sin orden"),
        "ordinal": ("mediana + StandardScaler", "el orden es informacion; se conserva como numero"),
        "binaria": ("passthrough", "ya es 0/1; escalarla solo complica la lectura"),
        "historial_con_vacios": (
            f"constante {HISTORY_FILL_VALUE} + indicador + StandardScaler",
            "el vacio es ausencia de historial (E7), no un valor desconocido",
        ),
        "numerica": ("mediana + StandardScaler", "continua y completa"),
    }
    rows = []
    for role, cols in groups.items():
        how, why = treatment[role]
        rows.append(
            {
                "rol": role,
                "n_columnas": len(cols),
                "tratamiento": how,
                "por_que": why,
                "columnas": ", ".join(cols[:4]) + (" ..." if len(cols) > 4 else ""),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Assemble the unfitted ``ColumnTransformer`` for the feature matrix.

    Unfitted on purpose: the object is built from the *column names* of ``X``
    and learns no statistic until :meth:`fit` is called on train.
    """
    groups = classify_columns(X)

    numeric_pipe = Pipeline(
        [("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]
    )
    # The indicator is the point: it carries "this client had no history yet"
    # as a feature of its own instead of letting a filled value pretend to be
    # an observation.
    history_pipe = Pipeline(
        [
            (
                "imputar",
                SimpleImputer(
                    strategy="constant",
                    fill_value=HISTORY_FILL_VALUE,
                    add_indicator=True,
                ),
            ),
            ("escalar", StandardScaler()),
        ]
    )
    nominal_pipe = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numerica", numeric_pipe, groups["numerica"]),
            ("ordinal", numeric_pipe, groups["ordinal"]),
            ("historial", history_pipe, groups["historial_con_vacios"]),
            ("nominal", nominal_pipe, groups["nominal"]),
            ("binaria", "passthrough", groups["binaria"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class PreparedData:
    """Everything the modelling phase needs, with the fit boundary explicit.

    ``preprocessor`` and ``reference`` are the two fitted objects of the
    project, and both were fitted on ``X_train`` only.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    reference: feat_mod.FeatureReference
    feature_names: list[str]
    split_day: int
    groups_train: pd.Series
    groups_test: pd.Series

    @property
    def shapes(self) -> pd.DataFrame:
        """Row and column counts before and after preprocessing."""
        return pd.DataFrame(
            [
                {
                    "conjunto": "train",
                    "filas": len(self.X_train),
                    "cols_entrada": self.X_train.shape[1],
                    "cols_salida": len(self.feature_names),
                    "tasa_compra": round(float(self.y_train.mean()), 4),
                    "clientes": int(self.groups_train.nunique()),
                },
                {
                    "conjunto": "test",
                    "filas": len(self.X_test),
                    "cols_entrada": self.X_test.shape[1],
                    "cols_salida": len(self.feature_names),
                    "tasa_compra": round(float(self.y_test.mean()), 4),
                    "clientes": int(self.groups_test.nunique()),
                },
            ]
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the train-fitted preprocessor, keeping the column names."""
        arr = self.preprocessor.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names, index=X.index)


def prepare(
    df: pd.DataFrame | None = None,
    split_day: int | None = None,
    extra: list[str] | None = None,
) -> PreparedData:
    """Build the modelling matrices end to end, fitting only on train.

    The order is the whole point of this function:

    1. temporal split (control E8) -- decided *before* anything is fitted;
    2. :func:`ferreyros.features.fit_reference` on train only;
    3. features built over the full timeline so client history can cross the
       boundary backwards (control E6), using that train-fitted reference;
    4. the preprocessor fitted on the train rows only (control E10).

    Passing ``df=None`` loads the raw data, so the notebook can call
    ``prepare()`` and get a reproducible matrix in one line.

    ``extra`` names admissible columns already present in ``df`` that are not
    part of the standard feature set -- today only ``segmento``, the label of
    model M0. It is explicit on purpose: ``feature_matrix`` is a whitelist, so
    a column nobody declared can never reach the model by accident.
    """
    split_day = split_day or cfg.SPLIT_DAY
    if df is None:
        df = data_mod.add_paid_price_and_spend(data_mod.load_raw())

    train_raw, _ = data_mod.temporal_split(df, split_day)
    reference = feat_mod.fit_reference(train_raw)

    featured = feat_mod.build_features(df, reference=reference)
    train, test = data_mod.temporal_split(featured, split_day)

    X_train = feat_mod.feature_matrix(train, extra=extra)
    # Same columns in the same order on both sides, so the fitted transformer
    # can be applied to test without silent realignment.
    X_test = feat_mod.feature_matrix(test, extra=extra)[list(X_train.columns)]
    y_train = train[cfg.TARGET_INCIDENCE]
    y_test = test[cfg.TARGET_INCIDENCE]

    preprocessor = build_preprocessor(X_train).fit(X_train)
    names = [str(n) for n in preprocessor.get_feature_names_out()]

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        reference=reference,
        feature_names=names,
        split_day=split_day,
        groups_train=train[cfg.KEY_CLIENT],
        groups_test=test[cfg.KEY_CLIENT],
    )


# --------------------------------------------------------------------------
# Control E10 -- proven, not declared
# --------------------------------------------------------------------------


def verify_train_only_fit(prepared: PreparedData) -> pd.DataFrame:
    """Control E10: prove the preprocessor learned nothing from test.

    Four assertions. The second is the one that gives the control teeth: the
    first would also pass on a pipeline fitted over everything, so without a
    materiality check the control would be cosmetic.

    1. The scaler's means equal the means recomputed on train by hand.
    2. Fitting the same pipeline on train+test yields **different** parameters.
    3. Transforming test does not move the fitted parameters -- no implicit
       refit on the test set.
    4. The encoder's categories come from train; an unseen test category is
       ignored rather than silently creating a column.
    """
    pre = prepared.preprocessor
    X_train, X_test = prepared.X_train, prepared.X_test
    rows = []

    # 1 -- scaler means equal the hand-computed train means.
    scaler_cols, fitted_means = _scaler_means(pre)
    manual = np.array([_expected_mean(X_train, c) for c in scaler_cols])
    max_gap = float(np.max(np.abs(fitted_means - manual))) if len(scaler_cols) else 0.0
    rows.append(
        {
            "control": "E10.1",
            "verificacion": "La media del escalador es la media de train",
            "medida": f"desvio maximo {max_gap:.2e} en {len(scaler_cols)} columnas",
            "cumple": max_gap < 1e-8,
        }
    )

    # 2 -- fitting on train+test would give different parameters.
    X_all = pd.concat([X_train, X_test])
    pre_all = build_preprocessor(X_train).fit(X_all)
    _, means_all = _scaler_means(pre_all)
    n_differ = int(np.sum(np.abs(means_all - fitted_means) > 1e-9)) if len(scaler_cols) else 0
    rows.append(
        {
            "control": "E10.2",
            "verificacion": "Ajustar sobre train+test daria parametros distintos",
            "medida": f"{n_differ} de {len(scaler_cols)} medias cambiarian",
            "cumple": n_differ > 0,
        }
    )

    # 3 -- transforming test does not refit.
    before = fitted_means.copy()
    pre.transform(X_test)
    _, after = _scaler_means(pre)
    rows.append(
        {
            "control": "E10.3",
            "verificacion": "transform(test) no modifica los parametros ajustados",
            "medida": f"desvio maximo {float(np.max(np.abs(after - before))) if len(scaler_cols) else 0.0:.2e}",
            "cumple": bool(np.allclose(before, after)),
        }
    )

    # 4 -- encoder categories come from train.
    cats_ok, detail = _encoder_categories_from_train(pre, X_train, X_test)
    rows.append(
        {
            "control": "E10.4",
            "verificacion": "Las categorias del codificador provienen de train",
            "medida": detail,
            "cumple": cats_ok,
        }
    )

    out = pd.DataFrame(rows)
    out["estado"] = out["cumple"].map({True: "OK", False: "FALLA"})
    out.attrs["e10_ok"] = bool(out["cumple"].all())
    out.attrs["lectura"] = (
        f"Pipeline ajustado sobre {len(X_train):,} filas de train (dias 1-{prepared.split_day}); "
        f"{n_differ} parametros cambiarian si se ajustara sobre las {len(X_all):,} filas, "
        "de modo que el limite del ajuste es material y no declarativo."
    )
    return out


def _expected_mean(X: pd.DataFrame, col: str) -> float:
    """Mean the scaler should hold for ``col`` after this module's imputation."""
    serie = X[col]
    if col in HISTORY_GAP_COLS:
        filled = serie.fillna(HISTORY_FILL_VALUE)
    else:
        filled = serie.fillna(serie.median())
    return float(filled.mean())


def _scaler_means(pre: ColumnTransformer) -> tuple[list[str], np.ndarray]:
    """Columns and fitted means of every ``StandardScaler`` in the transformer.

    The indicator columns the history imputer appends have no counterpart in
    the input frame, so they are skipped: only the original columns are
    compared against a hand-computed mean.
    """
    cols: list[str] = []
    means: list[float] = []
    for name, trans, columns in pre.transformers_:
        if not isinstance(trans, Pipeline) or "escalar" not in dict(trans.steps):
            continue
        scaler = trans.named_steps["escalar"]
        fitted = np.asarray(scaler.mean_)
        # Original columns come first; appended indicators follow.
        for i, col in enumerate(list(columns)):
            cols.append(col)
            means.append(float(fitted[i]))
    return cols, np.asarray(means)


def _encoder_categories_from_train(
    pre: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[bool, str]:
    """Check every encoder category exists in train, and report test-only levels."""
    for name, trans, columns in pre.transformers_:
        if not isinstance(trans, Pipeline) or "codificar" not in dict(trans.steps):
            continue
        encoder = trans.named_steps["codificar"]
        unseen = 0
        for col, cats in zip(list(columns), encoder.categories_):
            train_levels = set(X_train[col].dropna().unique())
            if not set(cats).issubset(train_levels):
                return False, f"{col}: categorias fuera de train"
            unseen += len(set(X_test[col].dropna().unique()) - set(cats))
        total = sum(len(c) for c in encoder.categories_)
        return True, f"{total} categorias, todas de train; {unseen} niveles de test ignorados"
    return True, "sin codificador en el pipeline"


# --------------------------------------------------------------------------
# Diagnostics the modelling phase will need
# --------------------------------------------------------------------------


def out_of_range_report(prepared: PreparedData, top: int = 10) -> pd.DataFrame:
    """Features whose test values fall outside the train range.

    Not a defect in itself -- a temporal split guarantees it for any trend
    feature -- but it is the honest reason the validation must be temporal, and
    it flags which features extrapolate instead of interpolate.
    """
    rows = []
    for col in prepared.X_train.columns:
        tr, te = prepared.X_train[col], prepared.X_test[col]
        lo, hi = float(tr.min()), float(tr.max())
        outside = float(((te < lo) | (te > hi)).mean())
        if outside > 0:
            rows.append(
                {
                    "feature": col,
                    "rango_train": f"[{lo:.3g}, {hi:.3g}]",
                    "rango_test": f"[{float(te.min()):.3g}, {float(te.max()):.3g}]",
                    "pct_test_fuera_de_rango": round(100 * outside, 1),
                }
            )
    out = pd.DataFrame(rows).sort_values("pct_test_fuera_de_rango", ascending=False)
    out.attrs["n_features_extrapolan"] = len(out)
    return out.head(top).reset_index(drop=True)


def preprocessing_summary(prepared: PreparedData) -> pd.Series:
    """Headline facts of the processing phase, for the notebook to print."""
    groups = classify_columns(prepared.X_train)
    n_indicator = sum(1 for n in prepared.feature_names if "missingindicator" in n.lower())
    # Every output column is either passed through 1:1, an absence indicator,
    # or a one-hot level; the remainder is therefore the one-hot block.
    n_passthrough = sum(
        len(groups[role]) for role in ("numerica", "ordinal", "binaria", "historial_con_vacios")
    )
    return pd.Series(
        {
            "filas totales": len(prepared.X_train) + len(prepared.X_test),
            "corte temporal (dia)": prepared.split_day,
            "filas train / test": f"{len(prepared.X_train):,} / {len(prepared.X_test):,}",
            "features de entrada": prepared.X_train.shape[1],
            "features tras codificar": len(prepared.feature_names),
            "columnas indicadoras de ausencia": n_indicator,
            "columnas one-hot": len(prepared.feature_names) - n_passthrough - n_indicator,
            "nulos tras preprocesar": int(
                np.isnan(prepared.preprocessor.transform(prepared.X_train)).sum()
            ),
        }
    )
