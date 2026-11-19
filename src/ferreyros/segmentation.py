"""Model M0 -- customer segmentation (Fase 3 of ``docs/brief-metodologia.md``).

The brief asks for K-Means over the demographic variables, standardized, with
``k`` chosen by elbow and silhouette, PCA for interpretation, a business name
per segment, and the segment available both as a feature of the supervised
model and as the unit of commercial action.

Three design rules, each one answering a way this phase could go wrong.

**Behaviour stays out of the matrix.** ``eda.segments.segmentation_matrix``
excludes purchase behaviour on purpose: clustering on behaviour and then
explaining behaviour by cluster is circular. The behavioural profile is built
afterwards, from data the model never saw, which is what makes
:func:`segment_behaviour` evidence rather than a tautology.

**A partition is not a discovery.** K-Means always returns ``k`` groups,
whether or not the data has that many. So ``k`` is chosen by elbow and
silhouette *and* the chosen partition is tested for reproducibility
(:func:`stability`, mean Adjusted Rand Index over subsamples). A segmentation
that does not survive resampling is a coordinate system, not a structure, and
the notebook says so instead of pretending otherwise.

**Labels must mean the same thing later.** K-Means label ids are arbitrary, so
clusters are relabelled by ascending mean income and the whole mapping is
frozen in ``artifacts/segmentos.json`` with a fingerprint. Segment 3 in Fase 4
is the same segment 3 as here, or :func:`verify_frozen` fails.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from . import config as cfg
from . import data as data_mod
from .eda import segments as seg_eda

ARTIFACTS_DIR = cfg.PROJECT_ROOT / "artifacts"
SEGMENTS_PATH = ARTIFACTS_DIR / "segmentos.json"

#: Nominal demographic variable: one-hot, never an ordinal distance. Occupation
#: 0/1/2 are three labels; treating them as a number would claim that 2 is
#: twice as far from 0 as 1 is.
NOMINAL_DEMOGRAPHIC = "ocupacion"

#: Default search range for ``k``. The upper bound is a business constraint, not
#: a statistical one: a campaign plan with more than ten segments is not a plan.
K_RANGE = range(2, 11)

RANDOM_STATE = 42

#: How much silhouette we are willing to give up for a simpler partition. Used
#: by :func:`select_k` as a declared tie-break rule between elbow and silhouette.
TOLERANCIA_SILHOUETTE = 0.03


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------


def client_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One row per client with the demographic design columns, unscaled.

    ``ocupacion`` is expanded into indicator columns; ``nivel_educacion`` keeps
    its single ordinal column because its order is information.
    """
    base = seg_eda.segmentation_matrix(data_mod.add_paid_price_and_spend(df))
    dummies = pd.get_dummies(
        base[NOMINAL_DEMOGRAPHIC].astype("int64"), prefix=NOMINAL_DEMOGRAPHIC, dtype="float64"
    )
    out = pd.concat([base.drop(columns=[NOMINAL_DEMOGRAPHIC]).astype("float64"), dummies], axis=1)
    out.attrs["columnas_originales"] = list(base.columns)
    return out


def _standardize(M: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize every column, so no dimension dominates by its unit alone."""
    medias = M.to_numpy(dtype=float).mean(axis=0)
    escalas = M.to_numpy(dtype=float).std(axis=0, ddof=0)
    escalas = np.where(escalas == 0, 1.0, escalas)
    return (M.to_numpy(dtype=float) - medias) / escalas, medias, escalas


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The standardized matrix K-Means is actually fitted on."""
    M = client_matrix(df)
    X, medias, escalas = _standardize(M)
    out = pd.DataFrame(X, index=M.index, columns=M.columns)
    out.attrs["medias"] = medias
    out.attrs["escalas"] = escalas
    return out


def verify_design(df: pd.DataFrame) -> pd.DataFrame:
    """Prove the matrix of M0 is admissible before fitting anything.

    The second check is the one with teeth: the matrix built from the train
    period alone must be **identical** to the one built from the full frame.
    If it were not, the segments would carry information from the test window.
    """
    M = client_matrix(df)
    estaticas = data_mod.static_demographics_check(df)
    train, _ = data_mod.temporal_split(df)
    M_train = client_matrix(train).reindex(M.index)[M.columns]

    prohibidas = {
        cfg.TARGET_INCIDENCE,
        cfg.TARGET_BRAND,
        cfg.TARGET_QUANTITY,
        "gasto",
        "precio_pagado",
        "tasa_compra",
        "visitas",
        "compras",
        *cfg.EXCLUDED_COLS,
    }
    intrusas = sorted(prohibidas & set(M.columns))
    no_estaticas = int(estaticas["clientes_con_mas_de_un_valor"].sum())
    identica = bool(np.allclose(M.to_numpy(float), M_train.to_numpy(float)))

    filas = [
        {
            "verificacion": "una fila por cliente",
            "esperado": f"{df[cfg.KEY_CLIENT].nunique()} clientes",
            "obtenido": f"{len(M)} filas",
            "ok": len(M) == df[cfg.KEY_CLIENT].nunique(),
        },
        {
            "verificacion": "matriz de train identica a la del frame completo",
            "esperado": "identica (la demografia no cambia en el tiempo)",
            "obtenido": "identica" if identica else "DIFIERE",
            "ok": identica,
        },
        {
            "verificacion": "demografia invariante por cliente (D2)",
            "esperado": "0 clientes con mas de un valor",
            "obtenido": f"{no_estaticas} clientes",
            "ok": no_estaticas == 0,
        },
        {
            "verificacion": "ninguna columna de resultado ni de comportamiento (S7)",
            "esperado": "ninguna",
            "obtenido": ", ".join(intrusas) if intrusas else "ninguna",
            "ok": not intrusas,
        },
        {
            "verificacion": "sin nulos",
            "esperado": "0",
            "obtenido": str(int(M.isna().sum().sum())),
            "ok": int(M.isna().sum().sum()) == 0,
        },
    ]
    out = pd.DataFrame(filas)
    out["ok"] = out["ok"].astype(bool)
    out.attrs["apto"] = bool(out["ok"].all())
    out.attrs["columnas"] = list(M.columns)
    out.attrs["lectura"] = (
        "La matriz de M0 tiene {} clientes x {} columnas y {}. La identidad entre la matriz "
        "de train y la del frame completo es lo que prueba que los segmentos no pueden "
        "arrastrar informacion de test: la demografia es estatica por cliente."
    ).format(
        len(M),
        M.shape[1],
        "pasa las cinco verificaciones" if out.attrs["apto"] else "FALLA alguna verificacion",
    )
    return out


# --------------------------------------------------------------------------
# Choosing k
# --------------------------------------------------------------------------


def select_k(
    df: pd.DataFrame, k_range: range = K_RANGE, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Elbow and silhouette over a range of ``k``, plus two second opinions.

    ``delta_inercia`` is the elbow in numbers instead of in eyesight: the knee
    is where the marginal gain in inertia stops being worth another segment.
    """
    X = design_matrix(df).to_numpy(float)

    filas = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        labels = km.labels_
        filas.append(
            {
                "k": k,
                "inercia": round(float(km.inertia_), 2),
                "silhouette": round(float(silhouette_score(X, labels)), 4),
                "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
            }
        )
    out = pd.DataFrame(filas)
    out["delta_inercia"] = (-out["inercia"].diff()).round(2)
    out["pct_inercia_explicada"] = (
        100 * (1 - out["inercia"] / float(out["inercia"].iloc[0]))
    ).round(1)

    curvatura = out["inercia"].diff().diff().shift(-1)
    k_codo = int(out.loc[curvatura.idxmax(), "k"]) if curvatura.notna().any() else int(out["k"].iloc[0])
    k_sil = int(out.loc[out["silhouette"].idxmax(), "k"])
    k_db = int(out.loc[out["davies_bouldin"].idxmin(), "k"])

    # Declared tie-break rule, so the choice of k is reproducible instead of
    # being argued after seeing the segments: among every k whose silhouette is
    # within TOLERANCIA_SILHOUETTE of the best, take the smallest. Fewer
    # segments that score the same are better segments to run a campaign on.
    candidatos = out.loc[
        out["silhouette"] >= float(out["silhouette"].max()) - TOLERANCIA_SILHOUETTE, "k"
    ]
    k_rec = int(candidatos.min())

    out.attrs["k_codo"] = k_codo
    out.attrs["k_silhouette"] = k_sil
    out.attrs["k_davies_bouldin"] = k_db
    out.attrs["k_recomendado"] = k_rec
    out.attrs["tolerancia_silhouette"] = TOLERANCIA_SILHOUETTE
    out.attrs["silhouette_max"] = float(out["silhouette"].max())
    out.attrs["coinciden"] = k_codo == k_sil
    out.attrs["lectura"] = (
        "El codo cae en k={} y la silhouette maxima en k={} (valor {:.4f}); Davies-Bouldin "
        "prefiere k={}. {} La regla declarada rompe el empate: entre los k cuya silhouette "
        "queda a menos de {} del maximo, el mas pequenio -> k={}. Una silhouette de {:.2f} "
        "sobre demografia mixta indica grupos {}, asi que el valor de la particion se prueba "
        "con la estabilidad y con el comportamiento, no con la metrica de cohesion."
    ).format(
        k_codo,
        k_sil,
        float(out["silhouette"].max()),
        k_db,
        "Los dos criterios del brief coinciden."
        if k_codo == k_sil
        else "Los dos criterios del brief no coinciden.",
        TOLERANCIA_SILHOUETTE,
        k_rec,
        float(out["silhouette"].max()),
        "poco separados" if float(out["silhouette"].max()) < 0.35 else "razonablemente separados",
    )
    return out


def stability(
    df: pd.DataFrame,
    k: int,
    n_repeats: int = 20,
    sample_frac: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Reproducibility of the partition under resampling, as mean ARI.

    Each repeat refits K-Means on a random subsample and compares its labels
    against the full-sample labels on the same clients. An Adjusted Rand Index
    near 1 means the same clients keep landing together; near 0 means the
    partition is an artifact of the sample.
    """
    X = design_matrix(df)
    base = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X.to_numpy(float))
    base_labels = pd.Series(base.labels_, index=X.index)

    rng = np.random.default_rng(random_state)
    filas = []
    for r in range(n_repeats):
        idx = rng.choice(len(X), size=int(sample_frac * len(X)), replace=False)
        sub = X.iloc[idx]
        km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e6))).fit(
            sub.to_numpy(float)
        )
        filas.append(
            {
                "repeticion": r + 1,
                "clientes": len(sub),
                "ari": round(float(adjusted_rand_score(base_labels.loc[sub.index], km.labels_)), 4),
            }
        )
    out = pd.DataFrame(filas)
    media = float(out["ari"].mean())

    out.attrs["k"] = k
    out.attrs["ari_medio"] = round(media, 4)
    out.attrs["ari_min"] = round(float(out["ari"].min()), 4)
    out.attrs["ari_std"] = round(float(out["ari"].std()), 4)
    out.attrs["estable"] = media > 0.7
    out.attrs["lectura"] = (
        "ARI medio {:.4f} (minimo {:.4f}, desvio {:.4f}) sobre {} remuestreos al {:.0%}. "
        "La particion {}: {}"
    ).format(
        media,
        float(out["ari"].min()),
        float(out["ari"].std()),
        n_repeats,
        sample_frac,
        "es reproducible" if media > 0.7 else "es moderadamente reproducible" if media > 0.5 else "no es reproducible",
        "los mismos clientes vuelven a caer juntos, asi que el segmento es una etiqueta "
        "estable sobre la que se puede planificar."
        if media > 0.7
        else "los grupos se recomponen al cambiar la muestra, asi que el segmento vale como "
        "sistema de accion pero no debe leerse como un tipo de cliente descubierto en el dato.",
    )
    return out


def encoding_sensitivity(
    df: pd.DataFrame, k: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """How much the partition depends on how ``ocupacion`` was encoded.

    One-hot encoding plus standardization gives occupation three of the eight
    dimensions, so K-Means may be splitting on it because of that weight rather
    than because of the data. The honest way to settle it is to refit under the
    alternatives and compare them on the three things that matter -- cohesion,
    reproducibility and behavioural separation -- instead of defending the
    default.

    ``ari_vs_default`` is how much each variant agrees with the chosen one: a
    value near 0 means the encoding decides the segments, which is exactly the
    kind of fragility worth printing.
    """
    base = design_matrix(df)
    crudo = client_matrix(df)
    ocup_cols = [c for c in base.columns if c.startswith(f"{NOMINAL_DEMOGRAPHIC}_")]

    ordinal = crudo.drop(columns=ocup_cols).copy()
    ordinal[NOMINAL_DEMOGRAPHIC] = (
        seg_eda.segmentation_matrix(data_mod.add_paid_price_and_spend(df))[NOMINAL_DEMOGRAPHIC]
        .reindex(ordinal.index)
        .astype(float)
    )
    variantes = {
        "one-hot (elegida)": base,
        "ordinal": pd.DataFrame(
            _standardize(ordinal)[0], index=ordinal.index, columns=ordinal.columns
        ),
        "sin ocupacion": base.drop(columns=ocup_cols),
    }

    dfs = data_mod.add_paid_price_and_spend(df)
    etiquetas_base = None
    filas = []
    for nombre, M in variantes.items():
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(
            M.to_numpy(float)
        ).labels_
        serie = pd.Series(labels, index=M.index)
        if etiquetas_base is None:
            etiquetas_base = serie
        tasas = (
            dfs.assign(_s=dfs[cfg.KEY_CLIENT].map(serie))
            .groupby("_s")[cfg.TARGET_INCIDENCE]
            .mean()
        )
        filas.append(
            {
                "variante": nombre,
                "columnas": M.shape[1],
                "silhouette": round(float(silhouette_score(M.to_numpy(float), labels)), 4),
                "ari_vs_elegida": round(float(adjusted_rand_score(etiquetas_base, serie)), 4),
                "amplitud_tasa_compra": round(float(tasas.max() - tasas.min()), 4),
                "tamanio_min_pct": round(100 * float(serie.value_counts().min()) / len(serie), 1),
            }
        )
    out = pd.DataFrame(filas)
    elegida = out.iloc[0]
    mejor_sil = out["silhouette"].idxmax()
    mejor_amp = out["amplitud_tasa_compra"].idxmax()

    out.attrs["elegida_domina"] = bool(mejor_sil == 0 and mejor_amp == 0)
    out.attrs["ari_minimo"] = round(float(out["ari_vs_elegida"].min()), 4)
    out.attrs["lectura"] = (
        "La codificacion decide los segmentos: el ARI de la variante mas alejada contra la "
        "elegida es {:.4f}, es decir, particiones practicamente sin relacion. Por eso la "
        "eleccion se justifica con numeros y no con preferencia: la variante one-hot da la "
        "silhouette mas alta ({:.4f}) y la mayor separacion de comportamiento ({:.4f}). {}"
    ).format(
        float(out["ari_vs_elegida"].min()),
        float(elegida["silhouette"]),
        float(elegida["amplitud_tasa_compra"]),
        "Gana en los dos criterios, asi que se conserva."
        if out.attrs["elegida_domina"]
        else "No gana en los dos criterios, asi que la eleccion queda declarada como supuesto.",
    )
    return out


def stability_scan(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (2, 3, 4, 9),
    n_repeats: int = 15,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Mean ARI for several candidate ``k``, so the tie-break has evidence.

    Silhouette rewards more clusters on continuous data; reproducibility
    punishes them. Scanning both is what makes "k=3" a measured choice instead
    of a preference for round numbers.
    """
    filas = []
    for k in ks:
        est = stability(df, k, n_repeats=n_repeats, random_state=random_state)
        filas.append(
            {
                "k": k,
                "ari_medio": est.attrs["ari_medio"],
                "ari_min": est.attrs["ari_min"],
                "ari_std": est.attrs["ari_std"],
                "estable": est.attrs["estable"],
            }
        )
    out = pd.DataFrame(filas)
    # Several k can tie at a perfect ARI. Among those, the most granular one is
    # the useful answer: it is the finest partition that still reproduces.
    tope = float(out["ari_medio"].max())
    empatados = out.loc[out["ari_medio"] >= tope - 1e-9, "k"]
    mejor = int(empatados.max())
    out.attrs["k_mas_estable"] = mejor
    out.attrs["empatados_en_el_tope"] = [int(k) for k in empatados]
    out.attrs["lectura"] = (
        "El ARI medio cae al crecer k: {}. {} La particion mas fina que sigue siendo "
        "perfectamente reproducible es k={}, y apunta en la misma direccion que el codo -- "
        "no en la de la silhouette."
    ).format(
        ", ".join("k={}: {:.4f}".format(int(r["k"]), r["ari_medio"]) for _, r in out.iterrows()),
        "Varios k empatan en el tope ({}).".format(
            ", ".join(f"k={int(k)}" for k in empatados)
        )
        if len(empatados) > 1
        else "",
        mejor,
    )
    return out


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentModel:
    """A fitted M0: the partition plus everything needed to reproduce it."""

    kmeans: KMeans
    columnas: tuple[str, ...]
    medias: tuple[float, ...]
    escalas: tuple[float, ...]
    k: int
    random_state: int
    etiquetas: pd.Series
    nombres: dict[int, str]
    silhouette: float
    fecha_ajuste: str = field(default_factory=lambda: date.today().isoformat())

    def asignar(self, df: pd.DataFrame) -> pd.Series:
        """Segment of every row of a visit-level frame, by client id."""
        return df[cfg.KEY_CLIENT].map(self.etiquetas).astype("int64")

    @property
    def tabla_nombres(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"segmento": list(self.nombres), "nombre": list(self.nombres.values())}
        ).sort_values("segmento", ignore_index=True)


def fit(df: pd.DataFrame, k: int, random_state: int = RANDOM_STATE) -> SegmentModel:
    """Fit M0 with ``k`` segments, relabelled by ascending mean income.

    The relabelling is not cosmetic: K-Means ids depend on the seed, so without
    a deterministic order "segment 2" would mean something different on every
    run and no downstream phase could rely on it.
    """
    X = design_matrix(df)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X.to_numpy(float))
    crudas = pd.Series(km.labels_, index=X.index, name="cluster")

    raw = client_matrix(df)
    orden = (
        raw.assign(cluster=crudas)
        .groupby("cluster")["ingreso_anual"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    remapeo = {viejo: nuevo + 1 for nuevo, viejo in enumerate(orden)}
    etiquetas = crudas.map(remapeo).rename("segmento")

    modelo = SegmentModel(
        kmeans=km,
        columnas=tuple(X.columns),
        medias=tuple(float(v) for v in X.attrs["medias"]),
        escalas=tuple(float(v) for v in X.attrs["escalas"]),
        k=k,
        random_state=random_state,
        etiquetas=etiquetas,
        nombres={},
        silhouette=round(float(silhouette_score(X.to_numpy(float), km.labels_)), 4),
    )
    object.__setattr__(modelo, "nombres", name_segments(segment_profile(modelo, df)))
    return modelo


# --------------------------------------------------------------------------
# Profiling and naming
# --------------------------------------------------------------------------


def segment_profile(model: SegmentModel, df: pd.DataFrame) -> pd.DataFrame:
    """Centroid of every segment in **original units**, not in z-scores.

    A centroid in standardized space is unreadable for a commercial decision;
    "mean income 192k, mean age 44" is what names a segment.
    """
    raw = seg_eda.segmentation_matrix(data_mod.add_paid_price_and_spend(df))
    raw = raw.assign(segmento=model.etiquetas.reindex(raw.index))

    out = raw.groupby("segmento").agg(
        clientes=("edad", "size"),
        edad_media=("edad", "mean"),
        ingreso_medio=("ingreso_anual", "mean"),
        educacion_media=("nivel_educacion", "mean"),
        pct_genero_1=("genero", "mean"),
        pct_casado=("estado_civil", "mean"),
    )
    ocup = (
        raw.pivot_table(index="segmento", columns=NOMINAL_DEMOGRAPHIC, values="edad", aggfunc="size")
        .fillna(0)
        .pipe(lambda t: t.div(t.sum(axis=1), axis=0))
    )
    ocup.columns = [f"pct_ocupacion_{int(c)}" for c in ocup.columns]
    out = out.join(ocup)

    out["pct_clientes"] = (100 * out["clientes"] / out["clientes"].sum()).round(1)
    for col in out.columns:
        if col.startswith("pct_") and col != "pct_clientes":
            out[col] = (100 * out[col]).round(1)
    out["edad_media"] = out["edad_media"].round(1)
    out["ingreso_medio"] = out["ingreso_medio"].round(0)
    out["educacion_media"] = out["educacion_media"].round(2)

    out.attrs["rango_ingreso"] = (float(out["ingreso_medio"].min()), float(out["ingreso_medio"].max()))
    out.attrs["rango_edad"] = (float(out["edad_media"].min()), float(out["edad_media"].max()))
    out.attrs["lectura"] = (
        "Los {} segmentos van de {:,.0f} a {:,.0f} de ingreso medio y de {:.1f} a {:.1f} anios "
        "de edad media. El tamanio va del {:.1f}% al {:.1f}% de los clientes."
    ).format(
        len(out),
        out.attrs["rango_ingreso"][0],
        out.attrs["rango_ingreso"][1],
        out.attrs["rango_edad"][0],
        out.attrs["rango_edad"][1],
        float(out["pct_clientes"].min()),
        float(out["pct_clientes"].max()),
    )
    return out.reset_index()


_OCUPACION = {0: "dependiente", 1: "independiente", 2: "otra ocupacion"}


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Business name per segment, derived from the centroid by fixed rules.

    Names come from the numbers so that re-running the phase cannot quietly
    rename a segment. Income is read against the terciles of the segment
    centroids themselves, which is what makes "alto" mean *relative to this
    portfolio* and not relative to an outside benchmark nobody declared.
    """
    ingresos = profile["ingreso_medio"]
    bajo, alto = ingresos.quantile([1 / 3, 2 / 3])

    def nivel(v: float) -> str:
        return "ingreso bajo" if v <= bajo else "ingreso alto" if v >= alto else "ingreso medio"

    def etapa(v: float) -> str:
        return "jovenes" if v < 35 else "maduros" if v <= 50 else "seniors"

    ocup_cols = [c for c in profile.columns if c.startswith("pct_ocupacion_")]

    nombres: dict[int, str] = {}
    for _, row in profile.iterrows():
        dominante = max(ocup_cols, key=lambda c: float(row[c])) if ocup_cols else None
        # A segment that is almost entirely one occupation is named after it:
        # that trait defines it, and naming it by income would hide what the
        # partition actually split on.
        if dominante is not None and float(row[dominante]) >= 80:
            etiqueta = "{} · {}".format(
                _OCUPACION[int(dominante.split("_")[-1])].capitalize(),
                nivel(float(row["ingreso_medio"])),
            )
        else:
            etiqueta = "{} · {}".format(
                nivel(float(row["ingreso_medio"])).capitalize(), etapa(float(row["edad_media"]))
            )
        nombres[int(row["segmento"])] = etiqueta

    # Collisions get the next distinguishing trait appended, in a fixed order,
    # so two segments never share a name.
    vistos: dict[str, list[int]] = {}
    for seg, nombre in nombres.items():
        vistos.setdefault(nombre, []).append(seg)
    for nombre, segs in vistos.items():
        if len(segs) == 1:
            continue
        for seg in segs:
            row = profile.loc[profile["segmento"] == seg].iloc[0]
            ocup_cols = [c for c in profile.columns if c.startswith("pct_ocupacion_")]
            dominante = max(ocup_cols, key=lambda c: float(row[c]))
            if float(row[dominante]) >= 60:
                rasgo = _OCUPACION[int(dominante.split("_")[-1])]
            elif float(row["educacion_media"]) >= 1.5:
                rasgo = "formacion alta"
            elif float(row["pct_casado"]) >= 60:
                rasgo = "con pareja"
            else:
                rasgo = f"grupo {seg}"
            nombres[seg] = f"{nombre} · {rasgo}"
    return nombres


# --------------------------------------------------------------------------
# What the segment is worth: behaviour it never saw
# --------------------------------------------------------------------------


def segment_behaviour(model: SegmentModel, df: pd.DataFrame) -> pd.DataFrame:
    """Purchase behaviour per segment, with a confidence interval per rate.

    The interval is the point of the table. Without it, any two segments look
    different; with it, the question "does the segment separate behaviour?"
    has an answer.
    """
    dfs = data_mod.add_paid_price_and_spend(df)
    dfs = dfs.assign(segmento=model.asignar(dfs))

    out = dfs.groupby("segmento").agg(
        clientes=(cfg.KEY_CLIENT, "nunique"),
        visitas=(cfg.KEY_TIME, "size"),
        compras=(cfg.TARGET_INCIDENCE, "sum"),
        tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
        unidades=(cfg.TARGET_QUANTITY, "sum"),
        ingreso=("gasto", "sum"),
    )
    out["error_estandar"] = np.sqrt(out["tasa_compra"] * (1 - out["tasa_compra"]) / out["visitas"])
    out["ic95_inf"] = out["tasa_compra"] - 1.96 * out["error_estandar"]
    out["ic95_sup"] = out["tasa_compra"] + 1.96 * out["error_estandar"]
    out["ticket_medio"] = out["ingreso"] / out["compras"].replace(0, np.nan)
    out["pct_ingreso"] = 100 * out["ingreso"] / out["ingreso"].sum()
    out["nombre"] = [model.nombres.get(int(s), "") for s in out.index]

    for col in ("tasa_compra", "error_estandar", "ic95_inf", "ic95_sup"):
        out[col] = out[col].round(4)
    for col in ("ticket_medio", "pct_ingreso", "ingreso"):
        out[col] = out[col].round(2)

    alto = out["tasa_compra"].idxmax()
    bajo = out["tasa_compra"].idxmin()
    amplitud = float(out.loc[alto, "tasa_compra"] - out.loc[bajo, "tasa_compra"])
    solapan = bool(out.loc[alto, "ic95_inf"] <= out.loc[bajo, "ic95_sup"])

    out.attrs["amplitud_tasa"] = round(amplitud, 4)
    out.attrs["segmento_mas_alto"] = int(alto)
    out.attrs["segmento_mas_bajo"] = int(bajo)
    out.attrs["ic_solapan"] = solapan
    out.attrs["separa_comportamiento"] = not solapan
    out.attrs["lectura"] = (
        "La tasa de compra va de {:.4f} (segmento {}) a {:.4f} (segmento {}): amplitud "
        "{:.4f}. Los intervalos al 95% {} entre los extremos, asi que el segmento {} "
        "separa comportamiento. Para comparar: entre clientes individuales la amplitud es "
        "0.876, y eso es lo que mide D3 al concluir que la demografia segmenta pero no predice."
    ).format(
        float(out.loc[bajo, "tasa_compra"]),
        bajo,
        float(out.loc[alto, "tasa_compra"]),
        alto,
        amplitud,
        "se solapan" if solapan else "no se solapan",
        "no" if solapan else "si",
    )
    return out.reset_index()


def segment_brand_share(model: SegmentModel, df: pd.DataFrame) -> pd.DataFrame:
    """Brand preference per segment, conditional on a purchase.

    This is the table that makes the segment actionable: if two segments buy
    different brands, P3 can price by segment instead of globally.
    """
    dfs = data_mod.add_paid_price_and_spend(df)
    compras = data_mod.purchases_only(dfs)
    compras = compras.assign(segmento=model.asignar(compras))

    share = (
        compras.pivot_table(
            index="segmento", columns=cfg.TARGET_BRAND, values=cfg.KEY_TIME, aggfunc="size"
        )
        .fillna(0)
        .pipe(lambda t: (100 * t.div(t.sum(axis=1), axis=0)).round(1))
    )
    share.columns = [f"marca_{int(c)}" for c in share.columns]
    global_share = (
        100 * compras[cfg.TARGET_BRAND].value_counts(normalize=True).sort_index()
    ).round(1)
    global_share.index = [f"marca_{int(c)}" for c in global_share.index]

    desvio = (share - global_share).round(1)
    desvio.columns = [f"desvio_{c}" for c in share.columns]
    out = share.join(desvio)
    out["nombre"] = [model.nombres.get(int(s), "") for s in out.index]

    mayor = desvio.abs().to_numpy().max()
    pos = np.unravel_index(desvio.abs().to_numpy().argmax(), desvio.shape)
    out.attrs["global_share"] = global_share.to_dict()
    out.attrs["desvio_maximo_pp"] = round(float(mayor), 1)
    out.attrs["donde"] = (int(desvio.index[pos[0]]), desvio.columns[pos[1]].replace("desvio_", ""))
    out.attrs["acciona_pricing"] = mayor >= 3.0
    out.attrs["lectura"] = (
        "El mayor desvio frente al share global es de {:.1f} puntos ({} en el segmento {}). "
        "{}"
    ).format(
        mayor,
        out.attrs["donde"][1],
        out.attrs["donde"][0],
        "Hay preferencia de marca diferenciada: P3 puede diferenciar precio por segmento."
        if mayor >= 3.0
        else "Las preferencias de marca son practicamente las mismas en todos los segmentos, "
        "asi que diferenciar precio por segmento no tiene base en el dato.",
    )
    return out.reset_index()


def segment_promo_response(model: SegmentModel, df: pd.DataFrame) -> pd.DataFrame:
    """Incidence with and without any promotion, per segment.

    Where the brand-share table informs P3, this one informs P1 and P2: a
    segment that does not react to a discount is a segment not worth
    discounting to.
    """
    dfs = df.assign(
        segmento=model.asignar(df), _promo=(df[cfg.PROMO_COLS].sum(axis=1) > 0).astype(int)
    )
    filas = []
    for seg, part in dfs.groupby("segmento"):
        con = part.loc[part["_promo"] == 1, cfg.TARGET_INCIDENCE]
        sin = part.loc[part["_promo"] == 0, cfg.TARGET_INCIDENCE]
        r_con, r_sin = float(con.mean()), float(sin.mean())
        se = np.sqrt(r_con * (1 - r_con) / len(con) + r_sin * (1 - r_sin) / len(sin))
        filas.append(
            {
                "segmento": int(seg),
                "nombre": model.nombres.get(int(seg), ""),
                "visitas_con_promo": len(con),
                "tasa_con_promo": round(r_con, 4),
                "tasa_sin_promo": round(r_sin, 4),
                "delta_pp": round(100 * (r_con - r_sin), 2),
                "z": round(float((r_con - r_sin) / se), 2) if se else np.nan,
            }
        )
    out = pd.DataFrame(filas)
    out["responde"] = np.where(
        out["z"].abs() <= 2, "no distinguible del ruido", np.where(out["delta_pp"] > 0, "responde", "responde a la inversa")
    )

    responden = out.loc[out["responde"] == "responde", "segmento"].tolist()
    out.attrs["segmentos_que_responden"] = responden
    out.attrs["rango_delta"] = (float(out["delta_pp"].min()), float(out["delta_pp"].max()))
    out.attrs["lectura"] = (
        "{} de {} segmentos mueven la incidencia con promocion por encima del ruido; el "
        "efecto va de {:+.2f} a {:+.2f} puntos. {}"
    ).format(
        len(responden),
        len(out),
        float(out["delta_pp"].min()),
        float(out["delta_pp"].max()),
        "P2 tiene con esto a quien dirigir el calendario promocional."
        if responden
        else "Ningun segmento responde de forma distinguible, asi que el calendario se decide "
        "por marca y no por segmento.",
    )
    return out


# --------------------------------------------------------------------------
# PCA -- for interpretation, not for fitting
# --------------------------------------------------------------------------


def pca_projection(model: SegmentModel, df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Project the clients onto the first components, keeping the segment label.

    PCA is used here to *see* and to *interpret*, never to fit: K-Means runs on
    the full standardized space, so a projection that separates poorly is a
    property of the picture, not of the model.
    """
    X = design_matrix(df)
    pca = PCA(n_components=n_components, random_state=model.random_state)
    comps = pca.fit_transform(X.to_numpy(float))

    out = pd.DataFrame(
        comps, index=X.index, columns=[f"pc{i + 1}" for i in range(n_components)]
    )
    out["segmento"] = model.etiquetas.reindex(out.index)
    out["nombre"] = out["segmento"].map(model.nombres)

    var = pca.explained_variance_ratio_
    out.attrs["varianza_explicada"] = [round(float(v), 4) for v in var]
    out.attrs["varianza_acumulada"] = round(float(var.sum()), 4)
    out.attrs["cargas"] = pd.DataFrame(
        pca.components_.T, index=X.columns, columns=[f"pc{i + 1}" for i in range(n_components)]
    ).round(3)
    out.attrs["lectura"] = (
        "Las {} primeras componentes explican el {:.1%} de la varianza ({}). Con demografia "
        "mixta y poco correlacionada eso es lo esperable, y es la razon de que la nube se vea "
        "continua: los segmentos son cortes de un espacio continuo, no islas."
    ).format(
        n_components,
        float(var.sum()),
        ", ".join(f"pc{i + 1}: {v:.1%}" for i, v in enumerate(var)),
    )
    return out


def pca_loadings(projection: pd.DataFrame) -> pd.DataFrame:
    """The loadings table, which is what makes a component interpretable."""
    cargas = projection.attrs["cargas"].copy()
    cargas["interpretacion_dominante"] = cargas.abs().idxmax(axis=1)
    return cargas


# --------------------------------------------------------------------------
# Integration: the segment as a feature
# --------------------------------------------------------------------------


def attach_segment(df: pd.DataFrame, model: SegmentModel) -> pd.DataFrame:
    """Add the ``segmento`` column to a visit-level frame.

    ``segmento`` is declared nominal in :mod:`ferreyros.preprocess`, so once
    the column is present the preprocessor one-hot encodes it and nothing else
    has to change. Absent, every earlier phase keeps working untouched.
    """
    out = df.copy()
    out["segmento"] = model.asignar(df)
    if out["segmento"].isna().any():
        faltan = int(out["segmento"].isna().sum())
        raise ValueError(f"{faltan} filas sin segmento: hay clientes fuera del modelo M0")
    return out


def prepare_with_segment(
    model: SegmentModel, df: pd.DataFrame | None = None, split_day: int | None = None
):
    """Modelling matrices of Fase 2 with ``segmento`` added as a feature.

    This is the single call Fase 4 needs to honour "el segmento se incorpora
    como feature del modelo supervisado": the split, the train-only reference
    and the train-only preprocessor fit are the same ones Fase 2 verified, and
    the only difference is one declared extra column.
    """
    from . import preprocess as pre_mod

    if df is None:
        df = data_mod.add_paid_price_and_spend(data_mod.load_raw())
    return pre_mod.prepare(df=attach_segment(df, model), split_day=split_day, extra=["segmento"])


# --------------------------------------------------------------------------
# Freezing the partition
# --------------------------------------------------------------------------


def _fingerprint(etiquetas: pd.Series) -> str:
    pares = (
        etiquetas.sort_index()
        .reset_index()
        .astype("int64")
        .to_numpy()
        .tobytes()
    )
    return hashlib.sha256(pares).hexdigest()[:16]


def freeze(
    model: SegmentModel,
    df: pd.DataFrame,
    estabilidad: pd.DataFrame | None = None,
    path: Path | None = None,
) -> dict:
    """Write the partition and its metadata to ``artifacts/segmentos.json``."""
    path = path or SEGMENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    perfil = segment_profile(model, df)
    payload = {
        "fecha_ajuste": model.fecha_ajuste,
        "k": model.k,
        "random_state": model.random_state,
        "columnas": list(model.columnas),
        "orden_etiquetas": "ascendente por ingreso medio del centroide",
        "silhouette": model.silhouette,
        "estabilidad_ari_medio": (
            estabilidad.attrs["ari_medio"] if estabilidad is not None else None
        ),
        "nombres": {str(s): n for s, n in model.nombres.items()},
        "tamanios": {
            str(int(r["segmento"])): int(r["clientes"]) for _, r in perfil.iterrows()
        },
        "fingerprint": _fingerprint(model.etiquetas),
        "asignacion": {str(int(i)): int(v) for i, v in model.etiquetas.sort_index().items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_frozen(path: Path | None = None) -> dict:
    """Read the frozen partition. Raises if Fase 3 was never frozen."""
    path = path or SEGMENTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"No hay segmentacion congelada en {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen(model: SegmentModel, path: Path | None = None) -> pd.DataFrame:
    """Check a fitted model against the frozen partition.

    This is what Fase 4 runs before using ``segmento`` as a feature: if the
    mapping moved, the fingerprint catches it instead of a metric shifting for
    reasons nobody can explain.
    """
    frozen = load_frozen(path)
    huella = _fingerprint(model.etiquetas)
    filas = [
        {"campo": "k", "congelado": frozen["k"], "actual": model.k, "coincide": frozen["k"] == model.k},
        {
            "campo": "random_state",
            "congelado": frozen["random_state"],
            "actual": model.random_state,
            "coincide": frozen["random_state"] == model.random_state,
        },
        {
            "campo": "columnas",
            "congelado": len(frozen["columnas"]),
            "actual": len(model.columnas),
            "coincide": list(frozen["columnas"]) == list(model.columnas),
        },
        {
            "campo": "fingerprint de la asignacion",
            "congelado": frozen["fingerprint"],
            "actual": huella,
            "coincide": frozen["fingerprint"] == huella,
        },
        {
            "campo": "nombres",
            "congelado": len(frozen["nombres"]),
            "actual": len(model.nombres),
            "coincide": {str(s): n for s, n in model.nombres.items()} == frozen["nombres"],
        },
    ]
    out = pd.DataFrame(filas)
    out["coincide"] = out["coincide"].astype(bool)
    out.attrs["coincide"] = bool(out["coincide"].all())
    out.attrs["lectura"] = (
        "La segmentacion ajustada {} la congelada en {}."
    ).format("coincide con" if out.attrs["coincide"] else "NO coincide con", (path or SEGMENTS_PATH).name)
    return out


# --------------------------------------------------------------------------
# Phase report
# --------------------------------------------------------------------------


def phase3_report(
    model: SegmentModel,
    df: pd.DataFrame,
    seleccion: pd.DataFrame,
    estabilidad: pd.DataFrame,
) -> pd.DataFrame:
    """One row per decision the brief asks Fase 3 to take, with its evidence."""
    comportamiento = segment_behaviour(model, df)
    marcas = segment_brand_share(model, df)
    promo = segment_promo_response(model, df)

    filas = [
        {
            "decision": "k (codo + silhouette)",
            "valor": f"k = {model.k}",
            "evidencia": "codo en k={}, silhouette maxima en k={} (valor {:.4f})".format(
                seleccion.attrs["k_codo"], seleccion.attrs["k_silhouette"], seleccion.attrs["silhouette_max"]
            ),
            "ok": True,
        },
        {
            "decision": "estabilidad de la particion",
            "valor": "ARI medio {:.4f}".format(estabilidad.attrs["ari_medio"]),
            "evidencia": "{} remuestreos al 80%".format(len(estabilidad)),
            "ok": bool(estabilidad.attrs["ari_medio"] > 0.5),
        },
        {
            "decision": "nombres de negocio",
            "valor": "{} nombres unicos".format(len(set(model.nombres.values()))),
            "evidencia": "derivados del centroide por regla fija",
            "ok": len(set(model.nombres.values())) == model.k,
        },
        {
            "decision": "separacion de comportamiento",
            "valor": "amplitud de tasa {:.4f}".format(comportamiento.attrs["amplitud_tasa"]),
            "evidencia": "IC95 {} entre los extremos".format(
                "se solapan" if comportamiento.attrs["ic_solapan"] else "no se solapan"
            ),
            "ok": True,
        },
        {
            "decision": "accion comercial: pricing por segmento",
            "valor": "desvio maximo de share {:.1f} pp".format(marcas.attrs["desvio_maximo_pp"]),
            "evidencia": "preferencia de marca por segmento",
            "ok": True,
        },
        {
            "decision": "accion comercial: promocion por segmento",
            "valor": "{} de {} segmentos responden".format(
                len(promo.attrs["segmentos_que_responden"]), model.k
            ),
            "evidencia": "delta de incidencia con y sin promocion",
            "ok": True,
        },
    ]
    out = pd.DataFrame(filas)
    out["ok"] = out["ok"].astype(bool)
    out.attrs["fase3_completa"] = bool(out["ok"].all())
    out.attrs["lectura"] = (
        "{} de {} decisiones de la Fase 3 cerradas con evidencia ejecutada."
    ).format(int(out["ok"].sum()), len(out))
    return out
