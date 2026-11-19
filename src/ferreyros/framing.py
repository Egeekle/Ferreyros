"""Phase 0 -- framing and preparation.

Freezes the four decisions that every later phase depends on, *before* any
model is fitted:

1. **Unit of analysis** -- ``cliente x visita``, verified unique on ``(id, dia_visita)``
2. **Target variable** -- ``incidencia_compra``, with the conditional universes of M2 and M3
3. **Admissible feature set** -- with a stated reason for every exclusion
4. **Validation strategy** -- temporal split frozen at day 550, plus GroupKFold by client

"Freezing" is literal: :func:`freeze` writes the contract and a fingerprint of
the split to ``artifacts/encuadre.json``, and :func:`verify_frozen` re-checks a
later dataset against it. A silent change in the data, the split boundary or
the feature set then surfaces as a failed verification instead of as an
unexplained jump in a metric.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import config as cfg

ARTIFACTS_DIR = cfg.PROJECT_ROOT / "artifacts"
FRAMING_PATH = ARTIFACTS_DIR / "encuadre.json"


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FramingContract:
    """The frozen Phase 0 decisions.

    Frozen as a dataclass so no later phase can mutate it in place: changing a
    decision means writing a new contract and saying so.
    """

    unidad_analisis: str = "cliente x visita"
    clave_unidad: tuple[str, ...] = (cfg.KEY_CLIENT, cfg.KEY_TIME)

    target_m1: str = cfg.TARGET_INCIDENCE
    target_m2: str = cfg.TARGET_BRAND
    target_m3: str = cfg.TARGET_QUANTITY

    universo_m1: str = "todas las visitas (58,693)"
    universo_m2_m3: str = "visitas con incidencia_compra == 1 (14,638)"

    split_day: int = cfg.SPLIT_DAY
    split_tipo: str = "temporal (no aleatorio)"
    cv_estrategia: str = "TimeSeriesSplit en train + GroupKFold por id para clientes nuevos"

    metrica_principal: str = "PR-AUC"
    metricas_apoyo: tuple[str, ...] = ("ROC-AUC", "precision@k", "recall@k")
    metrica_prohibida: str = "accuracy (75.1% prediciendo siempre no-compra)"

    umbral_decision: str = "derivado del costo de contacto en P1, no fijado en 0.5"

    baseline_tasa_base: float = 0.2494
    baseline_targeting_lift: float = 2.43

    fecha_congelado: str = field(default_factory=lambda: date.today().isoformat())

    def as_table(self) -> pd.DataFrame:
        """Render the contract as a two-column table for the notebook."""
        rows = []
        for key, value in asdict(self).items():
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(v) for v in value)
            rows.append({"decision": key, "valor": value})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. Unit of analysis
# --------------------------------------------------------------------------


def verify_unit_of_analysis(df: pd.DataFrame, contract: FramingContract | None = None) -> pd.DataFrame:
    """Verify that the unit of analysis is well defined.

    If ``(id, dia_visita)`` were not unique, every per-visit probability would
    be ill-posed and the lag features would silently mix rows. This check has
    to pass before anything else in Phase 0 means something.
    """
    contract = contract or FramingContract()
    keys = list(contract.clave_unidad)

    n_rows = len(df)
    n_groups = int(df.groupby(keys).ngroups)
    n_dup_rows = int(df.duplicated(subset=keys, keep=False).sum())

    rows = [
        {
            "verificacion": f"{' x '.join(keys)} identifica una fila unica",
            "observado": f"{n_groups:,} pares para {n_rows:,} filas",
            "estado": "OK" if n_groups == n_rows else "FALLA",
        },
        {
            "verificacion": "filas en pares duplicados",
            "observado": f"{n_dup_rows:,}",
            "estado": "OK" if n_dup_rows == 0 else "FALLA",
        },
        {
            "verificacion": "cobertura temporal por cliente",
            "observado": (
                f"{df.groupby(cfg.KEY_CLIENT).size().min()}-"
                f"{df.groupby(cfg.KEY_CLIENT).size().max()} visitas por cliente"
            ),
            "estado": "OK",
        },
    ]
    out = pd.DataFrame(rows)
    out.attrs["unidad_valida"] = bool((out["estado"] == "OK").all())
    return out


# --------------------------------------------------------------------------
# 2. Target definition
# --------------------------------------------------------------------------


def target_definition(df: pd.DataFrame) -> pd.DataFrame:
    """The three targets, their universes and their class balance.

    Stating the universe per model is what prevents the most expensive mistake
    available here: training brand choice or quantity on rows where no purchase
    happened (control E12).
    """
    purchases = df[df[cfg.TARGET_INCIDENCE] == 1]

    rows = [
        {
            "modelo": "M1 incidencia",
            "target": cfg.TARGET_INCIDENCE,
            "tipo": "clasificacion binaria",
            "universo": "todas las visitas",
            "n_filas": len(df),
            "balance": f"{df[cfg.TARGET_INCIDENCE].mean():.4f} positivos",
            "parametro_que_alimenta": "pi_it",
        },
        {
            "modelo": "M2 eleccion de marca",
            "target": cfg.TARGET_BRAND,
            "tipo": "clasificacion multiclase (5)",
            "universo": "solo visitas con compra",
            "n_filas": len(purchases),
            "balance": " / ".join(
                f"{j}:{(purchases[cfg.TARGET_BRAND] == j).mean():.3f}" for j in cfg.BRANDS
            ),
            "parametro_que_alimenta": "q_ijt",
        },
        {
            "modelo": "M3 cantidad",
            "target": cfg.TARGET_QUANTITY,
            "tipo": "regresion / conteo",
            "universo": "solo visitas con compra",
            "n_filas": len(purchases),
            "balance": f"media {purchases[cfg.TARGET_QUANTITY].mean():.2f}, max {int(purchases[cfg.TARGET_QUANTITY].max())}",
            "parametro_que_alimenta": "lambda_ijt",
        },
        {
            "modelo": "M0 segmentacion",
            "target": "-- (no supervisado)",
            "tipo": "clustering",
            "universo": "500 clientes, demografia",
            "n_filas": df[cfg.KEY_CLIENT].nunique(),
            "balance": "n/a",
            "parametro_que_alimenta": "segmento_i",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Admissible feature set
# --------------------------------------------------------------------------


def admissibility_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per raw column: admissible as an M1 feature, and why or why not.

    This is the Phase 0 artifact that makes controls E1-E3 structural. A column
    is excluded here with a reason, not filtered out later by memory.
    """
    rows = []
    for col in df.columns:
        if col in cfg.EXCLUDED_COLS:
            rows.append((col, "NO", "Excluida por indicacion explicita del diccionario (E1)"))
        elif col == cfg.TARGET_INCIDENCE:
            rows.append((col, "NO", "Es el target de M1"))
        elif col in cfg.POST_OUTCOME_COLS:
            rows.append((col, "NO", "Resultado de la visita, no condicion previa (E2/E3) - fuga"))
        elif col == cfg.KEY_CLIENT:
            rows.append((col, "NO", "Clave de agrupacion; identidad no generaliza a clientes nuevos"))
        elif col == cfg.KEY_TIME:
            rows.append((col, "DERIVADA", "No se usa cruda; alimenta features temporales y el split"))
        elif col in cfg.PRICE_COLS:
            rows.append((col, "SI", "Condicion comercial conocida antes de la visita; decision y_jtk"))
        elif col in cfg.PROMO_COLS:
            rows.append((col, "SI", "Condicion comercial conocida antes de la visita; decision z_jt"))
        elif col == "ultima_marca_comprada":
            rows.append((col, "SI", "Lag verificado contra shift(1) por cliente (E4)"))
        elif col == "ultima_cantidad_comprada":
            rows.append((col, "SI", "Lag binario del indicador de compra; el diccionario la nombra mal (E5)"))
        elif col in cfg.DEMOGRAPHIC_COLS:
            rows.append((col, "SI", "Estatica por cliente (D2); segmenta mas de lo que predice (D3)"))
        else:
            rows.append((col, "REVISAR", "No clasificada por el contrato de encuadre"))

    out = pd.DataFrame(rows, columns=["columna", "admisible", "razon"])
    out.attrs["n_admisibles"] = int((out["admisible"] == "SI").sum())
    out.attrs["n_excluidas"] = int((out["admisible"] == "NO").sum())
    out.attrs["sin_clasificar"] = int((out["admisible"] == "REVISAR").sum())
    return out


def admissible_raw_columns(df: pd.DataFrame) -> list[str]:
    """The raw columns the contract admits as M1 features."""
    table = admissibility_table(df)
    return table.loc[table["admisible"] == "SI", "columna"].tolist()


def forbidden_columns() -> set[str]:
    """Columns that must never appear in the M1 feature matrix."""
    return {
        cfg.TARGET_INCIDENCE,
        *cfg.POST_OUTCOME_COLS,
        *cfg.EXCLUDED_COLS,
        cfg.KEY_CLIENT,
        "precio_pagado",
        "gasto",
    }


def assert_feature_matrix_admissible(columns) -> pd.DataFrame:
    """Check a built feature matrix against the contract.

    Raises nothing; returns the verdict so the notebook can display it and
    assert on ``attrs['admisible']``.
    """
    present = set(columns)
    violations = sorted(present & forbidden_columns())
    out = pd.DataFrame(
        [
            {
                "verificacion": "ninguna columna prohibida en la matriz de M1",
                "n_features": len(present),
                "violaciones": len(violations),
                "detalle": ", ".join(violations) if violations else "ninguna",
                "estado": "OK" if not violations else "FALLA",
            }
        ]
    )
    out.attrs["admisible"] = not violations
    return out


# --------------------------------------------------------------------------
# 4. Validation strategy
# --------------------------------------------------------------------------


def validation_plan(contract: FramingContract | None = None) -> pd.DataFrame:
    """The validation strategy, decided now and not negotiable later.

    Each row states the decision and the reason it is forced by this dataset's
    structure rather than chosen by preference.
    """
    contract = contract or FramingContract()
    rows = [
        {
            "aspecto": "Tipo de split",
            "decision": contract.split_tipo,
            "razon": (
                "Con ~117 visitas por cliente, un split aleatorio reparte visitas del mismo "
                "cliente entre train y test e infla todas las metricas de forma irreparable."
            ),
        },
        {
            "aspecto": "Frontera del split",
            "decision": f"train: dia <= {contract.split_day} | test: dia > {contract.split_day}",
            "razon": "Deja ~180 dias de test (25%) conservando los 500 clientes y la tasa base.",
        },
        {
            "aspecto": "Validacion cruzada en train",
            "decision": contract.cv_estrategia,
            "razon": (
                "TimeSeriesSplit respeta el orden temporal para el tuning; GroupKFold por id "
                "mide generalizacion a clientes nuevos, que con 500 clientes es el riesgo real."
            ),
        },
        {
            "aspecto": "Metrica principal",
            "decision": contract.metrica_principal,
            "razon": "Desbalance 25/75: PR-AUC es sensible a la clase positiva, que es la que importa.",
        },
        {
            "aspecto": "Metrica prohibida",
            "decision": contract.metrica_prohibida,
            "razon": "Predecir siempre 'no compra' da 75.1% de accuracy y cero valor de negocio.",
        },
        {
            "aspecto": "Umbral de decision",
            "decision": contract.umbral_decision,
            "razon": (
                "El umbral optimo iguala margen esperado y costo de contacto: lo fija la "
                "optimizacion P1, no la convencion."
            ),
        },
        {
            "aspecto": "Baselines a superar",
            "decision": (
                f"tasa base {contract.baseline_tasa_base} | "
                f"targeting historico lift x{contract.baseline_targeting_lift}"
            ),
            "razon": "Un modelo que no bate el ordenamiento por tasa historica no justifica su existencia.",
        },
        {
            "aspecto": "Preprocesamiento",
            "decision": "Pipeline de scikit-learn con fit solo sobre train (E10)",
            "razon": "Ajustar escalado o codificacion sobre todo el dataset filtra informacion del test.",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Split freezing
# --------------------------------------------------------------------------


def split_manifest(df: pd.DataFrame, contract: FramingContract | None = None) -> dict:
    """Describe and fingerprint the frozen split.

    The fingerprint is a hash of the sorted ``(id, dia_visita)`` pairs on each
    side. It changes if the data changes or the boundary moves, which is
    exactly what we want to detect.
    """
    contract = contract or FramingContract()
    day = contract.split_day
    keys = list(contract.clave_unidad)

    manifest = {"split_day": day}
    for name, part in (
        ("train", df[df[cfg.KEY_TIME] <= day]),
        ("test", df[df[cfg.KEY_TIME] > day]),
    ):
        pairs = part[keys].sort_values(keys).astype("int64").to_numpy().tobytes()
        manifest[name] = {
            "filas": int(len(part)),
            "clientes": int(part[cfg.KEY_CLIENT].nunique()),
            "dia_min": int(part[cfg.KEY_TIME].min()),
            "dia_max": int(part[cfg.KEY_TIME].max()),
            "tasa_compra": round(float(part[cfg.TARGET_INCIDENCE].mean()), 4),
            "fingerprint": hashlib.sha256(pairs).hexdigest()[:16],
        }
    manifest["balanceado"] = bool(
        abs(manifest["train"]["tasa_compra"] - manifest["test"]["tasa_compra"]) < 0.01
        and manifest["train"]["clientes"] == manifest["test"]["clientes"]
    )
    return manifest


def split_manifest_table(df: pd.DataFrame, contract: FramingContract | None = None) -> pd.DataFrame:
    """The split manifest as a table for display."""
    manifest = split_manifest(df, contract)
    rows = []
    for name in ("train", "test"):
        row = {"split": name, **manifest[name]}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["split_day"] = manifest["split_day"]
    out.attrs["balanceado"] = manifest["balanceado"]
    return out


# --------------------------------------------------------------------------
# Freeze and verify
# --------------------------------------------------------------------------


def freeze(
    df: pd.DataFrame,
    contract: FramingContract | None = None,
    path: Path | None = None,
) -> dict:
    """Write the framing contract and split fingerprint to disk.

    Returns the frozen payload. Call once at the end of Phase 0; later phases
    load it instead of re-deciding.
    """
    contract = contract or FramingContract()
    path = path or FRAMING_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    admissible = admissibility_table(df)
    payload = {
        "contrato": asdict(contract),
        "unidad_analisis": {
            "valida": bool(verify_unit_of_analysis(df, contract).attrs["unidad_valida"]),
            "filas": int(len(df)),
            "clientes": int(df[cfg.KEY_CLIENT].nunique()),
            "dias": int(df[cfg.KEY_TIME].nunique()),
        },
        "features_admisibles": {
            "columnas_crudas_admisibles": admissible_raw_columns(df),
            "n_admisibles": admissible.attrs["n_admisibles"],
            "n_excluidas": admissible.attrs["n_excluidas"],
            "prohibidas": sorted(forbidden_columns()),
        },
        "split": split_manifest(df, contract),
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_frozen(path: Path | None = None) -> dict:
    """Load the frozen framing contract."""
    path = path or FRAMING_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No hay encuadre congelado en {path}. Ejecutar framing.freeze(df) en la Fase 0."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Re-check a dataset against the frozen contract.

    Run this at the top of every later phase. A mismatch means the data, the
    boundary or the feature contract moved, and the models trained before are
    no longer comparable with the ones trained after.
    """
    frozen = load_frozen(path)
    current = split_manifest(df, FramingContract(**{
        k: tuple(v) if isinstance(v, list) else v
        for k, v in frozen["contrato"].items()
    }))

    rows = [
        {
            "verificacion": "frontera del split",
            "congelado": frozen["split"]["split_day"],
            "actual": current["split_day"],
            "estado": "OK" if frozen["split"]["split_day"] == current["split_day"] else "FALLA",
        },
        {
            "verificacion": "filas totales",
            "congelado": frozen["unidad_analisis"]["filas"],
            "actual": len(df),
            "estado": "OK" if frozen["unidad_analisis"]["filas"] == len(df) else "FALLA",
        },
        {
            "verificacion": "clientes",
            "congelado": frozen["unidad_analisis"]["clientes"],
            "actual": int(df[cfg.KEY_CLIENT].nunique()),
            "estado": "OK" if frozen["unidad_analisis"]["clientes"] == df[cfg.KEY_CLIENT].nunique() else "FALLA",
        },
    ]
    for name in ("train", "test"):
        rows.append(
            {
                "verificacion": f"fingerprint {name}",
                "congelado": frozen["split"][name]["fingerprint"],
                "actual": current[name]["fingerprint"],
                "estado": "OK" if frozen["split"][name]["fingerprint"] == current[name]["fingerprint"] else "FALLA",
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["coincide"] = bool((out["estado"] == "OK").all())
    return out


# --------------------------------------------------------------------------
# Closing the loop: does the modelling phase obey the frozen contract?
# --------------------------------------------------------------------------


def verify_prepared_against_contract(prepared, path: Path | None = None) -> pd.DataFrame:
    """Check a :class:`ferreyros.preprocess.PreparedData` against the contract.

    Phase 0 freezes the decisions; Phase 2 builds the matrices. This function
    is what makes the link mechanical instead of a matter of trust: it verifies
    that the split boundary, the row counts, the client partition and the
    feature admissibility of the prepared data are the ones that were frozen.
    """
    frozen = load_frozen(path)
    split = frozen["split"]
    forbidden = set(frozen["features_admisibles"]["prohibidas"])

    input_cols = set(prepared.X_train.columns)
    violations = sorted(input_cols & forbidden)

    rows = [
        {
            "verificacion": "frontera del split",
            "congelado": split["split_day"],
            "actual": prepared.split_day,
            "estado": "OK" if split["split_day"] == prepared.split_day else "FALLA",
        },
        {
            "verificacion": "filas de train",
            "congelado": split["train"]["filas"],
            "actual": len(prepared.X_train),
            "estado": "OK" if split["train"]["filas"] == len(prepared.X_train) else "FALLA",
        },
        {
            "verificacion": "filas de test",
            "congelado": split["test"]["filas"],
            "actual": len(prepared.X_test),
            "estado": "OK" if split["test"]["filas"] == len(prepared.X_test) else "FALLA",
        },
        {
            "verificacion": "tasa de compra en train",
            "congelado": split["train"]["tasa_compra"],
            "actual": round(float(prepared.y_train.mean()), 4),
            "estado": "OK"
            if abs(split["train"]["tasa_compra"] - float(prepared.y_train.mean())) < 1e-4
            else "FALLA",
        },
        {
            "verificacion": "tasa de compra en test",
            "congelado": split["test"]["tasa_compra"],
            "actual": round(float(prepared.y_test.mean()), 4),
            "estado": "OK"
            if abs(split["test"]["tasa_compra"] - float(prepared.y_test.mean())) < 1e-4
            else "FALLA",
        },
        {
            "verificacion": "clientes presentes en ambos lados",
            "congelado": split["train"]["clientes"],
            "actual": int(prepared.groups_train.nunique()),
            "estado": "OK" if split["train"]["clientes"] == prepared.groups_train.nunique() else "FALLA",
        },
        {
            "verificacion": "ninguna columna prohibida en la matriz",
            "congelado": f"{len(forbidden)} prohibidas",
            "actual": ", ".join(violations) if violations else "ninguna",
            "estado": "OK" if not violations else "FALLA",
        },
    ]

    out = pd.DataFrame(rows)
    out.attrs["obedece_contrato"] = bool((out["estado"] == "OK").all())
    return out


# --------------------------------------------------------------------------
# Consolidated Phase 0 report
# --------------------------------------------------------------------------


def phase0_report(df: pd.DataFrame, contract: FramingContract | None = None) -> pd.DataFrame:
    """One row per Phase 0 decision, with its verification status."""
    contract = contract or FramingContract()
    unit = verify_unit_of_analysis(df, contract)
    adm = admissibility_table(df)
    split = split_manifest(df, contract)

    rows = [
        {
            "decision": "1. Unidad de analisis",
            "definicion": f"{contract.unidad_analisis} ({' x '.join(contract.clave_unidad)})",
            "estado": "OK" if unit.attrs["unidad_valida"] else "FALLA",
        },
        {
            "decision": "2. Variable objetivo",
            "definicion": f"{contract.target_m1} (M1); {contract.target_m2}/{contract.target_m3} condicionados a compra",
            "estado": "OK",
        },
        {
            "decision": "3. Features admisibles",
            "definicion": f"{adm.attrs['n_admisibles']} columnas crudas admisibles, {adm.attrs['n_excluidas']} excluidas con razon",
            "estado": "OK" if adm.attrs["sin_clasificar"] == 0 else "REVISAR",
        },
        {
            "decision": "4. Estrategia de validacion",
            "definicion": f"{contract.split_tipo}, corte dia {contract.split_day}; {contract.metrica_principal}",
            "estado": "OK" if split["balanceado"] else "REVISAR",
        },
    ]
    out = pd.DataFrame(rows)
    out.attrs["fase0_completa"] = bool((out["estado"] == "OK").all())
    return out
