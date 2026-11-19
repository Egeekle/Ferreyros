"""Ferreyros -- purchase-propensity solution for the Ferreycorp technical test.

Layered on the documents in ``docs/``:

* ``brief-problema.md``        -- the optimization problem and the ML layer that feeds it
* ``brief-alineamiento-eda.md`` -- the matrix this package's functions satisfy
* ``brief-metodologia.md``     -- the end-to-end methodology

Phase 0 (framing) freezes the analysis contract in ``artifacts/encuadre.json``;
later phases call ``framing.verify_frozen(df)`` before doing anything else.

Typical notebook entry point::

    from ferreyros import config as cfg, data, features, plots, preprocess
    from ferreyros.audit import control_cell, treatment_audit
    from ferreyros.eda import objective, decisions, constraints, segments

    df = data.load_raw()
    control_cell(df)
    framing.freeze(df)                # Fase 0: congela el contrato de analisis
    prepared = preprocess.prepare()   # matrices de modelado, fit solo en train
"""

from . import (
    audit,
    config,
    dashboard,
    data,
    eda,
    features,
    framing,
    models,
    plots,
    prescriptive,
    preprocess,
    segmentation,
)
from .data import add_paid_price_and_spend, load_raw, purchases_only, temporal_split
from .preprocess import prepare

__all__ = [
    "config",
    "dashboard",
    "data",
    "eda",
    "features",
    "framing",
    "preprocess",
    "segmentation",
    "models",
    "prescriptive",
    "plots",
    "audit",
    "load_raw",
    "purchases_only",
    "add_paid_price_and_spend",
    "temporal_split",
    "prepare",
]


def main() -> None:
    """Smoke entry point: load the data and print the control cell verdict."""
    df = load_raw()
    report = audit.control_cell(df)
    ok = report.attrs["all_ok"]
    print(f"Celda de control: {'OK' if ok else str(report.attrs['n_fallas']) + ' discrepancias'}")
    if not ok:
        print(report[report["veredicto"] != "OK"].to_string(index=False))
