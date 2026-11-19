#!/usr/bin/env python
"""Batch scoring of the portfolio -- the nightly job of Fase 6.

Loads the frozen contracts, scores every visit of the horizon with M1/M2/M3,
solves P1 for the given budget and writes the contact list the CRM consumes.

    uv run python scripts/score_batch.py
    uv run python scripts/score_batch.py --presupuesto 0.10 --margen 0.30 --fatiga 8

The job refuses to run if the frozen contracts do not match the data. That is
the point: a contact list built on a contract that moved is worse than no list,
because nobody can tell which one it was built on.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ferreyros import config as cfg  # noqa: E402
from ferreyros import data as data_mod  # noqa: E402
from ferreyros import framing  # noqa: E402
from ferreyros import prescriptive as px  # noqa: E402
from ferreyros import segmentation as seg_mod  # noqa: E402


def verificar_contratos(df, modelo_m0) -> dict:
    """Fail closed: every frozen contract has to match before scoring."""
    encuadre = framing.verify_frozen(df)
    if not encuadre.attrs["coincide"]:
        raise SystemExit("ABORTA: el dataset no coincide con artifacts/encuadre.json")

    segmentos = seg_mod.verify_frozen(modelo_m0)
    if not segmentos.attrs["coincide"]:
        raise SystemExit("ABORTA: la segmentacion no coincide con artifacts/segmentos.json")

    ruta = cfg.PROJECT_ROOT / "artifacts" / "modelos.json"
    metricas = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    return {
        "encuadre": "OK",
        "segmentos": "OK",
        "modelos_json": metricas.get("fecha", "ausente"),
        "pr_auc_referencia": metricas.get("m1", {}).get("pr_auc_test"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Scoring batch de la cartera (Fase 6)")
    p.add_argument("--presupuesto", type=float, default=0.10,
                   help="fraccion de visitas contactables (default 0.10)")
    p.add_argument("--margen", type=float, default=0.30,
                   help="margen bruto supuesto (default 0.30)")
    p.add_argument("--fatiga", type=int, default=8,
                   help="maximo de contactos por cliente en el horizonte (default 8)")
    p.add_argument("--salida", type=Path, default=None, help="ruta del CSV de salida")
    args = p.parse_args()

    print("== Scoring batch · Ferreycorp ==")
    df = data_mod.add_paid_price_and_spend(data_mod.load_raw())
    modelo_m0 = seg_mod.fit(df, k=3)

    estado = verificar_contratos(df, modelo_m0)
    print("contratos congelados:", estado)

    supuesto = px.MarginAssumption(margen_bruto=args.margen)
    bundle = px.fit_bundle(df, modelo_m0)
    scored = px.score_scenario(bundle, df, supuesto)

    a = supuesto.costo_contacto(df, float(scored["lambda"].mean()))
    n = int(args.presupuesto * len(scored))
    p1 = px.solve_p1(scored, a, n, fatiga=args.fatiga)

    salida = scored.copy()
    salida["contactar"] = 0
    salida.loc[p1.index, "contactar"] = 1
    salida["prioridad"] = salida["margen_mejor_marca"].rank(ascending=False, method="first")
    columnas = [
        cfg.KEY_CLIENT, cfg.KEY_TIME, "segmento", "pi", "lambda", "mejor_marca",
        "margen_mejor_marca", "margen_total", "contactar", "prioridad",
    ]

    ruta = args.salida or (
        cfg.PROJECT_ROOT / "artifacts" / f"scoring_{date.today().isoformat()}.csv"
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida[columnas].sort_values("prioridad").to_csv(ruta, index=False)

    print(f"\nvisitas puntuadas: {len(salida):,}")
    print(f"contactos seleccionados: {p1.attrs['contactos']:,} "
          f"({p1.attrs['contactos'] / len(salida):.1%}) en {p1.attrs['clientes_distintos']} clientes")
    print(f"objetivo neto esperado: {p1.attrs['objetivo']:.2f} unidades de margen")
    print(f"marcas ofrecidas: {p1.attrs['marcas_ofrecidas']}")
    print(f"escrito: {ruta.relative_to(cfg.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
