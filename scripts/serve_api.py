#!/usr/bin/env python
"""Scoring endpoint -- the point-query side of Fase 6.

Written on the standard library (``http.server``) on purpose: a FastAPI stub
that nobody can run proves nothing, and this starts with no extra dependency.

    uv run python scripts/serve_api.py
    curl "http://127.0.0.1:8765/health"
    curl "http://127.0.0.1:8765/score?id=200000247"
    curl "http://127.0.0.1:8765/plan?presupuesto=0.10"

Every ``/score`` answer carries the three factors that explain it. A number
without its explanation produces a phone call to the data team the first time a
salesperson disagrees with it.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ferreyros import config as cfg  # noqa: E402
from ferreyros import data as data_mod  # noqa: E402
from ferreyros import features as feat_mod  # noqa: E402
from ferreyros import framing  # noqa: E402
from ferreyros import models as mdl  # noqa: E402
from ferreyros import prescriptive as px  # noqa: E402
from ferreyros import segmentation as seg_mod  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
ESTADO: dict = {}


def arrancar() -> None:
    """Load everything once: contracts, models, scores and SHAP explanations."""
    print("cargando contratos y modelos...", flush=True)
    df = data_mod.add_paid_price_and_spend(data_mod.load_raw())
    modelo_m0 = seg_mod.fit(df, k=3)

    if not framing.verify_frozen(df).attrs["coincide"]:
        raise SystemExit("ABORTA: el dataset no coincide con artifacts/encuadre.json")
    if not seg_mod.verify_frozen(modelo_m0).attrs["coincide"]:
        raise SystemExit("ABORTA: la segmentacion no coincide con artifacts/segmentos.json")

    supuesto = px.MarginAssumption()
    bundle = px.fit_bundle(df, modelo_m0)
    scored = px.score_scenario(bundle, df, supuesto)

    dfs = seg_mod.attach_segment(df, modelo_m0)
    featured = feat_mod.build_features(dfs, reference=bundle.reference)
    _, horizonte = data_mod.temporal_split(featured, bundle.split_day)
    X = feat_mod.feature_matrix(horizonte, extra=bundle.extras)[bundle.columnas]

    print("calculando SHAP del horizonte...", flush=True)
    # shap_values devuelve las filas en el orden de SU muestra, no de X: indexar
    # con X.index entregaria la explicacion de otra visita.
    valores, nombres, muestra = mdl.shap_values(bundle.pipe_m1, X, n_sample=len(X))

    ESTADO.update(
        {
            "df": df,
            "supuesto": supuesto,
            "bundle": bundle,
            "scored": scored,
            "shap": pd.DataFrame(valores, index=muestra.index, columns=nombres),
            "costo_contacto": supuesto.costo_contacto(df, float(scored["lambda"].mean())),
        }
    )
    print(
        f"listo: {len(scored):,} visitas del horizonte "
        f"(dias {int(scored[cfg.KEY_TIME].min())}-{int(scored[cfg.KEY_TIME].max())})",
        flush=True,
    )


def responder_score(cliente: int) -> dict:
    scored = ESTADO["scored"]
    filas = scored[scored[cfg.KEY_CLIENT] == cliente]
    if filas.empty:
        return {"error": f"cliente {cliente} sin visitas en el horizonte"}

    fila = filas.sort_values(cfg.KEY_TIME).iloc[-1]
    shap_fila = ESTADO["shap"].loc[fila.name]
    top = shap_fila.abs().sort_values(ascending=False).head(3).index
    a = ESTADO["costo_contacto"]

    return {
        "cliente": int(cliente),
        "dia_visita": int(fila[cfg.KEY_TIME]),
        "segmento": int(fila["segmento"]),
        "probabilidad_compra": round(float(fila["pi"]), 4),
        "cantidad_esperada": round(float(fila["lambda"]), 3),
        "mejor_marca": int(fila["mejor_marca"]),
        "margen_esperado": round(float(fila["margen_mejor_marca"]), 4),
        "costo_de_contacto": round(float(a), 4),
        "decision": "contactar" if fila["margen_mejor_marca"] > a else "no contactar",
        "por_que": [
            {"factor": f, "contribucion": round(float(shap_fila[f]), 4)} for f in top
        ],
        "contrato": {
            "split_day": cfg.SPLIT_DAY,
            "margen_bruto_supuesto": ESTADO["supuesto"].margen_bruto,
            "aviso": "margen en unidades de margen supuesto; el dataset no contiene costos",
        },
    }


def responder_plan(presupuesto: float) -> dict:
    scored, a = ESTADO["scored"], ESTADO["costo_contacto"]
    n = int(presupuesto * len(scored))
    p1 = px.solve_p1(scored, a, n, fatiga=8)
    return {
        "presupuesto_pct_visitas": round(100 * presupuesto, 1),
        "tope_contactos": n,
        "contactos": p1.attrs["contactos"],
        "clientes": p1.attrs["clientes_distintos"],
        "objetivo_neto": p1.attrs["objetivo"],
        "marcas_ofrecidas": {str(k): int(v) for k, v in p1.attrs["marcas_ofrecidas"].items()},
        "top_10": [
            {
                "cliente": int(r[cfg.KEY_CLIENT]),
                "dia": int(r[cfg.KEY_TIME]),
                "marca": int(r["mejor_marca"]),
                "margen_esperado": round(float(r["margen_mejor_marca"]), 4),
            }
            for _, r in p1.nlargest(10, "margen_mejor_marca").iterrows()
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def _enviar(self, payload: dict, codigo: int = 200) -> None:
        cuerpo = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self) -> None:  # noqa: N802
        ruta = urlparse(self.path)
        params = parse_qs(ruta.query)
        try:
            if ruta.path == "/health":
                self._enviar(
                    {
                        "estado": "ok",
                        "visitas_en_memoria": int(len(ESTADO["scored"])),
                        "modelos": ESTADO["bundle"].resumen.to_dict(),
                    }
                )
            elif ruta.path == "/score":
                if "id" not in params:
                    self._enviar({"error": "falta el parametro id"}, 400)
                    return
                self._enviar(responder_score(int(params["id"][0])))
            elif ruta.path == "/plan":
                presupuesto = float(params.get("presupuesto", ["0.10"])[0])
                self._enviar(responder_plan(presupuesto))
            else:
                self._enviar(
                    {"error": "ruta desconocida", "rutas": ["/health", "/score?id=", "/plan?presupuesto="]},
                    404,
                )
        except Exception as exc:  # noqa: BLE001
            self._enviar({"error": type(exc).__name__, "detalle": str(exc)}, 500)

    def log_message(self, formato: str, *args) -> None:
        print(f"  {self.address_string()} {formato % args}", flush=True)


if __name__ == "__main__":
    arrancar()
    servidor = HTTPServer((HOST, PORT), Handler)
    print(f"escuchando en http://{HOST}:{PORT}  (ctrl-c para parar)", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nservidor detenido")
