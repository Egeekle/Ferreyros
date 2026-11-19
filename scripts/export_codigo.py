#!/usr/bin/env python
"""Concatenate the whole solution into ``entregables/codigo_solucion.txt``.

The brief accepts Word or TXT. TXT, because it needs no library to produce, no
library to open, and it diffs.

    uv run python scripts/export_codigo.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "entregables" / "codigo_solucion.txt"

ORDEN = [
    ("Configuracion y contrato de datos", ["src/ferreyros/config.py", "src/ferreyros/data.py"]),
    ("Fase 0 · Encuadre congelado", ["src/ferreyros/framing.py"]),
    ("Fase 1 · EDA", [
        "src/ferreyros/eda/__init__.py", "src/ferreyros/eda/objective.py",
        "src/ferreyros/eda/decisions.py", "src/ferreyros/eda/constraints.py",
        "src/ferreyros/eda/segments.py", "src/ferreyros/eda/dynamics.py",
        "src/ferreyros/eda/anomalies.py", "src/ferreyros/audit.py",
    ]),
    ("Fase 2 · Features y preprocesamiento", [
        "src/ferreyros/features.py", "src/ferreyros/preprocess.py",
    ]),
    ("Fase 3 · Segmentacion (M0)", ["src/ferreyros/segmentation.py"]),
    ("Fase 4 · Modelos M1, M2, M3", ["src/ferreyros/models.py"]),
    ("Fase 5 · Capa prescriptiva (P1, P2, P3)", ["src/ferreyros/prescriptive.py"]),
    ("Fase 6 · Despliegue y scoring", [
        "scripts/score_batch.py", "scripts/serve_api.py",
    ]),
    ("Figuras, dashboard y paquete", [
        "src/ferreyros/plots.py", "src/ferreyros/dashboard.py", "src/ferreyros/__init__.py",
        "scripts/export_codigo.py", "scripts/build_entregables.py",
    ]),
]

NOTEBOOKS = [
    "notebooks/00_encuadre.ipynb", "notebooks/01_eda.ipynb", "notebooks/02_procesamiento.ipynb",
    "notebooks/03_segmentacion.ipynb", "notebooks/04_modelos.ipynb", "notebooks/05_pricing.ipynb",
]


def barra(txt: str, char: str = "=") -> str:
    return f"{char * 78}\n{txt}\n{char * 78}\n"


def main() -> None:
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    partes: list[str] = []
    indice: list[str] = []

    partes.append(barra("PRUEBA TECNICA DATA SCIENTIST · FERREYCORP\nMODELO DE PROPENSION DE COMPRA — CODIGO DE LA SOLUCION"))
    partes.append(
        f"Generado: {date.today().isoformat()}\n"
        "Repositorio: estructura completa en el ZIP / GitHub adjunto\n\n"
        "El codigo esta organizado por fases. La logica vive en modulos de 'src/ferreyros/';\n"
        "los notebooks solo orquestan, muestran y concluyen. Esa separacion es deliberada:\n"
        "un notebook que contiene la logica no se puede probar ni reutilizar en produccion.\n\n"
        "Para ejecutar todo de principio a fin:\n"
        "    uv sync\n"
        "    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb\n"
        "    uv run python scripts/score_batch.py\n"
    )

    cuerpo: list[str] = []
    for seccion, rutas in ORDEN:
        cuerpo.append("\n" + barra(seccion))
        for ruta in rutas:
            f = ROOT / ruta
            if not f.exists():
                continue
            texto = f.read_text(encoding="utf-8")
            lineas = texto.count("\n") + 1
            indice.append(f"  {ruta:<42} {lineas:>5} lineas")
            cuerpo.append(barra(f"ARCHIVO: {ruta}  ({lineas} lineas)", "-"))
            cuerpo.append(texto.rstrip() + "\n")

    cuerpo.append("\n" + barra("NOTEBOOKS · codigo de las celdas ejecutadas"))
    for ruta in NOTEBOOKS:
        f = ROOT / ruta
        if not f.exists():
            continue
        nb = json.loads(f.read_text(encoding="utf-8"))
        celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
        indice.append(f"  {ruta:<42} {len(celdas):>5} celdas de codigo")
        cuerpo.append(barra(f"NOTEBOOK: {ruta}  ({len(celdas)} celdas de codigo)", "-"))
        for i, c in enumerate(celdas, 1):
            cuerpo.append(f"# --- celda {i} ---\n{''.join(c['source']).rstrip()}\n")

    partes.append("\n" + barra("INDICE", "-"))
    partes.append("\n".join(indice) + "\n")
    partes.extend(cuerpo)

    SALIDA.write_text("\n".join(partes), encoding="utf-8")
    kb = SALIDA.stat().st_size / 1024
    print(f"escrito {SALIDA.relative_to(ROOT)} · {kb:.0f} KB · {len(indice)} archivos")


if __name__ == "__main__":
    main()
