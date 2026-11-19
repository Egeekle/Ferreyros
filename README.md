# Modelo de Propensión de Compra · Ferreycorp

Prueba técnica para Data Scientist. **58 693 visitas, 500 clientes, 730 días, 5 marcas** →
tres decisiones ejecutables: a quién contactar, qué marca promocionar y a qué precio vender.

> El modelo de propensión es el punto de partida, no el entregable. Predecir quién compra no dice
> a quién contactar con un presupuesto limitado. Por eso la solución se estructuró primero como un
> **problema de optimización** y el machine learning existe para estimar sus parámetros.

---

## Ejecutar de principio a fin

```bash
uv sync

# Los 6 notebooks, en orden. Cada uno corre de extremo a extremo sin intervención.
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_encuadre.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_procesamiento.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_segmentacion.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelos.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_pricing.ipynb

# Producción (Fase 6)
uv run python scripts/score_batch.py          # lista de contacto → artifacts/scoring_<fecha>.csv
uv run python scripts/serve_api.py            # endpoint de scoring puntual en :8765

# Dashboard interactivo — se abre con doble clic, sin servidor
open dashboard/index.html
```

El dato vive bajo DVC (`data.dvc`). Si falta, `dvc pull` o copiar `compras_data.csv` en `data/`.

---

## Qué hay aquí

| Carpeta | Contenido |
|---|---|
| `notebooks/` | seis notebooks ejecutables, uno por fase; orquestan y concluyen, no contienen lógica |
| `src/ferreyros/` | 13 módulos con toda la lógica: datos, EDA, features, modelos, optimización, figuras |
| `scripts/` | scoring batch, endpoint REST y generadores de los entregables |
| `artifacts/` | **cuatro contratos congelados** con huella SHA-256 (encuadre, segmentos, modelos, prescriptivo) |
| `dashboard/` | página interactiva autocontenida (`index.html` + `data.js`) |
| `docs/` | briefs de metodología, problema y alineamiento + arquitectura de despliegue + figuras |
| `entregables/` | código en TXT, documentación PDF y presentación de 7 slides |
| `odd/tasks/` | bitácora de cada fase: specs, tareas y decisiones con su evidencia |

---

## Las seis fases

| Fase | Notebook | Qué cierra | Cómo se verificó |
|---|---|---|---|
| **0 · Encuadre** | `00_encuadre` | unidad de análisis, targets, features admisibles, split del día 550 | contrato congelado con huella; `verify_frozen` corre al inicio de cada fase |
| **1 · EDA** | `01_eda` | qué parámetros del problema son estimables y con qué forma | 44 cifras de referencia reproducidas · 13 controles de tratamiento |
| **2 · Features** | `02_procesamiento` | 68 variables derivadas sin fuga + `Pipeline` | control E10: el ajuste solo en train cambia 48 parámetros del escalador |
| **3 · Segmentación** | `03_segmentacion` | k = 3 segmentos, congelados | elegido por codo y **estabilidad** (ARI 1.0 vs 0.79 en k=9) |
| **4 · Modelos** | `04_modelos` | M1 logística · M2 random forest · M3 Poisson | comparación pareada fold a fold; validación con cliente nuevo **y** día futuro |
| **5 · Prescriptiva** | `05_pricing` | P1 targeting · P2 calendario · P3 precio | optimalidad de P1 verificada contra el LP; P2 y P3 contra simulación conjunta |
| **6 · Despliegue** | — | arquitectura multi-cloud, PoC y monitoreo | `score_batch.py` y `serve_api.py` ejecutables |

---

## Resultados

| Métrica | Valor |
| --- | --- |
| PR-AUC de M1 en test | **0.4776** (azar 0.2480) |
| Lift del decil superior | **×2.50** (política vigente ×2.46) |
| Ingreso capturado con el 10% de las visitas | **31.7%** (política vigente 28.2%) |
| P1 · targeting sobre la política vigente | **+205.32** unidades de margen |
| P2 · reasignar los mismos 125 días-promoción | **+769.06** |
| P3 · precio con banda ±10% y R7 | **+579.68** |

### Hallazgos que cambian decisiones

- **El calendario promocional actual destruye margen** (−183.84 en el horizonte de test).
  Reasignarlo, sin presupuesto nuevo, vale +769.06.
- **Contactar al azar deja objetivo neto negativo:** el costo de contacto supera el margen
  esperado de una visita promedio.
- **`GroupKFold` por cliente no es el límite pesimista** en un panel: infla el PR-AUC en 0.24
  porque entrena con los mismos días. Todo se decidió cruzando cliente nuevo y día futuro.
- **La demografía segmenta pero no predice:** 8 puntos de tasa de compra, 23.6 puntos de
  preferencia de marca.

### Límites declarados, no escondidos

- No hay costos en el dataset: todo está en unidades de margen bajo un supuesto del 30%, con
  sensibilidad entre 20% y 40%. Las cifras se mueven; la decisión no.
- La elasticidad propia de la marca 3 **no está identificada** (positiva en las cinco
  especificaciones): P3 congela su precio y pide un test A/B.
- Cuatro de los cinco KPIs son **proyecciones** hasta que exista grupo de control.

---

## Regenerar los entregables

```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from ferreyros import dashboard; dashboard.export_all()"
uv run python scripts/export_codigo.py
uv run python scripts/build_entregables.py
```

Las cifras de la documentación, del deck y del dashboard salen todas de `artifacts/` vía
`dashboard/data.json`: hay una sola fuente y tres renderizaciones, así que no pueden discrepar.
