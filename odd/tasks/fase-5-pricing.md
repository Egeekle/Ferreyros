# Fase 5 · Pricing y captura de valor (P1, P2, P3)

**Objetivo:** cerrar la Fase 5 de `docs/brief-metodologia.md` y la §4.7/§6 de
`docs/brief-problema.md` — elasticidades por marca y por segmento, curvas de propensión frente
al precio, los tres sub-problemas de optimización resueltos sobre los parámetros que estimó la
Fase 4, e impacto por área de negocio con KPIs medibles contra grupo de control.
**Rama:** master · **Entrega:** D4 del cronograma (18 nov) · **Runner:** sin pytest; la
verificación son frames de veredicto ejecutados y aseverados en el notebook.

## Specs

- **S1** — Brief metodología Fase 5: «elasticidad precio propia y cruzada por marca **y por
  segmento**, y se construyen curvas de propensión frente al precio. Esto responde la pregunta
  comercial real: *¿qué pasa con la demanda si muevo el precio de la marca 3 un 5%?*»
- **S2** — Brief problema §4.7: «**P1 es el modelo base entregable**: tiene solución exacta, es
  interpretable y produce de inmediato la cifra que el negocio quiere — a quién contactar y
  cuánto margen esperado genera».
- **S3** — Brief problema §4.7: P1 es un knapsack con presupuesto y fatiga cuyo «LP relajado es
  integral → greedy por margen esperado es **óptimo**». Esa optimalidad debe **verificarse**
  contra el LP, no afirmarse.
- **S4** — Brief problema §4.7: **P2** calendario promocional, MILP con R5, «se apoya en los
  lifts de §2.5». **P3** precio, MILP sobre la escalera discreta con R6–R7.
- **S5** — Brief problema §4.6, nota de R7: el ordenamiento estricto de precios es «una
  **decisión de modelado**... Se implementa como restricción activable y **se reporta el costo
  de imponerla**».
- **S6** — Brief problema §4.8: no hay costos (`c_j`, `a`, `d_j`, `B`, `ρ`). Entran como
  **supuestos declarados con análisis de sensibilidad**, nunca como números inventados sin marca.
- **S7** — Fase 4: la elasticidad propia de la marca 3 **no está identificada** (positiva en las
  5 especificaciones). P3 no puede optimizar su precio con esa elasticidad; la Fase 5 tiene que
  respetar ese límite en lugar de ignorarlo.
- **S8** — Brief metodología Fase 5: tabla de impacto por área (Comercial, Marketing, Pricing,
  Planeamiento), cada una atada a una cifra producida aquí.
- **S9** — Brief metodología Fase 5, KPIs: «*Ingreso incremental atribuible, medido con **A/B
  test o grupo de control** — sin medición contrafactual no hay captura de valor demostrable,
  solo una proyección*». Cada KPI con línea base, meta y método de medición.
- **S10** — Los contratos congelados siguen mandando: `framing.verify_frozen`,
  `segmentation.verify_frozen` y las métricas de `artifacts/modelos.json` antes de optimizar.
- **S11** — Estilo de la casa: la lógica vive en `src/`, el notebook orquesta, cada bloque cierra
  con su consecuencia, y el notebook corre de extremo a extremo con 0 errores.

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S6, S10 | inline | pendiente | supuestos de margen y costos + simulador contrafactual sobre M1/M2/M3 |
| T2 | S1, S7 | inline | pendiente | curvas de propensión frente al precio y elasticidades por segmento |
| T3 | S2, S3 | inline | pendiente | P1 resuelto y su optimalidad verificada contra el LP |
| T4 | S4, S5 | inline | pendiente | P2 (calendario) y P3 (precio) con R6–R7 y el costo de R7 |
| T5 | S8, S9, S11 | inline | pendiente | KPIs, impacto por área, `notebooks/05_pricing.ipynb`, congelado |

## Log

**L1** (petición original, 2026-11-18): «sigue con la fase 5 de la metodologia y problema»

**L2** — Estado al empezar: Fases 0–4 cerradas, con tres contratos congelados
(`artifacts/encuadre.json`, `artifacts/segmentos.json`, `artifacts/modelos.json`). La Fase 4
entregó los tres parámetros de la función objetivo (`π`, `q`, `λ`) y dejó dos cosas declaradas
que condicionan esta fase: faltan los costos, y la elasticidad de la marca 3 no está
identificada.

**L3** — Decisión de diseño: el contrafactual se construye **modificando el frame crudo y
reconstruyendo las features con `features.build_features` y la referencia congelada de train**,
no tocando columnas de la matriz ya construida. Es la única forma de que `precio_relativo_j`,
`precio_min_dia`, `desviacion_precio_j` y `descuento_implicito_j` queden coherentes entre sí: si
se edita una columna suelta, el escenario deja de ser un escenario posible.

**L4** — Engram no está disponible en esta sesión (los tools MCP no cargaron): el espejo
`odd/fase-5-pricing/tasks` queda **pendiente de sincronizar**.
