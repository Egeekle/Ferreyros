# Fase 1 · EDA — cierre contra el brief de metodología y el brief de problema

**Objetivo:** el notebook 01 se construyó siguiendo la matriz de alineamiento (bloques A–D
del brief de problema). Esta tarea lo cierra también contra el **checklist de la Fase 1 de
`docs/brief-metodologia.md`**, que pide seis cosas, y cubre las cuatro que hoy no están.
**Rama:** master · **Entrega:** D1 del cronograma (18 nov) · **Runner:** sin pytest; la
verificación son frames de veredicto ejecutados y aseverados en el notebook.

## Specs

- **S1** — Brief metodología, Fase 1: «Dinámica temporal: estacionalidad semanal/mensual,
  frecuencia de visita, **tendencia de la tasa de compra**». Hoy existe estabilidad por
  semestre (A1) y frecuencia de visita (C5). Falta estacionalidad semanal, mensual y
  tendencia con pendiente estimada.
- **S2** — Brief metodología, Fase 1: «Estructura de precios: trayectoria de los 5 precios en
  el tiempo, correlación entre marcas, **brecha de precio relativo**». Trayectoria y
  correlación están en B2. Falta la brecha de precio relativo.
- **S3** — Brief metodología, Fase 1: «**Efecto bruto de promoción: tasa de compra con y sin
  promoción, por marca**». Hoy A3 mide incidencia por *número* de promos y A5 mide el lift
  del *share condicional*. Falta la incidencia con y sin promo **por marca**: es una pregunta
  distinta (expansión de mercado vs. canibalización) y las dos entran en la función objetivo.
- **S4** — Brief metodología, Fase 1: «Anomalías: visitas con `incidencia_compra = 1` y
  `cantidad = 0`, **clientes con comportamiento atípico, cantidades extremas** (máximo
  observado: 15)». `consistency_checks` cubre el primer caso. Faltan los otros dos.
- **S5** — Criterio de cierre del brief: «cada hallazgo del EDA debe tener una consecuencia
  declarada sobre feature engineering o sobre el modelo. **Un gráfico sin consecuencia no
  entra al entregable.**» Todo bloque nuevo cierra con su consecuencia.
- **S6** — Trazabilidad verificable: una matriz que mapee los seis ítems de la Fase 1 del
  brief de metodología a la función que produce su evidencia y a la sección del notebook,
  en el mismo estilo que `audit.traceability_matrix()` (que mapea el brief de problema).
- **S7** — No duplicar lo que ya existe. Frecuencia de visita se lee de
  `constraints.fatigue_bounds` / `visit_interval_profile`; distribución del target por
  segmento demográfico, de `segments.purchase_rate_by_demographic`.
- **S8** — Estilo de la casa: la lógica vive en `src/`, el notebook orquesta, y el notebook
  corre de extremo a extremo con 0 errores.

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S1 | inline | hecho | `eda/dynamics.py`: estacionalidad semanal y mensual con prueba de replica entre anios, `warmup_profile` y `purchase_rate_trend` con pendiente antes y despues del arranque. Veredicto medido: sin estacionalidad (correlaciones +0.096 y +0.113), sin tendencia tras el bloque 4 (t=-1.16), arranque de 90 dias (0.1874 vs 0.2594) |
| T2 | S2, S3 | inline | hecho | `decisions.relative_price_gap`: precio relativo 0.70-1.33, marca 1 la mas barata el 96.6% de los dias, dispersion media 0.046. `objective.incidence_by_promo_per_brand`: delta de incidencia de +0.64 pp (marca 3, z=0.7) a +6.90 pp (marca 5, z=6.7) |
| T3 | S4 | inline | hecho | `eda/anomalies.py`: 0 de 6 incoherencias; cola de cantidad >=6 = 6.62% de compras y 18.42% de unidades en 180 clientes; 14 clientes sobre 3 sigmas con el 16.43% del ingreso |
| T4 | S5, S6, S8 | inline | hecho | `notebooks/01_eda.ipynb` regenerado: 126 celdas, 61 de codigo, 20 figuras, ejecutado de extremo a extremo con 0 errores y todos los `assert` en verde. `audit.metodologia_fase1_matrix`: 6/6 items resueltos |

## Log

**L1** (petición original, 2026-11-17): «arranca con la fase 01 con la fase metodologia asi
como la de brief problema»

**L2** — Estado al empezar: `notebooks/01_eda.ipynb` existe (102 celdas, 50 de código, 14
figuras) y cubre la matriz de alineamiento completa, más las Fases 0 y 2 en sus propios
notebooks. Lo que nadie había auditado es el notebook 01 contra el **checklist de seis puntos
de la Fase 1 del brief de metodología**, que es un eje distinto de la matriz A–D. Dos de los
seis puntos ya estaban cubiertos (perfil de los 500 clientes, target por segmento
demográfico); cuatro no.

**L3** — Engram no está disponible en esta sesión (los tools MCP no cargaron): el espejo
`odd/fase-1-eda-metodologia/tasks` queda **pendiente de sincronizar**.

**L4** — Hallazgo que invierte una conclusion: leido solo con el error estandar, el dato dice que
hay estacionalidad semanal **y** mensual (12 de 19 celdas se desvian mas de 2 errores estandar).
La prueba de replica entre los dos anios del panel la desmonta: las correlaciones entre perfiles
son +0.096 (semanal) y +0.113 (mensual). Es ruido de plegar 730 dias en 7 o 12 posiciones y
probarlas todas. Sin esa segunda prueba el notebook habria declarado una estacionalidad
inexistente y habria creado features de dia de la semana.

**L5** — Lo que si existe es un **regimen de arranque**: los primeros 3 bloques (hasta el dia 90)
tienen tasa 0.1874 frente a 0.2594 despues, y en el bloque 1 el 88.2% de las visitas no tiene
historial. El umbral que define el corte se calcula solo sobre train (bloques 4 en adelante hasta
el dia 550), asi que la decision no mira el test. Consecuencia: esas visitas entran al
entrenamiento, pero lo que debe explicarlas es el indicador de ausencia de historial de la Fase 2
(control E7), y las metricas de M1 se reportan tambien sin ellas.

**L6** — Cierra la pregunta que la Fase 2 dejo abierta en su L8. Sobre los 25 bloques la pendiente
de la tasa de compra es +0.00128 (t=+1.40), pero esa recta esta arrastrada por el arranque; desde
el bloque 4 es -0.00084 (t=-1.16, R2=0.06). No hay tendencia, asi que `semana` y `mes` como
indices de tendencia no tienen senal que justifique extrapolarlos fuera del rango de train (98% y
88% de las filas de test). La Fase 4 decide con la metrica en la mano, pero ahora con evidencia.

**L7** — Matiz al hallazgo central del EDA. A5 medi­a el lift del **share** condicional a compra;
`incidence_by_promo_per_brand` mide el efecto sobre la **incidencia**, que es compra incremental.
No coinciden: la marca 3, la mas elastica por share (x1.86), **no mueve la incidencia** (+0.64 pp,
z=0.7), mientras la marca 5 suma **+6.90 pp** con 24 dias de promocion. El arbitraje de P2 tiene
un destino concreto, y distinguir expansion de canibalizacion es lo que evita pagar descuento por
volumen ya vendido.

**L8** — Defecto propio corregido durante la verificacion: el `str.format` de
`anomalies.quantity_outliers` tenia 8 marcadores y 7 argumentos (faltaba `p99`) y fallaba con
`IndexError`. Lo detecto el smoke test antes de llegar al notebook.

**L9** — Sin commit: el repositorio viene con trabajo de sesiones anteriores sin versionar y nadie
pidio commitear. Los cambios quedan en el arbol de trabajo.
