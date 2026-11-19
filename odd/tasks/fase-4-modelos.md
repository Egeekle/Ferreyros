# Fase 4 · Modelo de propensión de compra (M1, M2, M3)

**Objetivo:** cerrar la Fase 4 de `docs/brief-metodologia.md` y la §5 de `docs/brief-problema.md`
— los tres modelos encadenados que alimentan la función objetivo, con progresión justificada por
ganancia medida, validación doble (temporal + por cliente), PR-AUC como métrica principal,
umbral derivado del costo de contacto y explicabilidad con SHAP.
**Rama:** master · **Entrega:** D3 del cronograma (18 nov) · **Runner:** sin pytest; la
verificación son frames de veredicto ejecutados y aseverados en el notebook.

## Specs

- **S1** — Interfaz de la §5.1 del brief de problema: **M1** incidencia binaria sobre 58,693
  visitas → `π_it`; **M2** elección de marca multiclase sobre las 14,638 compras → `q_ijt` y
  **elasticidades cruzadas**; **M3** cantidad sobre las 14,638 compras → `λ_ijt`.
- **S2** — §5.2: «Progresión justificada por ganancia medida: logística regularizada → Random
  Forest → Gradient Boosting con tuning por validación cruzada». Cada paso solo se acepta si la
  ganancia está medida.
- **S3** — §5.2: «Para M2, un logit multinomial tiene la ventaja de que sus coeficientes de
  precio **son** las elasticidades que la optimización necesita: la interpretabilidad no es un
  extra, es un requisito funcional de la capa prescriptiva».
- **S4** — §5.3: split temporal congelado del día 550 para el test; `TimeSeriesSplit` dentro de
  train para el tuning; `GroupKFold` por `id` en paralelo para medir generalización a clientes
  nuevos. Nunca un split aleatorio.
- **S5** — §5.3: «**Métrica principal PR-AUC**, con ROC-AUC de apoyo». Accuracy está prohibida
  como métrica de M1 (75.1% prediciendo siempre no-compra).
- **S6** — §5.3: «**Umbral de operación derivado del costo**, no fijado en 0.5: el umbral óptimo
  es el que resuelve P1, el que iguala margen esperado y costo de contacto».
- **S7** — §5.2 y Fase 4 del brief de metodología: «Explicabilidad con **SHAP** global y local,
  más lectura en *odds ratio* del modelo logístico».
- **S8** — §5.4: baselines a superar — tasa base 0.2494 y **lift ×2.43** del targeting por tasa
  histórica en el decil superior. El valor se mide contra esa política, no contra el azar.
- **S9** — Composición: `Valor esperado de la visita = P(compra) × P(marca | compra) ×
  E[cantidad] × precio`. Los tres modelos tienen que poder evaluarse juntos, porque eso es lo
  que consume el optimizador.
- **S10** — El encuadre de la Fase 0 y la segmentación de la Fase 3 siguen mandando:
  `framing.verify_frozen(df)` y `segmentation.verify_frozen(modelo)` antes de entrenar.
- **S11** — Decisión pendiente de la Fase 1: `semana` y `mes` son índices de tendencia cuyos
  valores de test caen fuera del rango de train (98% y 88% de las filas) y no hay tendencia que
  extrapolar (t=−1.16). Se decide **con la métrica en la mano**, no por principio.
- **S12** — Estilo de la casa: la lógica vive en `src/`, el notebook orquesta, cada bloque cierra
  con su consecuencia, y el notebook corre de extremo a extremo con 0 errores.

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S1, S2, S4, S5, S8, S11 | inline | hecho | M1 logistica C=0.03 (59 features): PR-AUC test **0.4776** vs 0.2480 de azar, ROC 0.6931. Progresion con comparacion pareada fold a fold: RF +0.0117 (t=0.84), HGB -0.0264 (t=-1.63), ninguna supera el ruido. Lift top10 x2.50 vs x2.46 de la politica historica sobre las mismas filas. S11 cerrada: quitar `semana`/`mes` cuesta -0.0009 en test |
| T2 | S6, S7 | inline | hecho | `threshold_by_cost` + `verify_threshold_rule`: la regla umbral=costo/margen se desvia como maximo 0.055 del optimo empirico. SHAP global (historial 51.7% de la importancia, 0 features demograficas en el top 12), SHAP local y odds ratios (tasa historica 1.740, promo_marca_5 1.311, promo_marca_3 0.765) |
| T3 | S1, S3 | inline | hecho | M2: RF con log-loss test 0.7418 y accuracy 0.7822 vs 0.3407 de la clase mayoritaria. Logit multinomial para las elasticidades: 4 de 5 propias identificadas; la marca 3 sale positiva en las 5 especificaciones |
| T4 | S1, S9 | inline | hecho | M3 Poisson: MAE test 0.9790 vs 1.1768 del baseline de la media (-16.8%). Composicion: el valor esperado captura el **31.7%** del ingreso real de test con el 10% de las visitas, frente a 29.0% de solo M1 y 28.2% de la politica historica |
| T5 | S10, S12 | inline | hecho | `notebooks/04_modelos.ipynb`: 59 celdas, 26 de codigo, 7 figuras, 0 errores, 90s de ejecucion. Metricas congeladas en `artifacts/modelos.json`; 8/8 requisitos en el reporte de fase |

## Log

**L1** (petición original, 2026-11-18): «sigue con la fase 4 de la metodologia y problema»

**L2** — Estado al empezar: Fases 0–3 cerradas con sus notebooks y dos contratos congelados
(`artifacts/encuadre.json`, `artifacts/segmentos.json`). `preprocess.prepare()` entrega las
matrices con el ajuste solo en train (control E10) y `segmentation.prepare_with_segment()` las
entrega con el segmento como feature (86 columnas).

**L3** — Dos decisiones de herramientas, tomadas por operabilidad porque Ferreycorp **ejecuta**
la solución:
- **SHAP sí**: agregado como dependencia (`shap 0.54.1`), verificado con `TreeExplainer` sobre
  `HistGradientBoosting` y `LinearExplainer` sobre la logística. Es requisito explícito del
  enunciado, no un extra.
- **Gradient boosting con `HistGradientBoostingClassifier` de scikit-learn** en lugar de
  XGBoost/LightGBM. Es la misma familia (GBDT por histogramas) y evita que el notebook dependa
  de `libomp`, que en macOS es el fallo de importación más común de LightGBM. La sustitución
  queda declarada en el notebook.

**L4** — Engram no está disponible en esta sesión (los tools MCP no cargaron): el espejo
`odd/fase-4-modelos/tasks` queda **pendiente de sincronizar**.

**L5** — Defecto propio corregido durante la verificacion: la regla que decidia si la complejidad
paga usaba una tolerancia fija de 0.005 de PR-AUC, y con esa regla el random forest (+0.0117)
ganaba. Pero los folds tienen desvios de 0.019 a 0.042, asi que la tolerancia fija no distinguia
senal de ruido. Lo cambie por una **comparacion pareada fold a fold** (los candidatos se puntuan
sobre los mismos folds): RF +0.0117 con error estandar 0.0139 -> t=0.84; HGB t=-1.63. Ninguna
supera el ruido, asi que se queda la logistica. La conclusion es la misma que ya tenia escrita,
pero ahora esta medida.

**L6** — Hallazgo metodologico que cambia como se leen las metricas de todo el proyecto:
**`GroupKFold` no es el limite pesimista en este panel**. Para el random forest da PR-AUC 0.7790
frente a 0.5390 del esquema que cruza cliente nuevo **y** dia futuro: **+0.2400** de inflacion.
GroupKFold deja fuera al cliente pero entrena con los mismos dias, y como los 500 clientes
comparten calendario, precios y promociones, el modelo ya vio las condiciones de demanda del fold
de validacion. Mide "cliente nuevo, periodo conocido", no generalizacion. Por eso agregue
`grouped_time_folds` y todas las decisiones de la fase se toman con ese esquema.

**L7** — M1 apenas bate la politica vigente **como clasificador**: lift x2.50 contra x2.46 de
ordenar por tasa de compra historica, sobre las mismas filas de test. Es coherente con el EDA (la
amplitud entre clientes es 0.876 y la demografica ~0.12: la tasa historica ya contiene casi toda
la senal). Lo que si crea valor es la **cadena completa**: ordenar por ingreso esperado
(pi x q x lambda x precio) captura el 31.7% del ingreso real con el 10% de las visitas, frente al
29.0% de ordenar por probabilidad y el 28.2% de la politica historica. Es la justificacion medida
de por que el brief pedia tres modelos y no uno. Matiz honesto: con presupuesto del 30% la ventaja
se diluye (57.4% vs 58.4%) -- la cadena paga donde el presupuesto aprieta.

**L8** — Limite declarado, no resuelto: la **elasticidad propia de la marca 3 sale positiva**
(+4.9 a +17.0) en las cinco especificaciones -- con y sin penalizacion L2, con controles de
lealtad, con la ultima marca comprada y con el segmento. No es un accidente de ajuste. La tabla de
variacion explica por que le toca a ella: 21 puntos de precio (vs 26-44), 14.4% de rango relativo
(vs 28-51%), la menor dispersion del precio relativo (0.0269) y el 5.1% de las compras. Sin
variacion de precio independiente de la demanda, lo que se mide no es respuesta al precio sino la
reaccion del precio a la demanda. Es el supuesto que el brief de problema declaro en §4.8,
confirmado con numero. Consecuencia: P3 congela el precio de la marca 3 o lo mueve dentro de un
supuesto con sensibilidad, y lo primero que vale pedirle al negocio es un test A/B de precio.

**L9** — Tres analisis independientes senalan lo mismo sobre la marca 3: su promocion no mueve la
incidencia (Fase 1, z=0.7), su elasticidad propia no esta identificada (Fase 4) y es el odds ratio
mas negativo de M1 (0.765). Y el reverso, tambien triple, sobre la marca 5: mayor efecto
incremental de promocion (+6.90 pp), odds ratio 1.311 y 43.5%/38.9% de las compras de los
segmentos 2 y 3. La recomendacion de P2 descansa en tres evidencias que no comparten metodo.

**L10** — Sin commit: el repositorio arrastra trabajo sin versionar de sesiones anteriores y nadie
pidio commitear. `pyproject.toml` y `uv.lock` quedan modificados por `uv add shap`.
