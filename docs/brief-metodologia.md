# Brief de Metodología — Modelo de Propensión de Compra

**Prueba Técnica Data Scientist · Ferreycorp**
Entrega: miércoles 18 de noviembre, 11:59 pm · `seleccion@ferreycorp.com.pe`
Documento de trabajo: define la lógica de la solución antes de escribir código.

---

## 1. Objetivo

Construir un modelo que prediga **si un cliente comprará o no un determinado producto** en una visita dada, y traducir esa predicción en decisiones accionables de negocio (targeting, promoción y precio).

La formulación operativa es a nivel **cliente × visita**:

> Dado un cliente con su perfil demográfico, su historial de compra y las condiciones comerciales del día (precios y promociones de las 5 marcas), estimar `P(compra = 1)` y, condicionado a que compre, **qué marca** elegirá y **cuánto** comprará.

Esta descomposición es deliberada: la prueba evalúa *Modelos de Pricing*, y la elasticidad precio-demanda solo es estimable si el precio entra en un modelo de elección, no solo en el de incidencia.

---

## 2. Lo que exige la prueba y dónde se resuelve

| Tarea del enunciado | Fase de esta metodología | Entregable donde aparece |
|---|---|---|
| 1. Análisis Exploratorio (EDA) | Fase 1 | Notebook + slide 3 |
| 2. Metodología | Este brief → Fase 0–5 | Doc PDF + slide 4 |
| 3. Procesamiento y limpieza | Fase 2 | Notebook |
| 4. Modelo de propensión (clustering + clasificación, fine tuning, selección, explicabilidad) | Fases 3–4 | Notebook + Doc PDF |
| 5. Impacto en negocio y captura de valor | Fase 5 | Doc PDF + slide 5 |

**Skills bajo evaluación:** Data Science · Machine Learning · Modelos de Pricing · Storytelling · Cloud Computing. Las dos últimas no se cubren con código: exigen una narrativa de valor y una arquitectura de despliegue explícitas. Están cubiertas en Fase 5 y Fase 6.

---

## 3. Datos: hechos verificados

Medidos sobre `data/compras_data.csv`, no supuestos:

| Hecho | Valor | Implicación metodológica |
|---|---|---|
| Observaciones | 58,693 visitas | Suficiente para modelos no lineales |
| Clientes únicos | 500 | **Pocos clientes.** La unidad de generalización es el cliente, no la fila |
| Visitas por cliente | 117 promedio (72–358) | Panel longitudinal, no muestra i.i.d. |
| Horizonte | días 1 a 730 | ~2 años → permite split temporal |
| Tasa de compra | **24.94%** | Desbalance moderado; no requiere SMOTE, sí ajuste de umbral |
| Marcas | 5 (`id_marca` 1–5) | Elección multiclase condicional |
| Rango de precios | marca 1: 1.10–1.59 · marca 5: 2.11–2.80 | Escalera de precios clara, marca 1 = económica, marca 5 = premium |
| Exposición a promoción | marca 1: 34.4% · marca 3: 4.3% | Muy heterogénea; no asumir efecto promocional uniforme |
| Nulos / duplicados | 0 / 0 | La limpieza se centra en consistencia lógica, no en imputación |

### Dos trampas que la metodología debe evitar explícitamente

**Fuga de datos (*leakage*).** `id_marca` y `cantidad` son **resultados** de la visita, no condiciones previas. Usarlos como features del modelo de incidencia produce métricas casi perfectas y un modelo inútil en producción. Solo `ultima_marca_comprada` y `ultima_cantidad_comprada` (rezagos) son admisibles como historial.

**Validación aleatoria inválida.** Con 117 visitas por cliente, un `train_test_split` aleatorio coloca visitas del mismo cliente en train y test, inflando el desempeño. La validación debe ser **temporal** (corte por `dia_visita`) y reportarse también **por cliente** (`GroupKFold` sobre `id`) para medir generalización a clientes nuevos.

**Variable excluida:** `tamanio_ciudad`, por indicación explícita del diccionario de datos.

---

## 4. Metodología — Fases

### Fase 0 · Encuadre y preparación
Definir la unidad de análisis (cliente × visita), la variable objetivo (`incidencia_compra`), el conjunto de features admisibles y la estrategia de validación **antes** de modelar. Congelar el split temporal: train ≈ días 1–550, test ≈ días 551–730.

### Fase 1 · Análisis Exploratorio (EDA)
- Distribución de la variable objetivo, global y por segmento demográfico.
- Perfil de los 500 clientes: edad, género, estado civil, educación, ingreso, ocupación.
- Dinámica temporal: estacionalidad semanal/mensual, frecuencia de visita, tendencia de la tasa de compra.
- Estructura de precios: trayectoria de los 5 precios en el tiempo, correlación entre marcas, brecha de precio relativo.
- Efecto bruto de promoción: tasa de compra con y sin promoción, por marca.
- Anomalías: visitas con `incidencia_compra = 1` y `cantidad = 0`, clientes con comportamiento atípico, cantidades extremas (máximo observado: 15).

**Criterio de cierre:** cada hallazgo del EDA debe tener una consecuencia declarada sobre feature engineering o sobre el modelo. Un gráfico sin consecuencia no entra al entregable.

### Fase 2 · Procesamiento y feature engineering
- Validación de consistencia entre `incidencia_compra`, `id_marca` y `cantidad`.
- Exclusión de `tamanio_ciudad` y de toda feature post-resultado.
- **Features de precio:** precio mínimo del día, precio relativo de cada marca frente al promedio, brecha contra la última marca comprada.
- **Features de promoción:** número de marcas en promoción ese día, si está en promoción la última marca comprada.
- **Features de historial (rezagadas, sin fuga):** días desde la última compra, frecuencia de compra acumulada, ticket promedio histórico, lealtad a marca.
- **Features demográficas:** tratamiento ordinal para educación, binning de edad e ingreso.
- Codificación y escalado dentro de un `Pipeline` de scikit-learn, ajustado **solo** sobre train.

### Fase 3 · Segmentación de clientes (no supervisada)
K-Means sobre las variables demográficas, con estandarización previa y selección de *k* por método del codo y *silhouette*; PCA para visualización e interpretación de componentes. Cada segmento recibe un nombre de negocio y un perfil de comportamiento de compra. El segmento se incorpora como feature del modelo supervisado y, sobre todo, como **unidad de acción comercial**: las campañas se diseñan por segmento, no por cliente individual.

### Fase 4 · Modelo de propensión de compra (supervisado)

**Cadena de decisión en tres modelos encadenados:**

1. **Incidencia** — clasificación binaria: `P(compra)`. Es el modelo de propensión solicitado.
2. **Elección de marca** — clasificación multiclase sobre las visitas con compra: `P(marca | compra)`. Aquí se estiman las **elasticidades cruzadas de precio**.
3. **Cantidad** — regresión sobre las visitas con compra. Cierra el cálculo de ingreso esperado.

`Valor esperado de la visita = P(compra) × P(marca | compra) × E[cantidad] × precio`

**Selección de modelos.** Progresión de complejidad creciente, cada paso justificado por la ganancia medida:
Regresión logística (baseline interpretable y fuente directa de coeficientes de elasticidad) → Random Forest → Gradient Boosting (XGBoost / LightGBM).

**Fine tuning.** Búsqueda de hiperparámetros con validación cruzada respetando el orden temporal (`TimeSeriesSplit`) y, en paralelo, `GroupKFold` por `id` para medir generalización a clientes no vistos.

**Métricas.** ROC-AUC y **PR-AUC** como métrica principal (apropiada bajo desbalance 25/75), más precisión y recall al umbral de operación elegido. El umbral no se fija en 0.5: se optimiza sobre el **costo de negocio** (costo de contactar un cliente que no compra vs. valor de capturar uno que sí).

**Explicabilidad.** SHAP a nivel global (ranking de drivers) y local (por qué este cliente, en esta visita). Para los coeficientes del modelo logístico, lectura en términos de *odds ratio*. La explicabilidad es requisito del enunciado, no un extra.

### Fase 5 · Pricing y captura de valor

**Elasticidades.** Con el modelo de elección de marca se estima la elasticidad precio propia y cruzada por marca y por segmento, y se construyen curvas de propensión frente al precio. Esto responde la pregunta comercial real: *¿qué pasa con la demanda si muevo el precio de la marca 3 un 5%?*

**Impacto por área de negocio:**

| Área | Uso del modelo |
|---|---|
| Comercial / Ventas | Priorización de la cartera por propensión; foco de la fuerza de ventas en el decil superior |
| Marketing | Promoción dirigida por segmento; fin del descuento indiscriminado |
| Pricing | Precio diferenciado por segmento basado en elasticidad estimada |
| Planeamiento / Inventario | Demanda esperada por marca a partir del ingreso esperado |

**KPIs de captura de valor** (línea base, meta y método de medición para cada uno):
- *Lift* en tasa de conversión del decil superior frente a la selección aleatoria actual.
- Margen incremental por promoción enfocada (ahorro en descuento no necesario).
- Ingreso incremental atribuible, medido con **A/B test o grupo de control** — sin medición contrafactual no hay captura de valor demostrable, solo una proyección.
- Estabilidad del modelo en producción: PSI de las features, *drift* de la tasa de compra.

### Fase 6 · Arquitectura y despliegue (Cloud Computing)

Arquitectura de referencia, con la nube como decisión explícita y no como adorno:

```
Ingesta           → Data Lake (S3 / ADLS), zonas raw · curated · features
Procesamiento     → Jobs de ETL y feature engineering versionados
Feature Store     → Features reutilizables entre entrenamiento e inferencia
Entrenamiento     → Jobs de ML gestionados + registro de experimentos (MLflow)
Model Registry    → Versionado, etapas (staging / producción)
Inferencia        → Batch nocturno (scoring de la cartera) + endpoint REST para scoring puntual
Consumo           → Dashboard comercial (Power BI / QuickSight) + CRM
Monitoreo         → Drift de datos, degradación de métricas, alertas y reentrenamiento programado
```

El reentrenamiento se dispara por calendario o por degradación detectada, no de forma manual.

---

## 5. Cronograma propuesto (al 18 de noviembre)

| Día | Bloque | Salida |
|---|---|---|
| D1 | Fases 0–1: encuadre y EDA | Notebook con EDA cerrado y hallazgos con consecuencia |
| D2 | Fase 2–3: limpieza, features y segmentación | Dataset modelable + segmentos nombrados |
| D3 | Fase 4: los tres modelos, tuning y SHAP | Métricas en test temporal y por cliente |
| D4 | Fases 5–6: pricing, KPIs, arquitectura | Documento PDF + 7 slides |
| D5 | Reproducibilidad y revisión | Notebook ejecutable de extremo a extremo, repositorio ordenado |

---

## 6. Checklist de entregables

- [ ] Código de la solución — Word o TXT
- [ ] Documentación: objetivos, arquitectura, metodología, resultados — PDF
- [ ] Presentación, **máx. 7 slides** — PDF, cubriendo: Objetivos · Cronograma · EDA · Metodología · Captura de valor
- [ ] Repositorio (GitHub) o ZIP
- [ ] **Notebook que ejecute paso a paso y muestre resultados** — Ferreycorp probará la operatividad; debe correr de principio a fin sin intervención

> La sugerencia del enunciado sobre el notebook es, en la práctica, un requisito: la solución se evalúa ejecutándola.

---

## 7. Decisiones metodológicas que diferencian esta solución

1. **Validación temporal y por cliente**, no aleatoria — con 500 clientes y 117 visitas cada uno, el split aleatorio es el error más probable y el más costoso.
2. **Exclusión explícita de features post-resultado** (`id_marca`, `cantidad`) del modelo de incidencia.
3. **Umbral de decisión derivado del costo de negocio**, no fijado en 0.5.
4. **Cadena de tres modelos** que convierte propensión en ingreso esperado y habilita el análisis de pricing que la prueba evalúa.
5. **Captura de valor medida contra un control**, no proyectada sobre supuestos.
