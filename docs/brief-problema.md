# Brief de Estructuración del Problema

**Prueba Técnica Data Scientist · Ferreycorp**
Modelo base de optimización + capa predictiva de Machine Learning
Fuentes: `label/Diccionario de Datos.txt` · `data/compras_data.csv` (58,693 × 24)

---

## 1. Por qué estructurar primero el problema de optimización

El enunciado pide un modelo de propensión de compra. Un clasificador, por sí solo, entrega una probabilidad — no una decisión. La probabilidad no dice **a qué cliente contactar**, **qué marca ofrecerle**, **a qué precio** ni **cuándo poner una promoción**, y son esas cuatro decisiones las que generan margen.

Por eso la solución se estructura en dos capas con responsabilidades separadas:

```
CAPA PRESCRIPTIVA  (modelo base de optimización)
   ¿Qué hago?  → decide precio, promoción y targeting
         ▲
         │  consume parámetros estimados: π, q, λ
         │
CAPA PREDICTIVA  (modelos de Machine Learning)
   ¿Qué va a pasar?  → estima la función de respuesta del cliente
```

El modelo de optimización es la **función objetivo del negocio**; los modelos de ML son los **estimadores de sus coeficientes**. Esta separación es lo que convierte un ejercicio de clasificación en un sistema de decisión, y es también lo que hace medible la captura de valor: el óptimo del modelo da la cota superior contra la cual se compara la práctica actual.

---

## 2. El dataset en detalle

### 2.1 Estructura general

| Propiedad | Valor medido |
|---|---|
| Filas | 58,693 (una por **cliente × visita**) |
| Columnas | 24 (23 utilizables) |
| Clientes únicos | **500** |
| Horizonte | días 1 a 730 (**730 días distintos, sin huecos**) |
| Visitas por cliente | media 117.4 · mín 72 · máx 358 |
| Visitas por día | media 80.4 · mín 6 · máx 179 |
| Nulos | 0 |
| Duplicados exactos | 0 |
| Tasa de compra global | **24.94%** (14,638 compras) |
| Ingreso observado | 82,756.58 en 730 días (113.37/día) |

Es un **panel longitudinal balanceado en el tiempo pero desbalanceado por cliente**: pocos clientes observados muchas veces. Esto determina toda la estrategia de validación (§5.3).

### 2.2 Las 24 columnas, según el diccionario y lo medido

| Columna | Diccionario | Tipo | Medido | Rol en el modelo |
|---|---|---|---|---|
| `id` | Código del cliente | int, 500 niveles | 2000000001–2000000500 | Clave de agrupación |
| `dia_visita` | Día desde apertura | int 1–730 | 730 valores | Eje temporal / split |
| `incidencia_compra` | 0 no compró, 1 compró | binaria | 24.94% positivos | **Target modelo 1** |
| `id_marca` | Identificador de marca | 0–5 | 0 = no compra (44,055) | **Target modelo 2** · ⚠ post-resultado |
| `cantidad` | Cantidad comprada | 0–15 | 0 si no compra | **Target modelo 3** · ⚠ post-resultado |
| `ultima_marca_comprada` | Última marca comprada | 0–5 | ✅ lag real verificado | Feature de historial |
| `ultima_cantidad_comprada` | "Última cantidad" | **binaria 0/1** | ⚠ no es cantidad | Feature de historial |
| `precio_marca_1..5` | Precio de la marca x | float | 5 series diarias | **Variable de decisión** |
| `promo_marca_1..5` | Marca en promoción ese día | binaria | 5 series diarias | **Variable de decisión** |
| `genero` | Género | binaria 0/1 | 0: 61.4% · 1: 38.6% | Feature / segmentación |
| `estado_civil` | Estado civil | binaria 0/1 | 0: 60.7% · 1: 39.3% | Feature / segmentación |
| `edad` | Edad | int 18–75 | media 38.7 | Feature / segmentación |
| `nivel_educacion` | Nivel de educación | ordinal 0–3 | 0:14% 1:63% 2:20% 3:2% | Feature / segmentación |
| `ingreso_anual` | Ingreso anual | int | 38,247–309,364 | Feature / segmentación |
| `ocupacion` | Ocupación | nominal 0–2 | 0:36% 1:51% 2:13% | Feature / segmentación |
| `tamanio_ciudad` | **NO CONSIDERAR** | 0–2 | — | **Excluida por indicación** |

### 2.3 Dos hallazgos estructurales en las columnas de historial

Verificados, no supuestos:

**`ultima_marca_comprada` es un lag legítimo.** Ordenando por `(id, dia_visita)`, coincide **100%** con `groupby('id').id_marca.shift(1)`. Es la marca comprada en la visita anterior del mismo cliente, no la de la visita actual. Es usable como feature sin fuga de datos.

**`ultima_cantidad_comprada` está mal nombrada en el diccionario.** Tiene solo **2 valores (0 y 1)**, no una cantidad. Es el indicador de si la visita anterior terminó en compra — es decir, el lag de `incidencia_compra`, no de `cantidad`. Se usará con ese significado y se documentará la discrepancia. La cantidad rezagada real, si se necesita, hay que construirla.

### 2.4 Estructura competitiva: las 5 marcas

| Marca | Precio medio | Rango observado | Puntos de precio | Días en promo | Unidades | Ingreso | Share ingreso |
|---|---|---|---|---|---|---|---|
| 1 | 1.34 | 1.10 – 1.59 (44.5%) | 37 | **34.4%** | 4,717 | 6,306 | 7.6% |
| 2 | 1.71 | 1.26 – 1.90 (50.8%) | 30 | **31.6%** | 12,828 | 21,768 | 26.3% |
| 3 | 2.01 | 1.87 – 2.14 (14.4%) | 21 | 4.3% | 2,994 | 6,022 | 7.3% |
| 4 | 2.15 | 1.76 – 2.26 (28.4%) | 26 | 11.8% | 8,877 | 19,040 | 23.0% |
| 5 | **2.64** | 2.11 – 2.80 (32.7%) | 44 | **3.6%** | 11,198 | 29,621 | **35.8%** |

Lectura de negocio: hay una escalera de precios clara en las medias (1.34 → 2.64), pero **los rangos se solapan** (marca 4 baja a 1.76, por debajo del mínimo de la marca 3). La marca 5 es premium, casi nunca se promociona y aun así aporta el **35.8% del ingreso**. Las marcas 1 y 2 se promocionan un tercio de los días y la marca 1 aporta solo 7.6% del ingreso: **el descuento actual está concentrado donde menos valor se captura.** Esa es la primera hipótesis de arbitraje que el modelo debe cuantificar.

### 2.5 Señal de respuesta: la demanda reacciona a precio y promoción

Evidencia observacional, suficiente para justificar la estructura del modelo:

**Precio → incidencia.** Tasa de compra por quintil del precio mínimo disponible ese día:

| Quintil precio mínimo | 1.10–1.33 | 1.33–1.35 | 1.35–1.42 | 1.42–1.48 | 1.48–1.59 |
|---|---|---|---|---|---|
| Tasa de compra | **28.3%** | 24.8% | 25.3% | 25.3% | **19.2%** |

Del quintil más barato al más caro la incidencia cae de 28.3% a 19.2%: una caída relativa del **32%**. Hay elasticidad que estimar.

**Promoción → incidencia.** Tasa de compra según cuántas marcas están en promoción:

| Marcas en promo | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Tasa de compra | 21.8% | 26.0% | **29.0%** | 27.7% |

**Promoción → elección de marca** (share dado que compra, con promo vs. sin promo):

| Marca | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Lift del share | ×1.08 | ×1.39 | **×1.86** | ×1.16 | **×1.66** |

Las marcas que **menos** se promocionan (3 y 5) son las que **más** responden cuando se las promociona. Las que más se promocionan (1 y 2) responden poco: la marca 1 apenas ×1.08. Esto es exactamente la clase de desalineación que un modelo de optimización corrige.

### 2.6 Heterogeneidad entre clientes: donde está el valor del targeting

| Métrica por cliente | Valor |
|---|---|
| Tasa de compra: mín / mediana / máx | **4.6% / 18.4% / 92.2%** |
| Gasto: mediana / máximo | 107.6 / 1,859.2 |
| Top 10% de clientes | **37.0% del ingreso** |
| Top 20% de clientes | **52.7% del ingreso** |
| Top 50% de clientes | 79.6% del ingreso |

Ordenando clientes por su tasa de compra histórica, el decil superior compra al 60.5% frente al 24.9% global: un **lift de ×2.43 sin usar ningún modelo**. Ese es el piso de desempeño que el modelo de ML debe superar, y la razón por la que el problema de targeting tiene solución no trivial.

Las diferencias demográficas, en cambio, son modestas: educación nivel 2 compra al 29.8% vs. nivel 3 al 18.1%; ocupación 1 al 28.1% vs. ocupación 0 al 20.2%. **La demografía segmenta, pero no predice**: el poder predictivo vendrá del comportamiento y del contexto comercial, no del perfil. Consecuencia de diseño: la demografía se usa para clusterizar y accionar comercialmente; las features de comportamiento y precio se usan para predecir.

### 2.7 Consistencia e integridad

Verificado sin excepciones:

- `incidencia_compra = 1` ⟹ `cantidad ≥ 1` y `id_marca ∈ {1..5}` — 0 violaciones
- `incidencia_compra = 0` ⟹ `cantidad = 0` e `id_marca = 0` — 0 violaciones
- Demografía **estática** por cliente: 0 clientes con más de un valor en edad, ingreso, género, estado civil, educación u ocupación
- `ultima_marca_comprada = 0` en 44,133 filas: primera visita o visita previa sin compra — no es un nulo, es ausencia de historial y debe tratarse como categoría propia
- Tasa de compra estable por semestre (23.4% / 25.5% / 26.3% / 24.8%): **sin drift estructural**, el split temporal es válido

---

## 3. Variables nuevas a construir

El diccionario describe 23 campos utilizables. La señal relevante para la decisión está en las **relaciones** entre ellos, no en los campos crudos. Familias a derivar:

### 3.1 Competitivas de precio (contexto de la visita)
- `precio_min_dia`, `precio_medio_dia`, `rango_precios_dia`
- `precio_relativo_j` = `precio_marca_j / precio_medio_dia` — posición competitiva de cada marca
- `gap_vs_ultima_marca` = precio de la última marca comprada menos el precio mínimo del día → mide el incentivo a cambiar de marca
- `desviacion_precio_j` = precio del día vs. precio histórico medio de la marca → captura si hoy está "barata" en términos propios

### 3.2 De promoción
- `n_promos_dia` (0–3 observado)
- `promo_en_ultima_marca` — si la marca habitual del cliente está en promoción hoy
- `descuento_implicito_j` = 1 − precio del día / precio máximo histórico de la marca

### 3.3 De historial del cliente (rezagadas, sin fuga)
- `dias_desde_ultima_compra` (recencia)
- `frecuencia_compra_acumulada`, `n_visitas_acumuladas` — calculadas **expandiendo solo hacia el pasado**
- `tasa_compra_historica` del cliente
- `ticket_medio_historico`, `cantidad_media_historica`
- `lealtad_marca_j` = share histórico de la marca j en las compras del cliente
- `n_marcas_distintas_compradas` — proxy de propensión a cambiar
- `racha_sin_comprar`

### 3.4 Temporales
- `dia_semana`, `semana_del_anio`, `mes` derivados de `dia_visita`
- `visitas_del_cliente_en_ventana` (7 / 30 días)

### 3.5 Demográficas transformadas
- Binning de `edad` e `ingreso_anual`; `nivel_educacion` tratado como ordinal; `ocupacion` como nominal (one-hot)
- `segmento` = etiqueta del clustering (§5.1), usada como feature y como unidad de acción

**Regla de construcción obligatoria:** toda feature de historial se calcula con información estrictamente anterior a la visita (`shift`, ventanas expansivas por `id`), y todo ajuste de escala o codificación se entrena solo sobre el período de train. `id_marca` y `cantidad` de la fila actual **nunca** entran como features: son resultados de la visita.

---

## 4. El modelo base de optimización

### 4.1 Formulación de la decisión

> Para cada período, elegir **el precio de cada marca**, **qué marcas poner en promoción** y **a qué clientes ofrecer qué marca**, de modo que se maximice el margen de contribución esperado, respetando presupuesto promocional, capacidad de atención comercial, inventario, arquitectura de marcas y piso de margen.

### 4.2 Conjuntos

```
i ∈ I   clientes                       |I| = 500
j ∈ J   marcas                         |J| = 5
t ∈ T   períodos (días)                |T| = 730
k ∈ K_j puntos de precio de la marca j  |K_j| = 21 a 44 (escalera observada)
```

### 4.3 Variables de decisión

```
y_jtk ∈ {0,1}   la marca j se vende al punto de precio k en el período t
z_jt  ∈ {0,1}   la marca j está en promoción en el período t
x_ijt ∈ {0,1}   se ofrece / impulsa la marca j al cliente i en el período t
```

### 4.4 Parámetros

**Estimados por los modelos de ML** (aquí entra la capa predictiva):

```
π_it(p,z)    = P(compra = 1 | cliente i, precios p, promos z, contexto t)   ← Modelo 1
q_ijt(p,z)   = P(marca = j | compra, cliente i, precios p, promos z)        ← Modelo 2
λ_ijt        = E[cantidad | compra, marca j, cliente i]                     ← Modelo 3
```

**De negocio** (requieren input de Ferreycorp; ver §4.8):

```
c_j   costo unitario de la marca j
m_jk  = p_jk − c_j   margen unitario al punto de precio k
a     costo de un contacto / impulso comercial
d_j   costo de tener la marca j un período en promoción
B     presupuesto promocional del horizonte
F_i   máximo de contactos por cliente (fatiga)
Cap_j capacidad / inventario de la marca j
ρ     margen porcentual mínimo exigido
```

### 4.5 Función objetivo

Margen de contribución esperado, neto de costos de ejecución:

```
max  Σ_i Σ_j Σ_t  x_ijt · π_it(p,z) · q_ijt(p,z) · λ_ijt · Σ_k ( y_jtk · m_jk )
     ─────────────────────────────────────────────────────────────────────────
     − a · Σ_i Σ_j Σ_t x_ijt           (costo de contactar)
     − Σ_j Σ_t d_j · z_jt              (costo de promocionar)
```

El término `π · q · λ · m` es el **margen esperado de una visita**: probabilidad de comprar × probabilidad de elegir esa marca × unidades esperadas × margen unitario. Es la cantidad que la capa de ML existe para estimar.

### 4.6 Restricciones

```
(R1) Un precio por marca y período
     Σ_k y_jtk = 1                              ∀ j, t

(R2) Presupuesto de ejecución
     a·Σ_ijt x_ijt + Σ_jt d_j·z_jt ≤ B

(R3) Una oferta por cliente y período (no saturar)
     Σ_j x_ijt ≤ 1                              ∀ i, t

(R4) Fatiga: tope de contactos por cliente
     Σ_j Σ_t x_ijt ≤ F_i                        ∀ i

(R5) Calendario promocional acotado por marca
     Σ_t z_jt ≤ T_j                             ∀ j

(R6) Piso de margen
     Σ_k y_jtk · p_jk ≥ c_j · (1 + ρ)           ∀ j, t

(R7) Arquitectura de marcas (ordenamiento de precios)
     Σ_k y_jtk·p_jk ≤ Σ_k y_(j+1)tk·p_(j+1)k    ∀ j, t

(R8) Capacidad / inventario por marca
     Σ_i x_ijt · π_it · q_ijt · λ_ijt ≤ Cap_jt  ∀ j, t

(R9) Coherencia precio–promoción
     z_jt = 1 ⟹ el precio elegido pertenece al tramo bajo de K_j
```

Nota sobre **(R7)**: los rangos de precio observados se solapan (§2.4), así que el ordenamiento estricto es una **decisión de modelado** para proteger el posicionamiento de marca, no un hecho del dato. Se implementa como restricción activable y se reporta el costo de imponerla.

### 4.7 Tratabilidad y plan de resolución

El problema es un MINLP: `π` y `q` dependen de forma no lineal de los precios, que son variables de decisión. Se vuelve tratable con una decisión clave: **el precio no es continuo, es una escalera discreta ya observada en el dato** (21 a 44 puntos por marca, §2.4).

Al discretizar, `π`, `q` y `λ` se **precalculan** con los modelos de ML para cada escenario de precio y promoción, y entran al optimizador como coeficientes constantes. El MINLP se convierte en un **MILP** resoluble con solvers estándar (PuLP / OR-Tools / Pyomo).

Entrega en tres sub-problemas de complejidad creciente, cada uno útil por sí solo:

| # | Sub-problema | Decisión | Estructura | Resolución |
|---|---|---|---|---|
| **P1** | **Targeting** (precios y promos dados) | `x_ijt` | Knapsack / asignación con presupuesto y fatiga | LP relajado es integral → greedy por margen esperado es **óptimo**; sirve de base de comparación |
| **P2** | **Calendario promocional** | `z_jt` | MILP con R5 | Solver exacto; se apoya en los lifts de §2.5 |
| **P3** | **Precio** | `y_jtk` | MILP sobre la escalera discreta, con R6–R7 | Solver exacto o búsqueda exhaustiva sobre la escalera |

**P1 es el modelo base entregable**: tiene solución exacta, es interpretable y produce de inmediato la cifra que el negocio quiere — a quién contactar y cuánto margen esperado genera. P2 y P3 lo extienden hacia pricing, que es el skill explícitamente evaluado.

### 4.8 Supuestos declarados y sus límites

Tres honestidades que el documento final debe mantener visibles:

1. **No hay datos de costo.** `c_j`, `a`, `d_j` y `B` no están en el dataset. Se parametrizan como supuestos explícitos (margen bruto por marca, costo de contacto) y el resultado se entrega como **análisis de sensibilidad** sobre ellos, no como una cifra única. El valor del modelo está en la política de decisión, que es robusta al nivel exacto del costo, no en el monto absoluto.

2. **Los precios son observacionales, no experimentales.** No hubo variación aleatoria de precios, así que las elasticidades estimadas son **asociativas**. Pueden estar confundidas con estacionalidad o con la lógica comercial que fijó esos precios. Por eso la recomendación de pricing se entrega con su intervalo y se propone validarla con **A/B test** antes de desplegarla a toda la cartera.

3. **Riesgo de *predict-then-optimize*.** El optimizador trata `π`, `q` y `λ` como ciertos, y amplifica el error de predicción: elige sistemáticamente las celdas donde el modelo es más optimista. Mitigación: usar cotas inferiores conservadoras de las probabilidades (percentil bajo del intervalo) en el objetivo, y evaluar la política resultante frente a la política *naïve* bajo el modelo verdadero del conjunto de test.

---

## 5. La capa de Machine Learning que alimenta la optimización

### 5.1 Interfaz entre las dos capas

| Modelo | Tipo | Target | Universo de entrenamiento | Parámetro que entrega |
|---|---|---|---|---|
| **M0 Segmentación** | K-Means + PCA | — | 500 clientes, demografía | `segmento_i`: unidad de acción y feature |
| **M1 Incidencia** | Clasificación binaria | `incidencia_compra` | 58,693 visitas | `π_it` |
| **M2 Elección de marca** | Clasificación multiclase | `id_marca` ∈ {1..5} | 14,638 compras | `q_ijt` y **elasticidades cruzadas** |
| **M3 Cantidad** | Regresión / conteo | `cantidad` ≥ 1 | 14,638 compras | `λ_ijt` |

**M1 es el modelo de propensión que pide el enunciado.** M2 y M3 existen porque la función objetivo (§4.5) los necesita: sin M2 no hay elasticidad cruzada ni análisis de canibalización, y sin M3 el margen esperado no se puede calcular.

### 5.2 Selección y explicabilidad
Progresión justificada por ganancia medida: **logística regularizada** (baseline interpretable, y fuente directa de los coeficientes de elasticidad para el pricing) → **Random Forest** → **Gradient Boosting** (XGBoost / LightGBM) con tuning por validación cruzada. Explicabilidad con **SHAP** global y local, más lectura en *odds ratio* del modelo logístico. Para M2, un **logit multinomial** tiene la ventaja de que sus coeficientes de precio **son** las elasticidades que la optimización necesita: la interpretabilidad no es un extra, es un requisito funcional de la capa prescriptiva.

### 5.3 Validación, dada la estructura del panel
- **Split temporal**: train días 1–550 (45,497 filas, incidencia 24.98%) / test días 551–730 (13,196 filas, incidencia 24.80%). Verificado: ambos lados conservan los 500 clientes y la misma tasa base, así que el corte no introduce sesgo de composición.
- **`GroupKFold` por `id`** en paralelo, para medir generalización a **clientes nuevos** — con solo 500 clientes es el riesgo real de sobreajuste.
- Un `train_test_split` aleatorio sería inválido: con 117 visitas por cliente, repartiría visitas del mismo cliente entre train y test e inflaría las métricas.
- **Métrica principal PR-AUC** (desbalance 25/75), con ROC-AUC de apoyo.
- **Umbral de operación derivado del costo**, no fijado en 0.5: el umbral óptimo es el que resuelve P1, es decir el que iguala margen esperado y costo de contacto. La optimización, no la convención, fija el punto de corte.

### 5.4 Baselines a superar
Ningún modelo se acepta sin batir estas dos referencias:

1. **Tasa base**: predecir siempre la clase mayoritaria → 75.1% de *accuracy* y cero valor. Es la razón para no reportar *accuracy*.
2. **Targeting histórico**: ordenar clientes por tasa de compra pasada da **lift ×2.43** en el decil superior (§2.6). El modelo tiene que superar esto para justificar su existencia.

---

## 6. Cómo se conecta con la captura de valor

El modelo de optimización define la métrica de valor de forma natural: **la diferencia entre el margen esperado de la política óptima y el de la política actual**, evaluada sobre el período de test.

| Indicador | Definición | Fuente |
|---|---|---|
| Lift de conversión | Tasa de compra del decil superior / tasa base | M1 sobre test |
| Margen incremental por targeting | Objetivo de P1 óptimo − objetivo de la política actual | P1 |
| Ahorro promocional | Margen liberado al reasignar promoción de marcas 1–2 hacia 3 y 5 | P2 + §2.5 |
| Margen incremental por pricing | Objetivo de P3 óptimo − margen al precio observado | P3 |
| Concentración de esfuerzo | % de ingreso cubierto al contactar el top 20% | §2.6: 52.7% |
| Validez causal | Ingreso incremental medido contra grupo de control | A/B test |

El último es el que convierte una proyección en una captura demostrada. Sin contrafactual, las cifras anteriores son estimaciones del modelo, y el documento final las presentará como tales.

---

## 7. Resumen de decisiones de estructuración

1. **El modelo base es de optimización, no de clasificación.** Objetivo: margen de contribución esperado. Decisiones: precio, promoción, targeting. El clasificador es un insumo suyo.
2. **La cadena de tres modelos** (`π · q · λ`) no es elaboración: es exactamente lo que la función objetivo necesita para calcular margen esperado.
3. **El precio se modela como escalera discreta ya observada** (21–44 puntos por marca). Es lo que hace el problema un MILP tratable en lugar de un MINLP.
4. **P1 (targeting) se entrega como modelo base resoluble de forma exacta**; P2 y P3 lo extienden hacia pricing.
5. **El umbral de decisión sale de la optimización**, no de la convención 0.5.
6. **Validación temporal y por cliente**, nunca aleatoria: 500 clientes × 117 visitas lo exigen.
7. **`ultima_marca_comprada` es lag verificado y usable**; `ultima_cantidad_comprada` es binaria y está mal nombrada en el diccionario.
8. **Los supuestos de costo y la naturaleza observacional de los precios se declaran**, y los resultados se entregan con sensibilidad y propuesta de validación experimental.

---

## Anexo · Consultas de verificación

Todas las cifras de este brief provienen de `data/compras_data.csv` y son reproducibles. El notebook de la Fase 1 debe reproducirlas como primera celda de control, para que cualquier discrepancia posterior sea detectable de inmediato.
