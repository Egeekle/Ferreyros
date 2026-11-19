# Brief de Evaluación · Alineamiento del Tratamiento de Datos y EDA

**Prueba Técnica Data Scientist · Ferreycorp**
Instrumento para evaluar si el EDA y el procesamiento de datos sirven al problema estructurado
Documento de referencia evaluado: [`docs/brief-problema.md`](brief-problema.md)
Fecha de aplicación: _______ · Evaluador: _______

---

## 1. Qué evalúa este instrumento

No evalúa si el EDA es **completo**. Evalúa si es **alineado**.

El brief de problema define un modelo de optimización cuyo objetivo es
`max Σ x·π·q·λ·m − costos`, con variables de decisión de precio (`y_jtk`), promoción (`z_jt`) y targeting (`x_ijt`), más nueve restricciones. Todo ese aparato necesita parámetros. El EDA y el tratamiento de datos existen para **producir, validar o acotar esos parámetros** — no para describir el dataset.

De ahí el criterio rector de esta evaluación:

> **Principio de trazabilidad.** Todo output del EDA debe poder vincularse a un parámetro, una variable de decisión o una restricción del problema de optimización. Un gráfico o estadístico sin ese vínculo no suma: resta, porque consume espacio del entregable y atención del evaluador.

Un EDA técnicamente impecable que no toque las elasticidades, la heterogeneidad de clientes ni la escalera de precios **falla esta evaluación**, aunque tenga veinte gráficos correctos.

---

## 2. Matriz de trazabilidad

La columna *Evidencia exigida* es lo que el EDA debe producir. La columna *Criterio de aceptación* es lo que el evaluador verifica. Marcar `✔` / `✘` / `Parcial`.

### 2.1 Parámetros de la función objetivo

| # | Elemento del problema | Evidencia exigida en el EDA | Tratamiento de datos requerido | Criterio de aceptación | Estado |
|---|---|---|---|---|---|
| A1 | `π_it` — probabilidad de compra (Modelo M1) | Tasa de compra global y su estabilidad temporal | Target `incidencia_compra` sin transformar | Reporta **24.94%**; muestra estabilidad por semestre (23.4 / 25.5 / 26.3 / 24.8%) y concluye que no hay drift | ☐ |
| A2 | `π_it` — sensibilidad al precio | Tasa de compra por quintil del precio mínimo del día | Construida `precio_min_dia` | Muestra la caída **28.3% → 19.2%** y la nombra como señal de elasticidad a estimar | ☐ |
| A3 | `π_it` — sensibilidad a promoción | Tasa de compra por número de marcas en promoción | Construida `n_promos_dia` | Muestra **21.8% → 29.0%** y observa la no-monotonía en 3 promos | ☐ |
| A4 | `q_ijt` — elección de marca (Modelo M2) | Share de cada marca dado que compra | Universo restringido a las 14,638 compras | Shares por marca reportados; **el universo condicional está explícito** | ☐ |
| A5 | `q_ijt` — sustitución y canibalización | Lift del share de cada marca con promo vs. sin promo; matriz de cambio de marca | `ultima_marca_comprada` validada como lag | Reporta los lifts **×1.08 / ×1.39 / ×1.86 / ×1.16 / ×1.66** y concluye que 3 y 5 son las más elásticas | ☐ |
| A6 | `λ_ijt` — cantidad esperada (Modelo M3) | Distribución de `cantidad` condicionada a compra | Universo condicional; `cantidad=0` excluida | Media **2.77**, rango 1–15, concentración en 1–3 (79% de las compras) | ☐ |
| A7 | `m_jk` — margen unitario | Precio efectivamente pagado e ingreso por marca | Construida `precio_pagado` = precio de la marca comprada | Ingreso por marca reportado; **declara explícitamente que falta `c_j`** | ☐ |

### 2.2 Variables de decisión

| # | Elemento del problema | Evidencia exigida en el EDA | Tratamiento de datos requerido | Criterio de aceptación | Estado |
|---|---|---|---|---|---|
| B1 | `y_jtk` — escalera de precios | Enumeración de puntos de precio distintos por marca | Menú discreto de precios por marca | Reporta **37 / 30 / 21 / 26 / 44** puntos y sus rangos; justifica la discretización como lo que vuelve tratable el MILP | ☐ |
| B2 | `y_jtk` — dinámica de precios | Series temporales de los 5 precios; correlación entre marcas | Sin imputación (no hay nulos) | Muestra las 5 trayectorias y evalúa si los precios se mueven juntos | ☐ |
| B3 | `z_jt` — calendario promocional | Frecuencia de promoción por marca | Features de promoción | Reporta **34.4 / 31.6 / 4.3 / 11.8 / 3.6%** y contrasta frecuencia contra respuesta (A5) | ☐ |
| B4 | `x_ijt` — targeting | Heterogeneidad de la tasa de compra entre clientes | Agregación a nivel cliente | Reporta el rango **4.6% – 92.2%** y la mediana 18.4% | ☐ |
| B5 | `x_ijt` — concentración de valor | Curva de Pareto del ingreso por cliente | `gasto` = precio pagado × cantidad | Top 10% = **37.0%** del ingreso; top 20% = **52.7%** | ☐ |
| B6 | `x_ijt` — baseline a superar | Lift del targeting por tasa histórica | Ordenamiento por tasa de compra pasada | Reporta **lift ×2.43** en el decil superior y lo declara el piso que el modelo debe batir | ☐ |

### 2.3 Restricciones

| # | Restricción | Evidencia exigida en el EDA | Criterio de aceptación | Estado |
|---|---|---|---|---|
| C1 | R5 · tope de días en promoción | Frecuencia observada de promoción por marca | Cubierto por B3; los valores observados acotan `T_j` | ☐ |
| C2 | R6 · piso de margen | — | **Declara que `c_j` y `ρ` no están en el dato** y que serán supuestos con sensibilidad | ☐ |
| C3 | R7 · arquitectura de marcas | Comparación de rangos de precio entre marcas | Detecta el **solapamiento** (marca 4 baja a 1.76, bajo el mínimo de la 3) y concluye que el ordenamiento estricto es decisión de modelado, no hecho del dato | ☐ |
| C4 | R8 · capacidad / inventario | Unidades vendidas por marca y por día | Reporta **6.46 / 17.57 / 4.10 / 12.16 / 15.34** unidades/día | ☐ |
| C5 | R3–R4 · fatiga de contacto | Frecuencia de visita por cliente y por día | Visitas/cliente 117.4 (72–358); visitas/día 80.4 (6–179) — acota `F_i` y la escala operativa | ☐ |

### 2.4 Segmentación (M0)

| # | Elemento | Evidencia exigida | Criterio de aceptación | Estado |
|---|---|---|---|---|
| D1 | Variables de segmentación | Perfil demográfico de los 500 clientes | Las 6 variables perfiladas con su cardinalidad real | ☐ |
| D2 | Demografía estática | Verificación de invarianza por cliente | Confirma **0 clientes** con más de un valor en edad, ingreso, género, estado civil, educación u ocupación → justifica agregar a nivel cliente | ☐ |
| D3 | Poder predictivo de la demografía | Tasa de compra por nivel demográfico | Muestra que las diferencias son modestas (educación 29.8% vs 18.1%; ocupación 28.1% vs 20.2%) y concluye que **la demografía segmenta pero no predice** | ☐ |

---

## 3. Auditoría de tratamiento de datos

Estos puntos son **eliminatorios**: uno fallado invalida los resultados del modelo aunque las métricas se vean bien.

| # | Control | Verificación | Severidad | Estado |
|---|---|---|---|---|
| E1 | `tamanio_ciudad` excluida | No aparece en la matriz de features | **Bloqueante** — indicación explícita del diccionario | ☐ |
| E2 | `id_marca` fuera de las features de M1 | Es resultado de la visita, no condición previa | **Bloqueante** — fuga de datos | ☐ |
| E3 | `cantidad` fuera de las features de M1 | Mismo argumento | **Bloqueante** — fuga de datos | ☐ |
| E4 | `ultima_marca_comprada` validada como lag | Se verificó que coincide con `shift(1)` por cliente ordenado por `(id, dia_visita)` | **Bloqueante** — si no se valida, se usa a ciegas | ☐ |
| E5 | `ultima_cantidad_comprada` reinterpretada | Se documenta que es **binaria (0/1)**, no una cantidad, y que el diccionario la nombra mal | Alta — error de interpretación propagado | ☐ |
| E6 | Features de historial estrictamente retrospectivas | Toda ventana acumulada usa `shift` o expansión hacia el pasado, nunca la fila actual | **Bloqueante** — fuga temporal | ☐ |
| E7 | `ultima_marca_comprada = 0` tratada como categoría | Las 44,133 filas son ausencia de historial, no nulos; **no se imputan** | Alta | ☐ |
| E8 | Split temporal, no aleatorio | Corte en `dia_visita ≤ 550` / `> 550` | **Bloqueante** — un split aleatorio reparte visitas del mismo cliente entre train y test | ☐ |
| E9 | Split verificado sin sesgo de composición | Train 45,497 filas al 24.98%; test 13,196 al 24.80%; 500 clientes en ambos lados | Alta | ☐ |
| E10 | Escalado y codificación ajustados solo en train | `Pipeline` con `fit` sobre train | **Bloqueante** — fuga por preprocesamiento | ☐ |
| E11 | Consistencia lógica verificada | 0 violaciones de `compra=1 ⟹ cantidad≥1 ∧ marca∈{1..5}` y su recíproca | Media | ☐ |
| E12 | Universo condicional correcto en M2 y M3 | Entrenados sobre las 14,638 compras, no sobre las 58,693 filas | **Bloqueante** — target contaminado con no-compras | ☐ |
| E13 | Reproducibilidad | El notebook corre de extremo a extremo sin intervención | **Bloqueante** — Ferreycorp ejecutará la solución | ☐ |

---

## 4. Celda de control: cifras de referencia

El EDA debe reproducir estos valores como primera celda verificable. Cualquier discrepancia señala un error de carga, de filtro o de agregación **antes** de que contamine el modelado.

| Cifra | Valor esperado |
|---|---|
| Dimensiones | 58,693 × 24 |
| Clientes únicos | 500 |
| Días distintos | 730 (rango 1–730) |
| Nulos / duplicados | 0 / 0 |
| Tasa de compra | 0.2494 (14,638 compras) |
| Ingreso total observado | 82,756.58 |
| Cantidad media dado compra | 2.77 |
| Share de ingreso por marca | 7.6 / 26.3 / 7.3 / 23.0 / 35.8 % |
| Puntos de precio por marca | 37 / 30 / 21 / 26 / 44 |
| Frecuencia de promoción | 0.3438 / 0.3156 / 0.0428 / 0.1179 / 0.0359 |
| Lift de promo en share | ×1.08 / ×1.39 / ×1.86 / ×1.16 / ×1.66 |
| Incidencia por quintil de precio mínimo | 0.2827 → 0.1922 |
| Incidencia por n_promos (0→2) | 0.2184 → 0.2901 |
| Tasa de compra por cliente (mín/mediana/máx) | 0.046 / 0.184 / 0.922 |
| Pareto top 10% / 20% | 37.0% / 52.7% |
| Lift targeting decil superior | ×2.43 |
| Split 550: filas e incidencia | 45,497 @ 0.2498 / 13,196 @ 0.2480 |

---

## 5. Rúbrica de evaluación

Seis dimensiones, 5 puntos cada una. **Umbral de aprobación: 21/30, con ninguna dimensión bajo 3 y cero bloqueantes abiertos.**

| Dimensión | 1–2 (insuficiente) | 3 (aceptable) | 4–5 (sólido) | Nota |
|---|---|---|---|---|
| **Trazabilidad** | EDA descriptivo genérico; no menciona el problema de optimización | La mayoría de los outputs se vinculan a un parámetro | Cada output declara explícitamente qué parámetro, variable o restricción alimenta | ☐ |
| **Cobertura de parámetros** | Falta evidencia para `π`, `q` o `λ` | Los tres cubiertos, alguno superficialmente | Los tres cubiertos con la evidencia de §2.1 completa | ☐ |
| **Integridad de la decisión** | Ignora precio, promoción o targeting | Cubre los tres de forma básica | Cuantifica la desalineación promocional y la trata como hipótesis de arbitraje | ☐ |
| **Rigor anti-fuga** | Uno o más bloqueantes de §3 fallados | Sin bloqueantes, pero sin validación documentada | Fuga auditada y documentada, incluida la verificación del lag | ☐ |
| **Honestidad analítica** | Presenta correlaciones como causalidad; no declara faltantes | Menciona las limitaciones | Declara los 3 supuestos (costos, precios observacionales, predict-then-optimize) con su mitigación | ☐ |
| **Reproducibilidad y narrativa** | No corre; o corre sin conclusiones | Corre; conclusiones presentes | Corre de extremo a extremo, celda de control incluida, cada hallazgo con su consecuencia declarada | ☐ |

**Total: ___ / 30**

---

## 6. Señales de desalineación

Si el EDA presenta cualquiera de estas, hay un problema de alineamiento, no de técnica:

| Señal | Por qué es desalineación |
|---|---|
| Mapa de correlación de las 24 columnas sin conclusión | Mezcla targets con features y resultados con condiciones; no decide nada |
| Histograma de cada columna, una por una | Descripción sin pregunta; ninguna celda del problema lo pide |
| `accuracy` como métrica reportada | 75.1% se obtiene prediciendo siempre "no compra": el número oculta el fracaso |
| `train_test_split` aleatorio | Con 117 visitas por cliente, infla las métricas de forma irreparable |
| `ultima_marca_comprada = 0` imputada con la moda | Destruye la información "sin historial previo", que es señal real |
| `tamanio_ciudad` analizada o modelada | Contradice una indicación explícita del diccionario |
| Gráficos de precio sin conectar con incidencia o share | El precio es **variable de decisión**: describirlo sin medir respuesta no sirve al optimizador |
| EDA que no menciona la heterogeneidad entre clientes | Sin ella, el problema de targeting no tiene razón de existir |
| Promoción tratada con efecto uniforme entre marcas | Los lifts van de ×1.08 a ×1.86: el supuesto uniforme borra el hallazgo central |
| Conclusiones de pricing presentadas como causales | Los precios son observacionales; sin esa advertencia la recomendación es insostenible |

---

## 7. Preguntas de cierre para el evaluador

Si el EDA no permite responder estas seis preguntas, no está alineado con el problema — independientemente de su calidad técnica:

1. ¿Cuánto cae la probabilidad de compra si sube el precio? → alimenta `π`
2. ¿Qué marca gana share cuando se la promociona, y a costa de cuál? → alimenta `q` y la canibalización
3. ¿Cuántas unidades espero vender si el cliente compra? → alimenta `λ`
4. ¿Qué clientes concentran el ingreso y qué tan distintos son entre sí? → alimenta `x_ijt`
5. ¿Qué precios puedo efectivamente elegir por marca? → alimenta `y_jtk`
6. ¿Qué parámetros del problema **no** están en el dato? → delimita el alcance honesto de la solución

---

## 8. Resultado de la evaluación

| Campo | |
|---|---|
| Bloqueantes abiertos (§3) | ___ |
| Puntaje de rúbrica (§5) | ___ / 30 |
| Celda de control reproduce §4 | ☐ Sí ☐ No ☐ Parcial |
| Señales de desalineación detectadas (§6) | ___ |
| **Veredicto** | ☐ Alineado ☐ Alineado con observaciones ☐ Requiere corrección |

Observaciones:

---

> **Nota de alcance.** Este instrumento evalúa alineamiento entre el tratamiento de datos, el EDA y el problema estructurado. No evalúa el desempeño de los modelos de ML, que se juzga con las métricas de §5.3 del brief de problema (PR-AUC sobre split temporal más `GroupKFold` por cliente), ni la resolución del MILP.
