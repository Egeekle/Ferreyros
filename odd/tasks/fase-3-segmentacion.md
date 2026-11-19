# Fase 3 · Segmentación de clientes (modelo M0)

**Objetivo:** cerrar la Fase 3 de `docs/brief-metodologia.md` — K-Means sobre la demografía a
nivel cliente, con `k` elegido por codo y silhouette, PCA para interpretar, cada segmento con
nombre de negocio y perfil de compra, y el segmento disponible como feature de M1 y como
unidad de acción comercial.
**Rama:** master · **Entrega:** D2 del cronograma (18 nov) · **Runner:** sin pytest; la
verificación son frames de veredicto ejecutados y aseverados en el notebook.

## Specs

- **S1** — Brief metodología Fase 3: «K-Means sobre las variables demográficas, con
  estandarización previa y selección de *k* por método del codo y *silhouette*».
- **S2** — Brief metodología Fase 3: «PCA para visualización e interpretación de componentes».
- **S3** — Brief metodología Fase 3: «Cada segmento recibe un nombre de negocio y un perfil de
  comportamiento de compra». El nombre debe derivarse del centroide por regla reproducible, no
  inventarse en prosa.
- **S4** — Brief metodología Fase 3: «El segmento se incorpora como feature del modelo
  supervisado y, sobre todo, como **unidad de acción comercial**: las campañas se diseñan por
  segmento, no por cliente individual».
- **S5** — Brief problema §5: M0 opera sobre los **500 clientes** con demografía, y entrega
  `segmento_i` como «unidad de acción y feature». El universo es el cliente, no la visita.
- **S6** — Brief alineamiento D3 y Fase 1: «la demografía segmenta pero no predice» (amplitud
  demográfica ~0.12 frente a 0.876 entre clientes). La Fase 3 no puede contradecir eso: debe
  medir cuánto separa el segmento en comportamiento y declararlo, incluso si es poco.
- **S7** — Las variables de comportamiento quedan **fuera** de la matriz de clustering
  (`eda.segments.segmentation_matrix` ya las excluye a propósito): agrupar por comportamiento y
  luego explicar comportamiento por grupo es circular.
- **S8** — Reproducibilidad: `k`, semilla, orden de las etiquetas y el mapa cliente→segmento se
  congelan en `artifacts/segmentos.json` con huella digital, como la Fase 0 congeló el split.
  Un `segmento 3` tiene que significar lo mismo en la Fase 4 que aquí.
- **S9** — El encuadre congelado de la Fase 0 sigue mandando: la matriz de M0 no puede contener
  columnas prohibidas ni información de test. Debe verificarse, no declararse.
- **S10** — Estilo de la casa: la lógica vive en `src/`, el notebook orquesta, cada bloque
  cierra con su consecuencia, y el notebook corre de extremo a extremo con 0 errores.

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S1, S7, S9 | inline | hecho | `segmentation.py`: matriz de 500 x 8 con `verify_design` en 5/5; `select_k` (codo k=3, silhouette max k=9) + regla de empate declarada + `stability_scan` (ARI 1.0000 en k=3 vs 0.7866 en k=9) |
| T2 | S3, S5, S6 | inline | hecho | `segment_profile`: los 3 segmentos son 100% un nivel de `ocupacion`; nombres por regla (>=80% de una ocupacion la nombra). `segment_behaviour`: amplitud de tasa 0.0795 con IC95 sin solaparse, frente a 0.876 entre clientes (D3 confirmado) |
| T3 | S2, S4 | inline | hecho | `pca_projection` (52.7% en 2 componentes) + `pca_loadings`; `segment_brand_share` con desvio maximo de 23.6 pp; `segment_promo_response` 3/3 responden; `prepare_with_segment`: 83 -> 86 columnas |
| T4 | S8, S10 | inline | hecho | `artifacts/segmentos.json` con huella `c22163cf686ccf32` + `verify_frozen` en 5/5; `notebooks/03_segmentacion.ipynb`: 46 celdas, 20 de codigo, 5 figuras, 0 errores |

## Log

**L1** (petición original, 2026-11-18): «sigue con la fase 3 sobre el modelo , metodologia y
problema»

**L2** — Estado al empezar: las Fases 0, 1 y 2 están cerradas con sus notebooks (`00_encuadre`,
`01_eda`, `02_procesamiento`). El EDA ya dejó preparados los dos insumos de M0:
`eda.segments.segmentation_matrix` (demografía a nivel cliente) y
`eda.segments.behaviour_by_client_for_validation` (comportamiento, deliberadamente aparte).
Lo que no existe es el modelo.

**L3** — Engram no está disponible en esta sesión (los tools MCP no cargaron): el espejo
`odd/fase-3-segmentacion/tasks` queda **pendiente de sincronizar**.

**L4** — `k = 3` elegido con los dos criterios del brief **y** un tercero que decide. Codo en
k=3; silhouette maxima en k=9 (0.3820 vs 0.3534). La regla de empate esta escrita en el codigo
antes de ver los segmentos (`TOLERANCIA_SILHOUETTE = 0.03`: entre los k a menos de 0.03 del
maximo, el mas pequenio). Lo que cierra la decision es la reproducibilidad: ARI medio **1.0000**
en k=3 sobre 20 remuestreos al 80%, contra **0.7866 con minimo 0.5606** en k=9. Un segmento que
cambia de integrantes segun la muestra no sirve para planificar una campania.

**L5** — Hallazgo estructural: los 3 segmentos son **exactamente** los 3 niveles de `ocupacion`
(100% cada uno), y ingreso (90.9k / 127.2k / 183.7k), edad (36.5 / 38.9 / 43.8) y educacion
(0.97 / 1.10 / 1.40) se ordenan detras. El clustering separo un eje socioeconomico y la ocupacion
es su marcador mas limpio. Por eso la regla de nombres pone la ocupacion primero cuando un
segmento es >=80% de un nivel: nombrarlo solo por ingreso esconderia por donde corto el modelo.

**L6** — Pregunta incomoda resuelta con numeros en lugar de defendida: `ocupacion` one-hot aporta
3 de las 8 dimensiones, asi que habia que descartar que los segmentos salieran de ese peso.
`encoding_sensitivity` reajusta con las alternativas: el ARI entre la particion elegida y la de
*sin ocupacion* es **0.0273** — particiones sin relacion. La codificacion **decide** los
segmentos, y eso queda declarado. La variante one-hot se conserva porque gana en las dos cosas
que se le van a pedir: silhouette (0.3534 vs 0.3316 y 0.2702) y separacion de comportamiento
(0.0795 vs 0.0655 y 0.0758). No se puede afirmar que estos sean "los" segmentos del negocio.

**L7** — El segmento vale por la accion comercial, no por la prediccion, y ahora esta medido.
Separa 0.0795 de tasa de compra (IC95 sin solaparse) frente a 0.876 entre clientes individuales:
confirma D3. Pero separa hasta **23.6 pp** de preferencia de marca (marca 2 es el 54.6% de las
compras del segmento 1 y el 12.3% del 3; la marca 5 es el 43.5% del segmento 2 y el 12.9% del 1).
Eso da destinatario al arbitraje de la marca 5 que encontro la Fase 1: segmentos 2 y 3, no el 1.

**L8** — Tension declarada para P1/P2: los 3 segmentos responden a promocion (+4.36 a +5.55 pp),
y el que mas responde es el segmento 1 (+5.55 pp, z=10.0), que es el de tasa base mas baja y
menor ticket. Sin costo de contacto ni margen en el dato, la tension se declara; con ellos, el
optimizador decide.

**L9** — Integracion como feature con dos cambios minimos y retrocompatibles: `preprocess.prepare`
acepta `extra` (el hook ya existia en `features.feature_matrix`, que es una lista blanca) y
`segmento` entra en `NOMINAL_COLS`. Sin adjuntarlo, todo lo anterior corre igual: `prepare()`
sigue dando 83 columnas. Con el, 86, y tanto el encuadre de la Fase 0 (7/7) como el control E10
de la Fase 2 siguen verificados. Check diferencial: `02_procesamiento.ipynb` reejecutado de
extremo a extremo, 14 celdas de codigo, 0 errores.

**L10** — Sin commit: el repositorio arrastra trabajo sin versionar de sesiones anteriores y nadie
pidio commitear.
