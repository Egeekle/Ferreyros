# Fase 2 · Procesamiento y feature engineering

**Objetivo:** cerrar la Fase 2 de `docs/brief-metodologia.md` — dataset modelable con
features sin fuga y preprocesamiento ajustado solo en train (control E10, el único
bloqueante que el notebook 01 dejó PENDIENTE).
**Rama:** master · **Entrega:** D2 del cronograma (18 nov) · **Runner:** sin pytest en el
proyecto; la verificación son frames de veredicto (idioma ya establecido:
`verify_no_leakage`, `consistency_checks`) ejecutados y aseverados en el notebook.

## Specs

- **S1** — Las estadísticas de referencia de las features deben ajustarse **solo sobre
  train**. Hoy tres grupos se calculan sobre train+test y eso es fuga de preprocesamiento:
  `desviacion_precio_j` usa `out[col].mean()` global, `descuento_implicito_j` usa
  `out[col].max()` global, y `edad_bin` / `ingreso_bin` usan `pd.qcut` sobre todo el frame.
- **S2** — El cambio no debe romper la llamada existente del notebook 01
  (`features.build_features(dfs)` sin argumentos).
- **S3** — Las features de historial se siguen construyendo sobre la línea de tiempo
  completa y estrictamente hacia el pasado: el historial de un cliente en test **sí** puede
  usar sus visitas de train. Eso es información pasada, no fuga (control E6).
- **S4** — Brief metodología Fase 2: «Codificación y escalado dentro de un `Pipeline` de
  scikit-learn, ajustado **solo** sobre train.» Debe existir ese Pipeline.
- **S5** — Brief alineamiento E7: las 44,133 filas sin historial «son ausencia de historial,
  no nulos; **no se imputan**». La ausencia debe llegar al modelo como información, no
  borrada con la media.
- **S6** — El control **E10** («Escalado y codificación ajustados solo en train»,
  Bloqueante) debe pasar de PENDIENTE a OK con evidencia ejecutada, no declarada.
- **S7** — Notebook de Fase 2 en el estilo de la casa: la lógica vive en `src/`, el notebook
  orquesta, y cada bloque cierra con su **consecuencia** declarada.

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S1, S2, S3 | inline | hecho | `features.FeatureReference` + `fit_reference` + `verify_reference_fit`: 12 estadísticos ajustados en train, 8 difieren del valor sobre train+test. Check diferencial: 9 de 68 columnas cambian, todas intencionadas, 0 inesperadas |
| T2 | S4, S5 | inline | hecho | `preprocess.prepare()`: 60 → 83 columnas, 0 nulos, 10 indicadores de ausencia, 16 niveles one-hot |
| T3 | S6 | inline | hecho | `audit.treatment_audit(..., prepared=...)`: 13/13 controles OK, 0 pendientes, 0 bloqueantes fallados |
| T4 | S7 | inline | hecho | `notebooks/02_procesamiento.ipynb`: 36 celdas, 14 de código, ejecutado de extremo a extremo con 0 errores y todos los `assert` en verde |

## Log

**L1** (petición original, 2026-11-15): «sigue la fase 2 de feature engineering»

**L2** — Estado al empezar: `src/ferreyros/features.py` ya existe y corre limpio sobre
pandas 3.1.1 (las 5 familias, 68 columnas, `verify_no_leakage` con 0 violaciones). Lo que
falta de la Fase 2 es el preprocesamiento: no hay `Pipeline`, y el notebook 01 cierra
diciendo que «E10 queda pendiente por diseño».

**L3** — Hallazgo propio durante la exploración: tres estadísticas de `features.py` se
ajustan sobre el frame completo (medias y máximos de precio, bordes de los quintiles
demográficos). Nadie lo pidió, pero contradice E10, así que entra como S1.

**L4** — Engram no está disponible en esta sesión (los tools MCP no cargaron): el espejo
`odd/fase-2-feature-engineering/tasks` queda **pendiente de sincronizar**.

**L5** — T1 cerrado. El check diferencial contra la lógica anterior confirma que solo cambian
las 9 columnas previstas (`desviacion_precio_1..5`, `descuento_implicito_3`, `_5`, `edad_bin`,
`ingreso_bin`). Las marcas 1, 2 y 4 no cambian porque ya alcanzan su precio máximo en train.
El hallazgo material: el precio máximo de la marca 3 es 2.07 en train y 2.14 en todo el frame,
y el de la marca 5 pasa de 2.71 a 2.80. Esos precios solo existen en test, así que el
`descuento_implicito` de **cada fila de train** se estaba midiendo contra un precio futuro.
Fuga real, invisible en la matriz de features porque ninguna columna prohibida aparecía.

**L6** — T2 y T3 cerrados. `preprocess.verify_train_only_fit` prueba E10 con cuatro
aserciones; la segunda es la que importa: las 48 medias del escalador cambiarían si el ajuste
incluyera test, de modo que el límite no es declarativo. E10 pasa de PENDIENTE a OK. La firma
de `treatment_audit` queda retrocompatible: sin `prepared`, E10 vuelve a PENDIENTE y el
notebook 01 sigue corriendo igual.

**L7** — T4 cerrado. Ambos notebooks reejecutados de extremo a extremo (0 errores): el 01
porque los valores de 9 features cambiaron y sus outputs guardados habrían quedado
desalineados con el código, y su nota de cierre ahora apunta a la Fase 2 en lugar de dejar E10
abierto.

**L8** — Hallazgo no previsto que condiciona la Fase 4: 16 de las 60 features tienen valores de
test fuera del rango de train. `semana` (98%) y `mes` (88%) son índices de tendencia que un
modelo lineal extrapolaría; el precio de la marca 5 sube por encima de todo lo visto en train y
afecta al 79.6% de test. Queda declarado en `preprocess.out_of_range_report`, no escondido. La
decisión sobre `semana`/`mes` se toma en la Fase 4 con la métrica en la mano.

**L9** — Riesgo: se evaluó el ítem 3 (contratos que otros consumen) y se descarta con
evidencia. Las firmas nuevas son parámetros opcionales con default seguro, el notebook 01
reejecuta limpio, y el check diferencial acota el cambio de comportamiento a las 9 columnas
previstas. El cambio de valores era el objetivo de S1, no un efecto colateral.

**L10** — Sin commit: el árbol de trabajo ya traía cambios sin confirmar de la sesión anterior
(`docs/brief-alineamiento-eda.md` en stage, los módulos de la Fase 1 sin seguimiento). Agruparlos
en un commit de Fase 2 mezclaría trabajo ajeno, así que la decisión queda para Jessie.
