# Fase 6 · Arquitectura y despliegue (Cloud Computing)

**Objetivo:** cerrar la Fase 6 de `docs/brief-metodologia.md` — arquitectura de referencia con la
nube como decisión y no como adorno, más la pieza que demuestra que el despliegue es real: el
scoring ejecutable. Ampliada a pedido del usuario con **PoC, mejoras futuras y combinación
multi-cloud** (GCP/Azure/AWS) con cluster y orquestación como eje.
**Rama:** master · **Runner:** sin pytest; la verificación es ejecutar los scripts.

## Specs

- **S1** — Brief metodología Fase 6: arquitectura de ingesta → lake → feature store →
  entrenamiento → registry → inferencia (batch + REST) → consumo → monitoreo.
- **S2** — «El reentrenamiento se dispara por calendario o por degradación detectada, **no de
  forma manual**.»
- **S3** (pedido del usuario, 2026-11-19): agregar **PoC y mejoras futuras** con servicios de
  cluster y flujo (*cluster flow*), pudiendo **combinar proveedores** GCP/Azure/AWS, para el
  documento final.
- **S4** — Cada elección de servicio lleva la alternativa descartada y el criterio por el que se
  elegiría la otra. Un diagrama sin alternativas no es una decisión de arquitectura.
- **S5** — El scoring tiene que **correr hoy**, sin dependencias nuevas: un stub que nadie puede
  ejecutar no demuestra nada.
- **S6** — El despliegue versiona el **paquete** modelo + contrato + supuestos, no «el modelo».

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S1, S2, S4, S6 | inline | hecho | `docs/arquitectura-despliegue.md`: arquitectura por capas, tabla de monitoreo con umbrales y disparadores, sección «qué NO recomiendo» |
| T2 | S3 | inline | hecho | §2.1 cluster (Databricks/Dataproc/EMR/Synapse) y flow (Composer/MWAA/ADF/Step Functions) con recomendación; §2.2 combinación multi-cloud; §5 PoC de 2 semanas con 6 criterios binarios; §6 siete mejoras futuras ordenadas por valor |
| T3 | S5 | inline | hecho | `scripts/score_batch.py`: 13,196 visitas puntuadas, 1,319 contactos en 209 clientes, CSV escrito. `scripts/serve_api.py`: `/health`, `/score?id=`, `/plan?presupuesto=` respondiendo por `curl` |

## Log

**L1** (petición original, 2026-11-18): «sigue con la fase 5 de la metodologia y problema» →
continuada a Fase 6.

**L2** (ampliación, 2026-11-19): «anade un PoC o mejoras futuras con Cloud Computing asi como
servicios Cluster Flow principalmente puede variar el proveedor puedesn combinar servicios de
cualquier empresa sea GCP/Azure/AWS para anadir al documento Final».

**L3** — Decisión de arquitectura declarada: **el dato pesado se queda en una sola nube (GCP) y lo
que cruza son metadatos y resultados** (identidad en Entra ID, consumo en Power BI, alertas en
Teams). Son kilobytes, así que el egreso es irrelevante. Mover features entre nubes a diario sería
la versión caricaturesca del multi-cloud, y conviene decirlo antes de que alguien lo proponga.

**L4** — Recomendación de cluster y flow con su criterio de reversión: empezar en Dataproc
Serverless y migrar a Databricks **solo** cuando haya más de un consumidor del feature store;
Airflow gestionado por el *backfill* histórico, que es lo que este proyecto va a necesitar cuando
cambie una definición de feature.

**L5** — Defecto propio corregido durante la verificación del endpoint: `mdl.shap_values` devuelve
las filas **en el orden de su propia muestra**, no de `X`. Indexar con `X.index` entregaba la
explicación de otra visita. Se detectó porque un cliente con 92% de tasa de compra mostraba
contribuciones negativas de su historial. Corregido usando el índice de la muestra devuelta; ahora
ese cliente muestra `n_compras_previas` +4.11 y `tasa_compra_historica` +1.96.

**L6** — El criterio de aceptación del PoC que de verdad importa no es el PR-AUC: es que **cortar
una fuente produzca alerta y no-publicación**, en lugar de una lista de contacto silenciosamente
mala. Esa es la diferencia entre un PoC y una demo.
