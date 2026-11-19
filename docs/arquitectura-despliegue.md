# Arquitectura y despliegue · Fase 6

**Prueba Técnica Data Scientist · Ferreycorp** — Modelo de Propensión de Compra

Este documento responde una pregunta concreta: **¿qué hace falta para que lo que corre en seis
notebooks corra todas las noches sobre la cartera real, y para que alguien se entere cuando deje
de funcionar?**

La nube aquí es una decisión, no un adorno. Cada servicio que aparece tiene al lado la alternativa
que se descartó y el criterio por el que se elegiría la otra. Y la arquitectura es
deliberadamente **multi-cloud por capas**: no porque esté de moda, sino porque los tres
proveedores no son igual de buenos en las mismas cosas y el costo de mover datos entre capas es
distinto del costo de mover una capa entera de proveedor.

---

## 1. El punto de partida: qué hay que desplegar

Lo que este proyecto produce no es un modelo: son **cuatro contratos congelados y tres modelos**.

| Artefacto | Qué contiene | Por qué viaja con el modelo |
|---|---|---|
| `artifacts/encuadre.json` | unidad de análisis, targets, features admisibles, split día 550 con huella SHA-256 | si el contrato cambia, las métricas dejan de ser comparables |
| `artifacts/segmentos.json` | `k`, semilla, orden de etiquetas y el mapa de los 500 clientes con huella | *segmento 3* tiene que significar lo mismo en producción |
| `artifacts/modelos.json` | métricas de test de M1/M2/M3, regla del umbral, features descartadas | la cifra de un informe se tiene que poder rastrear a la corrida que la produjo |
| `artifacts/prescriptivo.json` | supuestos de margen, planes de P1/P2/P3, KPIs | el plan y el supuesto bajo el que se calculó son inseparables |

**Consecuencia de diseño:** el despliegue no versiona "el modelo". Versiona el **paquete**
modelo + contrato + supuestos. Un endpoint que sirva M1 sin el encuadre que lo define es un
endpoint que no se puede auditar.

---

## 2. Arquitectura de referencia

```
┌─ INGESTA ─────────────────┐   ┌─ PROCESAMIENTO (cluster) ───┐   ┌─ ENTRENAMIENTO ────────┐
│ CRM / POS / maestro de    │   │ Spark gestionado             │   │ Jobs de ML + MLflow     │
│ precios y promociones     │──▶│ zonas raw · curated · feature│──▶│ registro de experimentos│
│ (batch diario + CDC)      │   │ dbt para las transformaciones│   │ y de modelos (stages)   │
└───────────────────────────┘   └──────────────────────────────┘   └────────────┬────────────┘
             │                               │                                  │
             ▼                               ▼                                  ▼
┌─ LAKE (objeto) ───────────┐   ┌─ FEATURE STORE ─────────────┐   ┌─ INFERENCIA ───────────┐
│ raw: inmutable, particio- │   │ offline: entrenamiento      │   │ batch nocturno:        │
│ nado por fecha            │   │ online: scoring puntual      │   │ cartera completa       │
│ curated: tipado y validado│   │ MISMA definición en los dos │   │ REST: scoring puntual  │
│ features: versionadas     │   │ (control E10 en producción) │   │ + P1/P2/P3 como job    │
└───────────────────────────┘   └──────────────────────────────┘   └────────────┬───────────┘
             ▲                                                                   │
             │                  ┌─ ORQUESTACIÓN (flow) ───────┐                  ▼
             └──────────────────│ DAG diario: ingesta →        │   ┌─ CONSUMO ──────────────┐
                                │ features → scoring → P1 →    │   │ dashboard comercial    │
┌─ MONITOREO ───────────────┐   │ publicación → validación     │   │ CRM (lista de contacto)│
│ PSI de features vs train  │◀──│ reintentos, SLA, alertas     │   │ API para planeamiento  │
│ drift de la tasa base     │   └──────────────────────────────┘   └────────────────────────┘
│ degradación de PR-AUC     │
│ → dispara reentrenamiento │
└───────────────────────────┘
```

### 2.1 Las dos piezas que sostienen todo: cluster y flow

El resto de la arquitectura es relleno si estas dos no están bien elegidas.

**Cluster — procesamiento distribuido.** Hoy el dataset son 58,693 filas y corre en un portátil en
50 segundos. El cluster no es para este volumen: es para la cartera real de Ferreycorp (decenas de
miles de clientes × años de visitas × catálogo completo), donde el *feature engineering* histórico
—ventanas por cliente, lags, lealtad por marca— es el paso que no cabe en memoria.

| Opción | Cuándo es la correcta | Cuándo no |
|---|---|---|
| **Databricks** (sobre AWS, Azure o GCP) | hay varios equipos, se necesita Delta Lake, *time travel* y un catálogo gobernado; se quiere el mismo entorno en los tres proveedores | el costo por DBU no se justifica para un solo pipeline diario |
| **GCP Dataproc Serverless** | se quiere Spark sin administrar cluster y pagar por job; buena integración con BigQuery | se necesita un runtime muy personalizado o librerías nativas pesadas |
| **AWS EMR Serverless** | la casa ya es AWS y el equipo conoce el ecosistema | se quiere notebooks colaborativos de primera clase |
| **Azure Synapse Spark** | la organización corre sobre Microsoft 365 / Fabric y el gobierno pasa por Purview | se busca el Spark más reciente: Synapse suele ir detrás |

**Recomendación:** empezar en **Dataproc Serverless** (o EMR Serverless si la casa es AWS) y
migrar a Databricks **solo** cuando haya más de un consumidor del feature store. Pagar por un
catálogo gobernado antes de tener dos equipos que lo compartan es pagar por un problema que
todavía no existe.

**Flow — orquestación.** El DAG diario es donde se nota la diferencia entre un modelo y un
sistema: ingesta → validación → features → scoring → optimización P1 → publicación al CRM →
validación de salida, con reintentos, SLA y alertas por paso.

| Opción | Cuándo es la correcta | Cuándo no |
|---|---|---|
| **Cloud Composer / MWAA** (Airflow gestionado) | el equipo ya escribe Python; se quiere el ecosistema de operadores más grande y dependencias complejas entre tareas | para 5 tareas lineales es infraestructura de más |
| **Azure Data Factory** | la integración con fuentes Microsoft (Dynamics, SQL Server) es el 80% del trabajo | la lógica de orquestación es sobre todo código Python |
| **AWS Step Functions** | se quiere orquestación serverless, barata, con reintentos declarativos | se necesita *backfill* histórico cómodo, donde Airflow gana |
| **Prefect / Dagster** | se valora el *data lineage* y testear el pipeline localmente | la organización exige un servicio gestionado del proveedor |

**Recomendación:** **Airflow gestionado** (Composer en GCP o MWAA en AWS). La razón es concreta:
este proyecto necesita *backfill* —recalcular features históricas cuando cambie una definición— y
ahí Airflow es claramente superior a Step Functions.

### 2.2 La combinación multi-cloud recomendada

Elegir "lo mejor de cada uno" tiene un costo real: egreso de datos, dos consolas, dos modelos de
IAM. Se justifica solo cuando la ganancia es grande y el tráfico entre nubes es **bajo**. Esta
combinación cumple las dos condiciones:

| Capa | Servicio | Por qué este y no el del proveedor principal |
|---|---|---|
| Lake y gobierno | **GCS + BigQuery** (GCP) | BigQuery cobra por consulta y no por cluster encendido: para un pipeline diario con analistas haciendo consultas esporádicas es el más barato por lejos |
| Procesamiento | **Dataproc Serverless** (GCP) | pega al lake sin egreso; el dato pesado no cruza nubes |
| Entrenamiento y registro | **Vertex AI + MLflow** (GCP) | el entrenamiento vive donde viven los datos |
| Orquestación | **Cloud Composer** (GCP) | mismo motivo: el orquestador debe estar al lado de lo que orquesta |
| Identidad y gobierno corporativo | **Entra ID** (Azure) | la organización ya administra identidades en Microsoft; duplicarlas es crear un problema de seguridad, no resolverlo |
| Consumo de negocio | **Power BI** (Azure) | es lo que la fuerza comercial ya usa; migrar a otro visor es un costo de cambio sin beneficio |
| Alertas y mesa de ayuda | **Teams + Azure Monitor** | las alertas tienen que llegar donde la gente ya mira |

**La regla que ordena esta tabla:** el **dato pesado** se queda en una sola nube (GCP) y lo que
cruza es **metadato y resultados** —identidad, alertas, el extracto que alimenta Power BI—. Son
kilobytes, así que el egreso es irrelevante. Mover features entre nubes todos los días sería la
versión caricaturesca del multi-cloud.

**Cuándo NO hacer esto:** si Ferreycorp tiene un acuerdo corporativo con descuento por compromiso
en un solo proveedor, el ahorro de la combinación se evapora. Entonces todo a ese proveedor, con
los equivalentes de la tabla.

### 2.3 Feature store: el control E10 llevado a producción

La Fase 2 demostró que ajustar el escalado sobre train+test cambiaba 48 parámetros, y que el
`descuento_implicito` de cada fila de train se estaba midiendo contra un precio futuro. En un
notebook eso se resuelve con disciplina; en producción se resuelve con **una sola definición de
feature** compartida entre entrenamiento e inferencia.

Sin feature store, el fallo típico es silencioso: el pipeline de entrenamiento calcula
`tasa_compra_historica` excluyendo la visita actual y el de inferencia la incluye. Las métricas
offline siguen viéndose bien y el modelo rinde peor en producción sin que nadie sepa por qué. Es
exactamente la fuga que este proyecto persiguió en tres fases; replicarla en producción sería
tirar ese trabajo.

---

## 3. Inferencia

**Batch nocturno** (el caso principal). El DAG puntúa la cartera completa, resuelve P1 con el
presupuesto del mes y publica la lista de contacto al CRM. Es la frecuencia correcta porque la
decisión que habilita —a quién visita la fuerza de ventas mañana— es diaria, no instantánea.
`scripts/score_batch.py` es esta pieza, ejecutable hoy.

**Endpoint REST** (el caso puntual). Un vendedor consulta un cliente concreto: el endpoint
devuelve `π`, la mejor marca, el margen esperado y la decisión de contacto, más los **tres
factores SHAP** que la explican. `scripts/serve_api.py` lo implementa con la librería estándar,
sin dependencias nuevas, para que se pueda probar con un `curl` en la misma máquina.

**La explicación viaja con la predicción.** Un endpoint que devuelve 0.73 sin decir por qué
produce una llamada telefónica al equipo de datos cada vez que un vendedor no está de acuerdo.

---

## 4. Monitoreo: qué se mira y qué dispara una acción

Un modelo sin monitoreo es un modelo que ya se degradó y nadie lo sabe.

| Señal | Línea base de este proyecto | Umbral de alerta | Acción |
|---|---|---|---|
| **PSI de features** | distribución de train (días 1–550) | PSI > 0.2 en cualquier feature del top 10 de SHAP | revisar la fuente; si es real, reentrenar |
| **Drift de la tasa base** | 0.2498 en train, 0.2480 en test | desvío > 2 puntos sostenido 2 semanas | reentrenar y revisar el umbral de operación |
| **PR-AUC en producción** | 0.4776 en test | caída > 15% relativa | reentrenar; si no recupera, revisar el encuadre |
| **Features fuera de rango** | 16 de 60 ya tenían valores de test fuera del rango de train | nueva feature fuera de rango | investigar antes de confiar en el score |
| **Cobertura del segmento** | 500 clientes asignados | cliente nuevo sin segmento | asignar por centroide; si crece el volumen, reajustar M0 |
| **Frescura del dato** | — | ingesta con más de 36 h de retraso | no publicar lista de contacto con dato viejo |

**El reentrenamiento se dispara por calendario (mensual) o por degradación detectada, nunca a
mano.** Y una regla que viene directa de la Fase 1: el reentrenamiento **no** reajusta el split ni
las features admisibles. Si eso cambia, se reescribe el contrato y se dice.

---

## 5. PoC: dos semanas, alcance mínimo, criterios de aceptación

El objetivo del PoC **no** es mostrar que el modelo predice —eso ya está medido— sino que el
**flujo completo** corre sin intervención humana y que alguien se entera si se rompe.

### Alcance

| Día | Entregable | Servicios |
|---|---|---|
| 1–2 | Lake con las tres zonas y el histórico cargado; DVC apuntando a GCS | GCS, BigQuery |
| 3–5 | Job de features en Spark que reproduce **exactamente** las 59 features del notebook | Dataproc Serverless |
| 6–7 | Entrenamiento de M1/M2/M3 como job, con los artefactos congelados publicados en el registro | Vertex AI + MLflow |
| 8–9 | DAG diario completo: ingesta → features → scoring → P1 → publicación | Cloud Composer |
| 10–11 | Endpoint REST con explicación SHAP | Cloud Run |
| 12–13 | Monitoreo: PSI, drift, alerta a Teams; dashboard conectado al extracto | Azure Monitor, Power BI |
| 14 | Prueba de fuego: cortar una fuente a propósito y verificar que el DAG falla, alerta y no publica nada | — |

### Criterios de aceptación (binarios, no opinables)

1. Las features que produce el job de Spark coinciden con las del notebook **fila a fila**:
   `verify_reference_fit` y `verify_no_leakage` pasan sobre la salida del cluster.
2. `framing.verify_frozen` y `segmentation.verify_frozen` pasan contra los artefactos publicados
   en el registro.
3. El DAG corre 7 días seguidos sin intervención y publica la lista de contacto antes de las 6 am.
4. El PR-AUC del scoring batch sobre dato nuevo está dentro del 15% de 0.4776.
5. Cortar una fuente produce **alerta y no-publicación**, no una lista de contacto silenciosamente
   mala. Este es el criterio que de verdad separa un PoC de una demo.
6. El endpoint responde p95 < 300 ms y devuelve los tres factores SHAP.

### Costo aproximado del PoC

Dataproc Serverless y Cloud Run por uso (decenas de dólares al mes a este volumen), Composer es el
costo fijo relevante (~300–400 USD/mes en su configuración mínima). **Alternativa consciente:** si
el presupuesto del PoC es ajustado, Cloud Run Jobs + Cloud Scheduler reemplazan a Composer por
céntimos, a cambio de perder *backfill* cómodo. Es la decisión correcta para un PoC y la
equivocada para el estado final.

---

## 6. Mejoras futuras, en orden de valor

El orden no es por dificultad: es por cuánto desbloquean.

1. **Test A/B de precio sobre la marca 3.** Es lo único que resuelve el límite duro de la Fase 4:
   su elasticidad propia salió positiva en las cinco especificaciones porque el precio reacciona a
   la demanda. Sin experimento, P3 no puede tocar esa palanca. Diseño mínimo: dos grupos de
   sucursales, ±5% durante 6 semanas, potencia calculada sobre la varianza observada de su share.
2. **Grupo de control permanente para P1.** Cuatro de los cinco KPIs de la Fase 5 son
   *proyecciones*. Un 10% de la cartera asignado por la política vigente convierte la proyección
   en captura medida — y es lo que permite defender la cifra ante el negocio.
3. **Uplift modelling en lugar de propensión.** M1 responde *¿comprará?*; la pregunta de negocio es
   *¿comprará **porque** lo contactamos?*. Con el grupo de control del punto 2 ya hay dato para
   entrenar un modelo de efecto incremental, que es el objeto que P1 debería maximizar.
4. **Feature store en línea.** Habilita scoring puntual con historial fresco y elimina la
   divergencia entre entrenamiento e inferencia por construcción.
5. **El optimizador como servicio.** Hoy P1/P2/P3 corren en el notebook. Como servicio, Comercial
   puede pedir "plan para un presupuesto de X" y recibir la lista, en lugar de pedirle una corrida
   al equipo de datos.
6. **Reentrenamiento disparado por drift, con *shadow deployment*.** El modelo nuevo puntúa en
   paralelo sin decidir, se comparan dos semanas, y recién entonces se promueve.
7. **Elasticidades jerárquicas por segmento.** La Fase 5 midió un rango estrecho entre segmentos
   con 500 clientes; con la cartera real y un modelo bayesiano jerárquico, el precio diferenciado
   por segmento pasaría de "sin base en el dato" a una hipótesis contrastable.

---

## 7. Qué NO recomiendo, y por qué

Decir qué no hacer es parte del entregable: un plan que solo agrega piezas no es un plan.

- **Un endpoint de inferencia en tiempo real como pieza central.** La decisión es diaria. El
  tiempo real se agrega cuando exista un caso de uso que lo exija, no antes.
- **Un *data mesh* con dominios.** Con un pipeline y un equipo, el gobierno distribuido es
  ceremonia sin beneficio.
- **Migrar todo a Databricks desde el día 1.** Es la respuesta correcta a un problema que
  Ferreycorp tendrá cuando haya varios consumidores del feature store, no hoy.
- **Reentrenar diariamente.** La Fase 1 mostró que la tasa de compra es plana tras los primeros
  90 días: no hay dinámica que justifique reentrenar todos los días, y hacerlo solo agrega varianza
  y costo.
