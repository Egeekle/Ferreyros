# Entregables · dashboard, código, documentación y presentación

**Objetivo:** producir lo que el enunciado califica — código en Word/TXT, documentación PDF y
presentación de **máximo 7 slides** — más el dashboard interactivo pedido por el usuario.

## Specs

- **S1** — Enunciado: «Código de la solución — Word o TXT».
- **S2** — Enunciado: «Documentación: objetivos, arquitectura, metodología, resultados — PDF».
- **S3** — Enunciado: «Presentación, **máx. 7 slides** — PDF, cubriendo: Objetivos · Cronograma ·
  EDA · Metodología · Captura de valor».
- **S4** — Enunciado: «Notebook que ejecute paso a paso y muestre resultados — Ferreycorp probará
  la operatividad; debe correr de principio a fin sin intervención».
- **S5** (pedido del usuario): dashboard **frontend interactivo y dinámico** con los resultados.
- **S6** — Decisión acordada: dashboard **HTML autocontenido en el repo** (se abre con doble clic,
  versiona en git, entra en el ZIP), no una app con servidor.
- **S7** — Una sola fuente de cifras: documento, deck y dashboard salen de `artifacts/` vía
  `dashboard/data.json`. Si discrepan, es un error, no una variante.

## Tasks

| ID | S# | Ruta | Estado | Evidencia |
|---|---|---|---|---|
| T1 | S5, S6, S7 | inline | hecho | `src/ferreyros/dashboard.py` (exportador) + `dashboard/index.html` (29 paneles, 6 pestañas, SVG propio sin CDN) + `data.js` de 103 KB. Verificado con render headless: 0 errores, controles funcionando |
| T2 | S1 | inline | hecho | `scripts/export_codigo.py` → `entregables/codigo_solucion.txt`: 464 KB, 28 archivos, 11,663 líneas |
| T3 | S2 | inline | hecho | `entregables/documentacion.pdf`: **13 páginas** con 8 figuras, generado desde `data.json` |
| T4 | S3 | inline | hecho | `entregables/presentacion.pdf`: **exactamente 7 páginas** (verificado con conteo de `/Type /Page` excluyendo `/Pages`) |
| T5 | S4 | inline | hecho | los 6 notebooks reejecutados de extremo a extremo |

## Log

**L1** (petición, 2026-11-18): «termina todas las fases restantes con los resultados crea un
dashboard tipo frontend interactivo y dinamico».

**L2** — Decisión técnica forzada por el entorno: no hay pandoc, wkhtmltopdf, weasyprint ni LaTeX.
Sí hay **Brave 156**, así que los PDF se imprimen con `--headless --print-to-pdf` desde HTML
propio. Ventaja lateral: el HTML queda en el repo, de modo que quien no pueda correr el navegador
igual puede abrir la página.

**L3** — Detalle que decidió la arquitectura del dashboard: con `file://`, `fetch()` de un JSON
local lo **bloquea CORS**, pero un `<script src>` no. Por eso el exportador escribe `data.js`
además de `data.json`. Sin eso, «se abre con doble clic» habría sido falso.

**L4** — Honestidad de la interactividad: todos los escenarios están **precomputados** y la página
lo dice en un aviso visible. Un HTML estático que insinuara inferencia en vivo estaría vendiendo
algo que no es.

**L5** — Inconsistencia propia detectada al revisar la primera captura: la ganancia de P1 aparecía
como +116.17 en una tarjeta (medida contra ordenar por propensión) y como parte de +1554.06 en el
texto de KPIs (medida contra la política vigente). Dos definiciones distintas en la misma página.
Unificado contra la **política vigente** (+205.32), que es la pregunta de negocio: cuánto vale
adoptar P1 frente a lo que la empresa hace hoy.

**L6** — Las lecturas generadas en Python traen `**negrita**` de markdown; la página las convertía
en texto plano y se veían los asteriscos. Convertidas en el render (`md()`), incluido dentro de las
celdas de tabla.

**L7** — Sin commit: el repositorio arrastra trabajo sin versionar de sesiones anteriores y nadie
pidió commitear.
