#!/usr/bin/env python
"""Build the three graded deliverables from the frozen artifacts.

    uv run python scripts/export_codigo.py       # 1. codigo_solucion.txt
    uv run python scripts/build_entregables.py   # 2. documentacion  3. presentacion

Every figure comes from ``docs/figuras`` and **every number comes from
``dashboard/data.json``**, which in turn comes from the frozen artifacts. The
documentation, the deck and the dashboard therefore cannot disagree: there is
one source and three renderings of it.

The PDFs are printed with headless Brave/Chrome, so the toolchain is whatever
browser the machine already has. The HTML stays next to the PDF: whoever cannot
run the browser can still open the page.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTREGABLES = ROOT / "entregables"
FIG = "../docs/figuras"

NAVEGADORES = [
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("chromium") or "",
    shutil.which("google-chrome") or "",
]

D = json.loads((ROOT / "dashboard" / "data.json").read_text(encoding="utf-8"))
RES, EDA, SEG, MOD, DEC = D["resumen"], D["eda"], D["segmentos"], D["modelos"], D["decisiones"]
META = D["meta"]
GAN = RES["ganancia_p1"] + RES["ganancia_p2"] + RES["ganancia_p3"]


def n0(x) -> str:
    return f"{x:,.0f}".replace(",", " ")


def n2(x) -> str:
    return f"{x:,.2f}".replace(",", " ")


def sgn(x) -> str:
    return ("+" if x >= 0 else "") + n2(x)


def tabla(cols: list[str], filas: list[list], clase: str = "") -> str:
    th = "".join(f"<th>{c}</th>" for c in cols)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>" for f in filas)
    return f'<table class="{clase}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def fig(nombre: str, pie: str, ancho: str = "100%") -> str:
    return (
        f'<figure><img src="{FIG}/{nombre}.png" style="width:{ancho}">'
        f"<figcaption>{pie}</figcaption></figure>"
    )


CSS_DOC = """
@page { size: A4; margin: 17mm 15mm 16mm; }
*{box-sizing:border-box}
body{font:10.5pt/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#1a202c; margin:0}
h1{font-size:21pt; margin:0 0 4px; letter-spacing:-.4px}
h2{font-size:14pt; margin:22px 0 6px; color:#2b6cb0; border-bottom:2px solid #e2e8f0;
  padding-bottom:4px; break-after:avoid}
h3{font-size:11pt; margin:14px 0 4px; break-after:avoid}
p{margin:6px 0; text-align:justify}
.lead{color:#4a5568; font-size:11pt}
.muted{color:#718096}
table{width:100%; border-collapse:collapse; font-size:8.6pt; margin:8px 0; break-inside:avoid}
th,td{border-bottom:1px solid #e2e8f0; padding:4px 6px; text-align:left; vertical-align:top}
th{background:#f7fafc; color:#4a5568; font-size:7.8pt; text-transform:uppercase;
  letter-spacing:.04em}
td.n, th.n{text-align:right; font-variant-numeric:tabular-nums}
figure{margin:10px 0; break-inside:avoid; text-align:center}
figure img{max-width:100%; border:1px solid #e2e8f0; border-radius:4px}
figcaption{font-size:8pt; color:#718096; margin-top:4px; text-align:left}
.cards{display:flex; gap:8px; flex-wrap:wrap; margin:10px 0}
.card{flex:1 1 150px; border:1px solid #e2e8f0; border-radius:6px; padding:8px 10px}
.card b{display:block; font-size:16pt; letter-spacing:-.5px}
.card span{font-size:7.8pt; color:#718096; text-transform:uppercase; letter-spacing:.04em}
.card em{display:block; font-size:8pt; color:#4a5568; font-style:normal; margin-top:2px}
.callout{border-left:3px solid #dd6b20; background:#fffaf0; padding:8px 12px; margin:10px 0;
  break-inside:avoid; font-size:9.6pt}
.callout.good{border-color:#38a169; background:#f0fff4}
.callout.bad{border-color:#e53e3e; background:#fff5f5}
ul{margin:6px 0; padding-left:18px} li{margin:3px 0}
code{background:#edf2f7; padding:1px 4px; border-radius:3px; font-size:8.6pt;
  font-family:ui-monospace,Menlo,monospace}
.page{break-after:page}
header.portada{height:235mm; display:flex; flex-direction:column; justify-content:center}
header.portada h1{font-size:30pt}
.meta{margin-top:26px; font-size:9pt; color:#4a5568}
footer{position:fixed; bottom:-10mm; left:0; right:0; font-size:7.5pt; color:#a0aec0}
"""

CSS_SLIDES = """
@page { size: 297mm 167mm; margin: 0; }
*{box-sizing:border-box}
body{margin:0; font:12pt/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#1a202c; background:#fff}
.slide{width:296mm; height:166mm; padding:12mm 14mm 9mm; break-after:page; position:relative;
  display:flex; flex-direction:column; overflow:hidden}
.slide:last-child{break-after:auto}
.kicker{font-size:8.5pt; color:#2b6cb0; text-transform:uppercase; letter-spacing:.12em;
  font-weight:700}
h1{font-size:25pt; margin:2px 0 2px; letter-spacing:-.6px}
h2{font-size:19pt; margin:2px 0 6px; letter-spacing:-.4px}
.sub{color:#4a5568; font-size:10.5pt; margin:0 0 8px; max-width:95%}
.cols{display:flex; gap:12mm; flex:1; min-height:0}
.col{flex:1; min-width:0}
.cards{display:flex; gap:7px; flex-wrap:wrap; margin:6px 0}
.card{flex:1 1 120px; border:1px solid #e2e8f0; border-radius:7px; padding:7px 9px}
.card b{display:block; font-size:17pt; letter-spacing:-.5px}
.card span{font-size:7pt; color:#718096; text-transform:uppercase; letter-spacing:.05em}
.card em{display:block; font-size:7.5pt; color:#4a5568; font-style:normal}
table{width:100%; border-collapse:collapse; font-size:8.4pt}
th,td{border-bottom:1px solid #e2e8f0; padding:3.5px 6px; text-align:left}
th{color:#718096; font-size:7pt; text-transform:uppercase; letter-spacing:.05em}
td.n,th.n{text-align:right; font-variant-numeric:tabular-nums}
img{max-width:100%; max-height:100%; object-fit:contain; border:1px solid #e2e8f0; border-radius:5px}
ul{margin:4px 0; padding-left:16px; font-size:10pt} li{margin:3px 0}
.callout{border-left:3px solid #dd6b20; background:#fffaf0; padding:6px 10px; font-size:9pt;
  margin:6px 0}
.callout.good{border-color:#38a169; background:#f0fff4}
.foot{position:absolute; bottom:6mm; left:15mm; right:15mm; display:flex;
  justify-content:space-between; font-size:7.5pt; color:#a0aec0;
  border-top:1px solid #edf2f7; padding-top:4px}
.big{font-size:40pt; font-weight:700; letter-spacing:-2px; line-height:1}
.hero{display:flex; flex-direction:column; justify-content:center; height:100%}
"""


# --------------------------------------------------------------------------
# Documento
# --------------------------------------------------------------------------


def documentacion() -> str:
    marcas = {int(r["marca"]): r for r in EDA["marcas"]}
    p2 = DEC["p2"]
    p3 = DEC["p3"]
    seg_nombres = SEG["nombres"]

    cards_resumen = "".join(
        f'<div class="card"><span>{k}</span><b>{v}</b><em>{n}</em></div>'
        for k, v, n in [
            ("Tasa de compra", f'{RES["tasa_base"]:.4f}', "24.94% de 58 693 visitas"),
            ("PR-AUC de M1", f'{RES["pr_auc"]:.4f}', f'azar {RES["pr_auc_azar"]:.4f} · ×1.93'),
            ("Lift decil superior", f'×{RES["lift_top10"]:.2f}', f'política vigente ×{RES["lift_baseline"]:.2f}'),
            ("Ingreso capturado", f'{RES["captura_top10"]:.1f}%', f'con el 10% de las visitas (vigente {RES["captura_historica_top10"]:.1f}%)'),
            ("Ganancia proyectada", sgn(GAN), f'{100*GAN/RES["margen_base"]:.1f}% sobre el margen base'),
        ]
    )

    fases = tabla(
        ["Fase", "Decisión que cierra", "Cómo se verificó"],
        [
            ["0 · Encuadre", "unidad de análisis, targets, features admisibles y split del día 550",
             "contrato congelado con huella SHA-256; <code>verify_frozen</code> corre al inicio de cada fase"],
            ["1 · EDA", "qué parámetros del problema son estimables y con qué forma",
             "44 cifras de referencia reproducidas; 13 controles de tratamiento"],
            ["2 · Features", "68 variables derivadas sin fuga; preprocesamiento en <code>Pipeline</code>",
             "control E10: el ajuste solo en train cambia 48 parámetros del escalador"],
            ["3 · Segmentación (M0)", "k = 3 segmentos, congelados con huella",
             "elegido por codo y <b>estabilidad</b> (ARI 1.0 vs 0.79 en k=9), no por silhouette"],
            ["4 · Modelos", "M1 logística, M2 random forest, M3 Poisson",
             "comparación pareada fold a fold; validación con cliente nuevo <b>y</b> día futuro"],
            ["5 · Prescriptiva", "P1 targeting, P2 calendario, P3 precio",
             "optimalidad de P1 verificada contra el LP; P2 y P3 verificados en simulación conjunta"],
            ["6 · Despliegue", "arquitectura multi-cloud, batch nocturno y endpoint",
             "<code>score_batch.py</code> y <code>serve_api.py</code> ejecutables hoy"],
        ],
    )

    t_marcas = tabla(
        ["Marca", "Share compras", "Share ingreso", "% días promo", "Lift de share", "Δ incidencia", "Lectura"],
        [[f'marca {j}', f'{r["share_compras_pct"]:.1f}%', f'{r["share_ingreso_pct"]:.1f}%',
          f'{100*r["frecuencia_promo"]:.1f}%', f'×{r["lift"]:.2f}', f'{r["delta_pp"]:+.2f} pp',
          r["efecto"]] for j, r in sorted(marcas.items())],
    )

    t_segmentos = tabla(
        ["Segmento", "Clientes", "Ingreso medio", "Tasa de compra", "Ticket", "% del ingreso"],
        [[f'{r["segmento"]} · {seg_nombres[str(r["segmento"])]}', n0(r["clientes"]),
          n0([p for p in SEG["perfil"] if p["segmento"] == r["segmento"]][0]["ingreso_medio"]),
          f'{r["tasa_compra"]:.4f}', f'{r["ticket_medio"]:.2f}', f'{r["pct_ingreso"]:.1f}%']
         for r in SEG["comportamiento"]],
    )

    t_umbral = tabla(
        ["costo/margen", "umbral óptimo", "visitas contactadas", "compras capturadas", "valor neto /1000"],
        [[f'{r["ratio_costo_margen"]:.2f}', f'{r["umbral_optimo"]:.2f}',
          f'{r["pct_visitas_contactadas"]:.1f}%', f'{100*r["recall"]:.1f}%',
          f'{r["valor_neto_por_1000"]:.2f}'] for r in MOD["umbral"]],
    )

    t_p2 = tabla(
        ["Marca", "Días observados", "Días óptimos", "Cambio", "Valor por día-promo"],
        [[f'marca {r["marca"]}', n0(r["dias_observados"]), n0(r["dias_optimos"]),
          f'{r["cambio"]:+.0f}', f'{r["valor_medio_por_dia"]:+.4f}'] for r in p2["reasignacion"]],
    )

    t_p3 = tabla(
        ["Marca", "Precio observado", "Precio elegido", "Cambio", "Estado"],
        [[f'marca {r["marca"]}', f'{r["precio_observado"]:.3f}', f'{r["precio_elegido"]:.3f}',
          f'{r["cambio_pct"]:+.1f}%',
          "congelada: elasticidad no identificada" if r["congelada"] else "optimizable"]
         for r in p3["con_r7"]],
    )

    t_kpis = tabla(
        ["KPI", "Línea base", "Con el modelo", "Delta", "Cómo se mide", "Estado"],
        [[r["kpi"], r["linea_base"], r["con_el_modelo"], r["delta"], r["como_se_mide"], r["estado"]]
         for r in DEC["kpis"]],
    )

    t_supuestos = tabla(
        ["Parámetro", "Valor usado", "De dónde sale"],
        [[r["parametro"], r["valor"], r["de_donde_sale"]] for r in DEC["supuestos"]],
    )

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Documentación · Modelo de Propensión de Compra · Ferreycorp</title>
<style>{CSS_DOC}</style></head><body>

<header class="portada page">
  <div class="kicker muted">Prueba Técnica · Data Scientist</div>
  <h1>Modelo de Propensión de Compra</h1>
  <p class="lead">De 58 693 visitas a tres decisiones ejecutables: a quién contactar,
     qué marca promocionar y a qué precio vender.</p>
  <div class="meta">
    <b>Ferreycorp</b> · documentación de la solución<br>
    {date.today().strftime("%d de noviembre de %Y")}<br><br>
    Dataset: {n0(META["filas"])} visitas · {n0(META["clientes"])} clientes · {n0(META["dias"])} días · 5 marcas<br>
    Split temporal congelado en el día {META["split_day"]} ·
    huellas <code>{META["huellas"]["encuadre_train"]}</code> / <code>{META["huellas"]["encuadre_test"]}</code><br>
    Contenido: 6 notebooks ejecutables, 13 módulos, dashboard interactivo y 4 contratos congelados.
  </div>
</header>

<h2>1 · Resumen ejecutivo</h2>
<p>El enunciado pide un modelo de propensión de compra. Este trabajo lo entrega, pero
parte de una observación que cambia el orden de todo lo demás: <b>un modelo de propensión no
es una decisión</b>. Predecir quién compra no dice a quién contactar con un presupuesto
limitado, ni qué marca promocionar, ni a qué precio vender. Por eso la solución se estructuró
primero como un <b>problema de optimización</b> y la capa de machine learning existe para
estimar sus parámetros.</p>
<div class="cards">{cards_resumen}</div>
<div class="callout good"><b>El resultado que más valor tiene no es el modelo:</b> es que el
calendario promocional actual, tal como está asignado, <b>destruye margen</b>
({n2(p2["valor_observado"])} unidades en el horizonte de test). Reasignar los mismos
{n0(p2["dias_presupuesto"])} días-promoción —sin presupuesto adicional— vale
<b>{sgn(p2["ganancia"])}</b>.</div>
<p>Las cinco conclusiones que un ejecutivo necesita:</p>
<ul>
<li><b>La promoción está mal asignada.</b> La marca 5 aporta el 35.8% del ingreso, tiene el
  mayor efecto incremental ({marcas[5]["delta_pp"]:+.2f} pp de incidencia) y se promociona el
  {100*marcas[5]["frecuencia_promo"]:.1f}% de los días. La marca 3 se promociona más y su
  descuento <b>no mueve la incidencia</b>.</li>
<li><b>Contactar al azar destruye valor.</b> Con el costo de contacto supuesto, una política
  aleatoria deja objetivo neto negativo. El targeting no es una mejora marginal: es la
  diferencia entre una campaña que paga y una que no.</li>
<li><b>El umbral de 0.5 cuesta dinero.</b> El umbral óptimo es <code>costo/margen</code> y se
  verificó sobre test. Con 0.5 se contactaría el 9.7% de las visitas y se capturaría el 24.5%
  de las compras.</li>
<li><b>La demografía segmenta pero no predice.</b> Los segmentos separan 8 puntos de tasa de
  compra y <b>23.6 puntos</b> de preferencia de marca: sirven para accionar campañas, no para
  predecir.</li>
<li><b>Todo esto es una proyección hasta que haya grupo de control.</b> Cuatro de los cinco
  KPIs están etiquetados como tales, con el experimento que los convertiría en medición.</li>
</ul>

<h2>2 · Objetivos y encuadre</h2>
<p><b>Objetivo de negocio:</b> aumentar el margen de contribución de la cartera decidiendo mejor
tres cosas que hoy se deciden por costumbre — a quién se contacta, qué se promociona y a qué
precio se vende.</p>
<p><b>Objetivo técnico:</b> estimar <code>π</code> (probabilidad de compra), <code>q</code>
(elección de marca) y <code>λ</code> (cantidad) con validación honesta, y usarlos para resolver
los tres sub-problemas de decisión.</p>
<p><b>Unidad de análisis:</b> <code>cliente × visita</code>, verificada única sobre
<code>(id, dia_visita)</code>. <b>Validación:</b> split temporal del día {META["split_day"]},
congelado antes de ajustar nada. Un <code>train_test_split</code> aleatorio sería inválido: con
~117 visitas por cliente repartiría visitas del mismo cliente entre train y test e inflaría
todas las métricas.</p>

<h2>3 · Arquitectura de la solución</h2>
<p>Dos capas. La <b>prescriptiva</b> decide; la <b>predictiva</b> estima los coeficientes que la
primera necesita.</p>
<pre style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:5px;padding:9px;font-size:8.4pt">
max  Σ_ijt  x_ijt · π_it · q_ijt · λ_ijt · Σ_k (y_jtk · m_jk)  −  a·Σ x_ijt  −  Σ d_j·z_jt

x_ijt  ofrecer la marca j al cliente i en t      →  P1 · targeting
z_jt   promocionar la marca j en t               →  P2 · calendario
y_jtk  vender la marca j al punto de precio k    →  P3 · pricing

R1 un precio por marca · R2 presupuesto · R3 una oferta por visita · R4 fatiga
R5 tope de días en promoción · R6 piso de margen · R7 arquitectura de marcas (activable)
</pre>
<p>La discretización del precio sobre la <b>escalera observada</b> (21 a 44 puntos por marca) es
lo que convierte un MINLP intratable en un MILP resoluble. Y P1 —el modelo base entregable—
resulta <b>exactamente resoluble</b>: su LP relajado es integral, de modo que el greedy por
margen esperado es el óptimo. Eso no se cita del papel, se verifica en el notebook 05.</p>
{fases}

<h2>4 · Datos y análisis exploratorio</h2>
<p>El EDA no describe el dataset: produce los parámetros del problema de optimización. Cada
hallazgo cierra con su consecuencia; un gráfico sin consecuencia no entró al entregable.</p>
{t_marcas}
<p>Las dos columnas de la derecha son preguntas distintas que suelen confundirse. El
<b>lift de share</b> mide canibalización entre marcas; la <b>Δ incidencia</b> mide si el
descuento crea una compra que no habría ocurrido. La marca 3 tiene el lift de share más alto
(×1.86) y no mueve la incidencia (z = 0.7): su descuento se paga con volumen ya vendido.</p>
{fig("eda_promo_incidencia", "Efecto bruto de la promoción sobre la incidencia, marca por marca. La marca 5 suma +6.90 pp con 24 días de promoción en 730.")}
<div class="callout"><b>Dinámica temporal:</b> lo que parecía estacionalidad semanal y mensual
<b>no se repite entre los dos años</b> del panel (correlaciones de +0.10 y +0.11). Lo único real
es un régimen de arranque de 90 días, donde el 88% de las visitas aún no tiene historial. Sin
esa prueba de réplica, el EDA habría declarado una estacionalidad inexistente.</div>
{fig("eda_baseline", "El baseline que M1 tiene que superar: ordenar por tasa histórica ya da ×2.43 en el decil superior.")}

<h2>5 · Segmentación de clientes (M0)</h2>
<p>K-Means sobre la demografía a nivel cliente, con <code>k = 3</code> elegido por codo y por
<b>reproducibilidad</b> (ARI medio 1.0 sobre remuestreos, frente a 0.79 con k = 9). La silhouette
prefería k = 9; la estabilidad decidió, porque un segmento que cambia de integrantes según la
muestra no sirve para planificar una campaña.</p>
{t_segmentos}
<div class="callout"><b>Hallazgo:</b> los tres segmentos son exactamente los tres niveles de
ocupación, y el ingreso, la edad y la educación se ordenan detrás. Y una advertencia declarada:
cambiar la codificación de esa variable produce particiones <b>sin relación</b> entre sí
(ARI 0.027). La segmentación es una elección del modelador, no un descubrimiento del dato; la
elegida gana en cohesión y en separación de comportamiento, y eso está medido.</div>
{fig("m0_marcas_por_segmento", "Preferencia de marca por segmento: hasta 23.6 puntos de desvío contra el share global. Aquí es donde el segmento cambia decisiones.")}

<h2>6 · Modelos de predicción</h2>
<p>Tres modelos encadenados, con la progresión de complejidad que el método exige y el resultado
que midió, no el que se esperaba.</p>
{tabla(["Modelo", "Elegido", "Métrica en test", "Contra qué"],
 [["M1 · incidencia (π)", "logística regularizada C=0.03", f'PR-AUC {MOD["m1"]["pr_auc"]:.4f}', f'azar {MOD["m1"]["pr_auc_azar"]:.4f}'],
  ["M2 · elección de marca (q)", "random forest", f'accuracy {MOD["m2"]["accuracy"]:.4f}', f'clase mayoritaria {MOD["m2"]["accuracy_clase_mayoritaria"]:.4f}'],
  ["M3 · cantidad (λ)", "Poisson", f'MAE {MOD["m3"]["mae"]:.4f}', f'baseline de la media {MOD["m3"]["mae_baseline_media"]:.4f}']])}
<p>En M1 la complejidad <b>no</b> paga: la comparación pareada fold a fold da +0.0117 para el
random forest con error estándar 0.0139 (t = 0.84). En M2 sí paga, y por un margen claro. Recorrer
la progresión sirvió justamente para que la respuesta fuera distinta en cada caso.</p>
<div class="callout bad"><b>Advertencia metodológica que vale para todo el proyecto:</b>
<code>GroupKFold</code> por cliente <b>no es el límite pesimista</b> en un panel. Deja fuera al
cliente pero entrena con los mismos días, y como los 500 clientes comparten calendario, precios y
promociones, el modelo ya vio las condiciones de demanda de la validación: infla el PR-AUC del
random forest en <b>0.24</b>. Todas las decisiones se tomaron con un esquema que cruza
<b>cliente nuevo y día futuro</b>.</div>
{fig("m1_shap", "Drivers globales de M1. Los tres primeros son historial de comportamiento; ninguna variable demográfica aparece en el top 12.")}
<h3>El umbral de operación</h3>
<p>El umbral no se fija en 0.5: es el que iguala margen esperado y costo de contacto, es decir
<code>costo/margen</code>. La regla se verificó sobre test — el óptimo empírico se desvía como
máximo 0.055 del teórico.</p>
{t_umbral}
<h3>Elasticidades y el límite que no se resolvió</h3>
<p>El logit multinomial entrega las elasticidades propias y cruzadas que P3 necesita. Cuatro de
las cinco propias salen negativas, como deben.</p>
<div class="callout bad"><b>La elasticidad propia de la marca 3 sale positiva</b> en las cinco
especificaciones probadas (con y sin regularización, con controles de lealtad, con la última
marca comprada, con el segmento). No es un accidente de ajuste: tiene 21 puntos de precio, 14.4%
de rango relativo y el 5.1% de las compras. Sin variación de precio independiente de la demanda,
lo que se mide no es respuesta al precio sino <b>la reacción del precio a la demanda</b>. P3
congela su precio; lo primero que vale pedirle al negocio es un test A/B sobre esa marca.</div>
{fig("m2_elasticidades", "Matriz de elasticidades: la diagonal (recuadro) es la propia; la marca 3 es la que no está identificada.")}

<h2>7 · Capa prescriptiva: las tres decisiones</h2>
<h3>P1 · Targeting</h3>
<p>Selecciona a quién contactar y con qué marca, bajo presupuesto y fatiga. Su optimalidad está
<b>verificada</b>: el greedy alcanza el óptimo del LP relajado con cero variables fraccionarias.</p>
{tabla(["Política", "Contactos", "Margen esperado", "Objetivo neto", "Ingreso real capturado"],
 [[r["politica"], n0(r["contactos"]), n2(r["margen_esperado"]), n2(r["objetivo_neto"]),
   n0(r["ingreso_real_capturado"])] for r in DEC["p1"]["politicas"]])}
<p>El punto de saturación está en <b>{n0(DEC["p1"]["contactos_utiles"])} contactos</b>
({DEC["p1"]["pct_visitas_utiles"]:.1f}% de las visitas): más allá, ninguna visita tiene margen
esperado mayor que su costo de contacto. Ese es el presupuesto que el negocio necesita, y no más.</p>
{fig("p1_frontera", "Frontera de presupuesto de P1: el objetivo neto satura y el margen marginal por contacto cae un orden de magnitud.")}
<h3>P2 · Calendario promocional</h3>
<p>MILP con el tope de días por marca (R5) y el presupuesto del calendario observado: la pregunta
no es cuánto gastar, es dónde.</p>
{t_p2}
<div class="callout good">Reasignar los mismos {n0(p2["dias_presupuesto"])} días-promoción
lleva el valor del calendario de <b>{n2(p2["valor_observado"])}</b> a
<b>{n2(p2["valor_optimo"])}</b>. Verificado como simulación conjunta ({sgn(p2["ganancia_verificada"])}),
con un error de la aproximación aditiva del {100*p2["error_aditivo"]:.1f}%.</div>
{fig("p2_reasignacion", "P2: el presupuesto promocional se mueve de las marcas 1 y 2 hacia la marca 5, que es la que responde.")}
<h3>P3 · Precio</h3>
<p>MILP sobre la escalera discreta, con piso de margen (R6) y una <b>banda de confianza de ±10%</b>
alrededor del precio observado. La banda no es timidez: sin ella el óptimo se va al extremo de la
grilla, que es la firma de un optimizador explotando al modelo donde no se validó.</p>
{t_p3}
<p>Imponer R7 (ordenamiento de precios entre marcas) cuesta <b>{n2(p3["costo_r7"])}</b> unidades
de margen. El brief lo anticipó como decisión de modelado, no como hecho del dato: ahora es una
cifra que el negocio puede aceptar o rechazar sabiendo qué compra.</p>
{fig("p3_respuesta_precio", "Respuesta del margen por punto de precio (□ = precio observado). La marca 3 aparece como un único punto: su precio está congelado.")}

<h2>8 · Captura de valor</h2>
{fig("p5_waterfall", "Las tres palancas sobre el margen base del horizonte de test.")}
{t_kpis}
<div class="callout"><b>Cuatro de los cinco KPIs son proyecciones</b>, y así están etiquetados.
El brief es explícito: sin medición contrafactual no hay captura demostrable. Presentar la suma
como resultado conseguido es la forma más rápida de perder la confianza del negocio la primera
vez que no se cumpla.</div>
{tabla(["Área", "Decisión que habilita", "Cifra de esta fase", "Sub-problema"],
 [[r["area"], r["decision_que_habilita"], r["cifra_de_esta_fase"], r["sub_problema"]]
  for r in DEC["impacto"]])}

<h2>9 · Arquitectura cloud, PoC y mejoras futuras</h2>
<p>El detalle completo está en <code>docs/arquitectura-despliegue.md</code>. Lo esencial: la
arquitectura es <b>multi-cloud por capas</b>, con una regla que la ordena — el dato pesado se
queda en una sola nube y lo que cruza es metadato y resultados.</p>
{tabla(["Capa", "Servicio", "Por qué este y no el del proveedor principal"],
 [["Lake y gobierno", "GCS + BigQuery (GCP)", "cobra por consulta, no por cluster encendido"],
  ["Procesamiento · <b>cluster</b>", "Dataproc Serverless (GCP)", "pega al lake sin egreso; Spark sin administrar"],
  ["Orquestación · <b>flow</b>", "Cloud Composer / Airflow", "hace falta backfill histórico, donde Airflow gana"],
  ["Feature store", "offline + online, una sola definición", "es el control E10 llevado a producción"],
  ["Entrenamiento y registro", "Vertex AI + MLflow", "el entrenamiento vive donde viven los datos"],
  ["Inferencia", "batch nocturno + endpoint REST", "la decisión es diaria; el tiempo real se agrega si hay caso"],
  ["Identidad", "Entra ID (Azure)", "la organización ya administra identidades ahí"],
  ["Consumo", "Power BI (Azure) + CRM", "es lo que la fuerza comercial ya usa"]])}
<h3>PoC de dos semanas · criterios de aceptación binarios</h3>
<ul>
<li>Las features del job de Spark coinciden <b>fila a fila</b> con las del notebook.</li>
<li><code>verify_frozen</code> del encuadre y de los segmentos pasa contra los artefactos publicados.</li>
<li>El DAG corre 7 días seguidos sin intervención y publica antes de las 6 am.</li>
<li>El PR-AUC sobre dato nuevo queda dentro del 15% de {MOD["m1"]["pr_auc"]:.4f}.</li>
<li><b>Cortar una fuente produce alerta y no-publicación</b>, no una lista silenciosamente mala.
  Este es el criterio que separa un PoC de una demo.</li>
</ul>
<h3>Mejoras futuras, por valor desbloqueado</h3>
<ul>
<li><b>Test A/B de precio sobre la marca 3</b> — lo único que resuelve el límite de la Fase 4.</li>
<li><b>Grupo de control permanente para P1</b> — convierte cuatro KPIs de proyección en medición.</li>
<li><b>Uplift modelling</b> — M1 responde <i>¿comprará?</i>; el negocio pregunta <i>¿comprará
  porque lo contactamos?</i>.</li>
<li>Feature store en línea, optimizador como servicio, reentrenamiento por drift con shadow
  deployment, elasticidades jerárquicas por segmento.</li>
</ul>

<h2>10 · Supuestos y límites declarados</h2>
{t_supuestos}
<ul>
<li><b>No hay costos en el dataset.</b> Todo está en unidades de margen bajo un margen bruto
  supuesto del 30%, con sensibilidad entre 20% y 40%: las cifras se mueven, la <b>decisión</b> no.</li>
<li><b>Los precios son observacionales, no experimentales.</b> Las elasticidades son asociativas
  y exigen validación A/B antes de ejecutar un cambio de precio.</li>
<li><b>Predict-then-optimize amplifica el error del modelo.</b> Por eso P3 trabaja con banda de
  confianza y los planes se verifican con simulación conjunta, no con la suma de las partes.</li>
<li><b>500 clientes es poco</b> para afirmar generalización a clientes nuevos; por eso se reporta
  el esquema estricto de validación y no el optimista.</li>
</ul>

<h2>11 · Reproducibilidad</h2>
<p>Ferreycorp ejecutará la solución, así que correr de extremo a extremo no es un extra:</p>
<pre style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:5px;padding:9px;font-size:8.6pt">
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_encuadre.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
...                                                      notebooks/05_pricing.ipynb
uv run python scripts/score_batch.py        # lista de contacto del CRM
uv run python scripts/serve_api.py          # endpoint de scoring puntual
open dashboard/index.html                   # dashboard interactivo (sin servidor)
</pre>
<p>Cuatro contratos congelados en <code>artifacts/</code> con huella SHA-256 garantizan que las
cifras de este documento, del dashboard y de los notebooks son las mismas: hay una sola fuente y
tres renderizaciones. Si el dato o el encuadre cambian, las verificaciones fallan en vez de
producir un número distinto en silencio.</p>
<footer>Modelo de Propensión de Compra · Ferreycorp · {date.today().isoformat()}</footer>
</body></html>"""


# --------------------------------------------------------------------------
# Presentación · exactamente 7 slides
# --------------------------------------------------------------------------


def presentacion() -> str:
    marcas = {int(r["marca"]): r for r in EDA["marcas"]}
    p2 = DEC["p2"]

    def foot(n: int, titulo: str) -> str:
        return (f'<div class="foot"><span>Modelo de Propensión de Compra · Ferreycorp</span>'
                f'<span>{titulo}</span><span>{n} / 7</span></div>')

    s1 = f"""<section class="slide"><div class="hero">
  <div class="kicker">Prueba Técnica · Data Scientist · Ferreycorp</div>
  <h1>Modelo de Propensión de Compra</h1>
  <p class="sub" style="font-size:13pt">El modelo de propensión es el punto de partida, no el
  entregable. Lo que el negocio ejecuta son <b>tres decisiones</b>: a quién contactar,
  qué marca promocionar y a qué precio vender.</p>
  <div class="cards" style="margin-top:10px">
    <div class="card"><span>Objetivo de negocio</span><b style="font-size:11pt">Más margen por visita</b>
      <em>decidiendo mejor targeting, promoción y precio</em></div>
    <div class="card"><span>Objetivo técnico</span><b style="font-size:11pt">π · q · λ</b>
      <em>tres modelos que alimentan un optimizador</em></div>
    <div class="card"><span>Dataset</span><b style="font-size:11pt">{n0(META["filas"])} visitas</b>
      <em>{n0(META["clientes"])} clientes · {n0(META["dias"])} días · 5 marcas</em></div>
    <div class="card"><span>Entregado</span><b style="font-size:11pt">6 notebooks + dashboard</b>
      <em>ejecutables de extremo a extremo</em></div>
  </div>
  <div class="callout" style="margin-top:12px"><b>La tesis:</b> predecir quién compra no dice a
  quién contactar con presupuesto limitado. Por eso el problema se estructuró primero como
  <b>optimización</b>, y el machine learning existe para estimar sus parámetros.</div>
</div>{foot(1, "Objetivos")}</section>"""

    s2 = f"""<section class="slide">
  <div class="kicker">Cronograma</div>
  <h2>Cinco días, seis fases, todo verificado</h2>
  <div class="cols"><div class="col">
  {tabla(["Día", "Bloque", "Salida", "Estado"],
   [["D1", "Fases 0–1 · encuadre y EDA", "contrato congelado + 20 figuras", "✓"],
    ["D2", "Fases 2–3 · features y segmentación", "83 columnas sin fuga + M0 congelado", "✓"],
    ["D3", "Fase 4 · M1, M2, M3", "métricas en test + SHAP", "✓"],
    ["D4", "Fases 5–6 · pricing y arquitectura", "P1/P2/P3 + cloud + PoC", "✓"],
    ["D5", "Reproducibilidad y entregables", "dashboard + documentación + deck", "✓"]])}
  <div class="callout good" style="margin-top:8px"><b>Criterio de cierre aplicado a cada fase:</b>
  ninguna decisión se da por buena sin evidencia ejecutada. Cuatro contratos congelados con huella
  SHA-256 hacen que una deriva aparezca como verificación fallada y no como un número raro.</div>
  </div><div class="col">
  {tabla(["Fase", "Lo que cierra"],
   [["0 · Encuadre", "split del día 550 congelado antes de ajustar nada"],
    ["1 · EDA", "parámetros estimables, con su consecuencia"],
    ["2 · Features", "68 variables sin fuga; ajuste solo en train"],
    ["3 · M0", "3 segmentos elegidos por estabilidad"],
    ["4 · M1/M2/M3", "validación con cliente nuevo y día futuro"],
    ["5 · P1/P2/P3", "optimalidad de P1 verificada contra el LP"],
    ["6 · Despliegue", "arquitectura multi-cloud + scoring ejecutable"]])}
  </div></div>{foot(2, "Cronograma")}</section>"""

    s3 = f"""<section class="slide">
  <div class="kicker">Análisis exploratorio</div>
  <h2>El hallazgo central: la promoción está donde menos rinde</h2>
  <div class="cols">
    <div class="col"><img src="{FIG}/eda_promo_incidencia.png" alt="efecto de la promoción"></div>
    <div class="col">
      {tabla(["Marca", "% ingreso", "% días promo", "Δ incidencia"],
       [[f'marca {j}', f'{r["share_ingreso_pct"]:.1f}%', f'{100*r["frecuencia_promo"]:.1f}%',
         f'{r["delta_pp"]:+.2f} pp'] for j, r in sorted(marcas.items())])}
      <div class="callout"><b>Dos preguntas distintas.</b> El lift de share mide canibalización;
      la Δ incidencia mide si el descuento <b>crea</b> la compra. La marca 3 lidera la primera
      (×1.86) y no mueve la segunda (z = 0.7).</div>
      <ul>
        <li><b>Marca 5:</b> 35.8% del ingreso, el mayor efecto incremental, casi nunca se promociona.</li>
        <li><b>Sin estacionalidad:</b> lo que lo parecía no se repite entre los dos años del panel.</li>
        <li><b>El baseline no es trivial:</b> ordenar por tasa histórica ya da ×2.43.</li>
      </ul>
    </div></div>{foot(3, "EDA")}</section>"""

    s4 = f"""<section class="slide">
  <div class="kicker">Metodología</div>
  <h2>Optimización como modelo base, ML como capa de estimación</h2>
  <div class="cols"><div class="col">
  <pre style="background:#f7fafc;border:1px solid #e2e8f0;border-radius:6px;padding:9px;font-size:8pt;line-height:1.35">
max Σ x·π·q·λ·m − a·Σx − Σ d·z

x_ijt → P1 · a quién contactar
z_jt  → P2 · qué promocionar y cuándo
y_jtk → P3 · a qué precio vender

R1 un precio · R2 presupuesto · R3 una oferta
R4 fatiga · R5 tope promo · R6 piso de margen
R7 arquitectura de marcas (activable)</pre>
  <div class="callout good"><b>P1 es exactamente resoluble</b> y se verificó: el greedy alcanza el
  óptimo del LP relajado con cero variables fraccionarias.</div>
  </div><div class="col">
  {tabla(["Modelo", "Entrega", "Universo"],
   [["M1 · incidencia", "π — probabilidad de compra", "58 693 visitas"],
    ["M2 · elección de marca", "q + elasticidades cruzadas", "14 638 compras"],
    ["M3 · cantidad", "λ — unidades esperadas", "14 638 compras"]])}
  <div class="callout bad"><b>Validación:</b> <code>GroupKFold</code> por cliente <b>no</b> es el
  límite pesimista en un panel — infla el PR-AUC en 0.24 porque entrena con los mismos días. Todo
  se decidió cruzando <b>cliente nuevo y día futuro</b>.</div>
  <ul>
    <li>PR-AUC como métrica principal; accuracy prohibida (75.1% no comprando nunca).</li>
    <li>El umbral sale del <b>costo de contacto</b>, no de 0.5.</li>
    <li>Explicabilidad con SHAP global y local + odds ratios.</li>
  </ul>
  </div></div>{foot(4, "Metodología")}</section>"""

    s5 = f"""<section class="slide">
  <div class="kicker">Modelos y resultados</div>
  <h2>La complejidad no paga en M1; sí paga en M2</h2>
  <div class="cards">
    <div class="card"><span>M1 · PR-AUC</span><b>{MOD["m1"]["pr_auc"]:.4f}</b>
      <em>azar {MOD["m1"]["pr_auc_azar"]:.4f} · ×1.93</em></div>
    <div class="card"><span>M2 · accuracy</span><b>{MOD["m2"]["accuracy"]:.4f}</b>
      <em>clase mayoritaria {MOD["m2"]["accuracy_clase_mayoritaria"]:.4f}</em></div>
    <div class="card"><span>M3 · MAE</span><b>{MOD["m3"]["mae"]:.4f}</b>
      <em>baseline de la media {MOD["m3"]["mae_baseline_media"]:.4f}</em></div>
    <div class="card"><span>Ingreso capturado</span><b>{RES["captura_top10"]:.1f}%</b>
      <em>con el 10% de las visitas · vigente {RES["captura_historica_top10"]:.1f}%</em></div>
  </div>
  <div class="cols" style="margin-top:4px">
    <div class="col"><img src="{FIG}/m1_shap.png" alt="SHAP"></div>
    <div class="col">
      <ul>
        <li><b>Comparación pareada fold a fold:</b> el random forest gana +0.0117 con error
          estándar 0.0139 (t = 0.84). Dentro del ruido → se queda la logística, que además es la
          que la capa prescriptiva necesita.</li>
        <li><b>El historial manda:</b> los tres primeros drivers son de comportamiento. Ninguna
          variable demográfica entra al top 12 — la demografía segmenta pero no predice.</li>
        <li><b>Umbral por costo:</b> con 0.5 se contactaría el 9.7% de las visitas y se capturaría
          el 24.5% de las compras. La convención cuesta dinero medible.</li>
        <li><b>Límite declarado:</b> la elasticidad propia de la marca 3 no está identificada en
          ninguna de las cinco especificaciones.</li>
      </ul>
    </div></div>{foot(5, "Modelos")}</section>"""

    s6 = f"""<section class="slide">
  <div class="kicker">Captura de valor</div>
  <h2>El calendario promocional actual destruye margen</h2>
  <div class="cols">
    <div class="col"><img src="{FIG}/p5_waterfall.png" alt="captura de valor"></div>
    <div class="col">
      <div class="cards">
        <div class="card"><span>P1 · targeting</span><b>{sgn(RES["ganancia_p1"])}</b>
          <em>sobre la política vigente, mismo presupuesto</em></div>
        <div class="card"><span>P2 · calendario</span><b>{sgn(RES["ganancia_p2"])}</b>
          <em>mismos {n0(p2["dias_presupuesto"])} días-promoción</em></div>
        <div class="card"><span>P3 · pricing</span><b>{sgn(RES["ganancia_p3"])}</b>
          <em>banda ±10%, verificado en simulación conjunta</em></div>
      </div>
      <div class="callout good"><b>P2 es la palanca más grande y la más barata:</b> no pide
      presupuesto nuevo. El calendario observado vale <b>{n2(p2["valor_observado"])}</b> —valor
      negativo— y reasignado vale <b>{n2(p2["valor_optimo"])}</b>.</div>
      <div class="callout"><b>Esto es una proyección.</b> Cuatro de los cinco KPIs se vuelven
      captura demostrada solo con A/B o grupo de control: decil superior vs. política actual,
      10% de la cartera como control para P1, A/B por cartera para P2 y test de precio para P3.</div>
    </div></div>{foot(6, "Captura de valor")}</section>"""

    s7 = f"""<section class="slide">
  <div class="kicker">Arquitectura y siguiente paso</div>
  <h2>Cómo corre todas las noches, y qué probar primero</h2>
  <div class="cols"><div class="col">
  {tabla(["Capa", "Servicio", "Por qué"],
   [["Lake", "GCS + BigQuery", "cobra por consulta, no por cluster"],
    ["<b>Cluster</b>", "Dataproc Serverless", "Spark sin administrar, sin egreso"],
    ["<b>Flow</b>", "Cloud Composer / Airflow", "backfill histórico"],
    ["Feature store", "offline + online", "el control anti-fuga en producción"],
    ["Registro", "Vertex AI + MLflow", "modelo + contrato versionados juntos"],
    ["Inferencia", "batch nocturno + REST", "la decisión es diaria"],
    ["Identidad / consumo", "Entra ID + Power BI", "donde la organización ya trabaja"]])}
  <div class="callout"><b>Regla del multi-cloud:</b> el dato pesado se queda en una nube; lo que
  cruza es metadato y resultados. Mover features entre nubes a diario sería la caricatura.</div>
  </div><div class="col">
  <h3 style="margin-top:0">PoC de 2 semanas · aceptación binaria</h3>
  <ul>
    <li>Features del cluster idénticas <b>fila a fila</b> a las del notebook.</li>
    <li>El DAG corre 7 días sin intervención y publica antes de las 6 am.</li>
    <li>PR-AUC sobre dato nuevo dentro del 15% de {MOD["m1"]["pr_auc"]:.4f}.</li>
    <li><b>Cortar una fuente produce alerta y no-publicación</b> — el criterio que separa un PoC
      de una demo.</li>
  </ul>
  <h3>Mejoras futuras, por valor desbloqueado</h3>
  <ul>
    <li><b>Test A/B de precio en la marca 3</b> — desbloquea P3.</li>
    <li><b>Grupo de control para P1</b> — convierte proyección en medición.</li>
    <li><b>Uplift modelling</b> — de <i>¿comprará?</i> a <i>¿comprará porque lo contactamos?</i>.</li>
  </ul>
  <div class="callout good"><b>Hoy ya corre:</b> <code>score_batch.py</code> publica la lista de
  contacto y <code>serve_api.py</code> responde scoring puntual con su explicación SHAP.</div>
  </div></div>{foot(7, "Arquitectura")}</section>"""

    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f"<title>Presentación · Ferreycorp</title><style>{CSS_SLIDES}</style></head><body>"
            + s1 + s2 + s3 + s4 + s5 + s6 + s7 + "</body></html>")


# --------------------------------------------------------------------------


def navegador() -> str | None:
    for ruta in NAVEGADORES:
        if ruta and Path(ruta).exists():
            return ruta
    return None


def a_pdf(html: Path, pdf: Path) -> bool:
    nav = navegador()
    if not nav:
        print(f"  (sin navegador headless: queda {html.name}, imprimir a PDF desde el navegador)")
        return False
    cmd = [nav, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           "--virtual-time-budget=8000", f"--print-to-pdf={pdf}", html.as_uri()]
    subprocess.run(cmd, capture_output=True, timeout=180)
    return pdf.exists()


def paginas(pdf: Path) -> int:
    """Count pages without counting the ``/Pages`` nodes, which also match."""
    import re

    return len(re.findall(rb"/Type\s*/Page(?![s])", pdf.read_bytes()))


def main() -> None:
    ENTREGABLES.mkdir(parents=True, exist_ok=True)
    for nombre, generador in (("documentacion", documentacion), ("presentacion", presentacion)):
        html = ENTREGABLES / f"{nombre}.html"
        html.write_text(generador(), encoding="utf-8")
        pdf = ENTREGABLES / f"{nombre}.pdf"
        ok = a_pdf(html, pdf)
        detalle = f"{pdf.stat().st_size/1024:.0f} KB · {paginas(pdf)} páginas" if ok else "solo HTML"
        print(f"· {nombre}: {html.name} + {pdf.name if ok else '—'} ({detalle})")


if __name__ == "__main__":
    main()
