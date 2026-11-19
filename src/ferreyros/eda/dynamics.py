"""EDA -- temporal dynamics of demand.

Fase 1 of ``docs/brief-metodologia.md`` asks for "Dinamica temporal:
estacionalidad semanal/mensual, frecuencia de visita, tendencia de la tasa de
compra". Visit frequency already lives in ``constraints.fatigue_bounds`` and
``constraints.visit_interval_profile``; this module covers seasonality, trend,
and the start-of-panel regime that turned out to explain both.

Two caveats are structural and are declared in every function.

The dataset has no calendar dates. ``dia_visita`` is an index from 1 to 730, so
"week" here is the 7-day cycle of that index and "month" a 30-day block. Both
are **proxies**: a cycle effect is read as *there is a cycle*, never as
*Fridays sell more*.

A deviation that beats its standard error is not yet seasonality. Folding 730
days onto 7 or 12 positions and testing each one invites false positives, so
every seasonality verdict here requires the pattern to **replicate across the
two years** of the panel. That test is what turns the apparent weekly and
monthly cycles of this dataset into what they really are: an artifact of the
first 90 days.

Why it matters beyond the report: the features ``semana`` and ``mes`` built in
Fase 2 are trend indices whose test values lie outside the train range (98% and
88% of test rows). Whether that extrapolation is a risk worth taking depends on
whether there is any temporal signal to extrapolate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg

#: Length of the proxy week and month over ``dia_visita``.
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30

#: Days per year of the panel, used to split it in two for the replication test.
DAYS_PER_YEAR = 360

#: First block considered already stable when computing the reference rate.
#: Blocks 1-3 are the warm-up this module measures; see :func:`warmup_profile`.
FIRST_STABLE_BLOCK = 4


def _binomial_se(rate: float, n: pd.Series | float) -> pd.Series | float:
    """Standard error of a proportion, used as the noise floor of a cell."""
    return np.sqrt(rate * (1.0 - rate) / n)


def _cycle_profile(df: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Rate per cycle position, with the two years side by side.

    The correlation between the two yearly profiles is the replication test:
    a real cycle repeats, an artifact does not.
    """
    d = df.copy()
    d["posicion"] = pos
    d["anio_panel"] = np.where(d[cfg.KEY_TIME] <= DAYS_PER_YEAR, 1, 2)

    rate = float(df[cfg.TARGET_INCIDENCE].mean())
    out = (
        d.groupby("posicion", observed=True)
        .agg(
            visitas=(cfg.KEY_CLIENT, "size"),
            compras=(cfg.TARGET_INCIDENCE, "sum"),
            tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
        )
        .reset_index()
    )
    por_anio = d.pivot_table(
        index="posicion", columns="anio_panel", values=cfg.TARGET_INCIDENCE, aggfunc="mean"
    ).rename(columns={1: "tasa_anio_1", 2: "tasa_anio_2"})
    out = out.merge(por_anio.reset_index(), on="posicion", how="left")

    out["desvio_vs_global"] = out["tasa_compra"] - rate
    out["error_estandar"] = _binomial_se(rate, out["visitas"])
    out["supera_ruido"] = out["desvio_vs_global"].abs() > 2 * out["error_estandar"]

    corr = float(out["tasa_anio_1"].corr(out["tasa_anio_2"]))
    out.attrs["correlacion_entre_anios"] = round(corr, 3)
    out.attrs["celdas_sobre_ruido"] = int(out["supera_ruido"].sum())
    out.attrs["amplitud"] = round(float(out["tasa_compra"].max() - out["tasa_compra"].min()), 4)
    out.attrs["tasa_global"] = round(rate, 4)
    out.attrs["replica"] = corr > 0.5

    for col in ("tasa_compra", "tasa_anio_1", "tasa_anio_2", "desvio_vs_global", "error_estandar"):
        out[col] = out[col].round(4)
    return out


# --------------------------------------------------------------------------
# Weekly seasonality
# --------------------------------------------------------------------------


def weekly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase rate by position in the 7-day cycle, tested for replication."""
    out = _cycle_profile(df, ((df[cfg.KEY_TIME] - 1) % DAYS_PER_WEEK) + 1)
    out = out.rename(columns={"posicion": "dia_del_ciclo"})

    visitas_cv = float(out["visitas"].std() / out["visitas"].mean())
    out.attrs["cv_visitas"] = round(visitas_cv, 4)
    out.attrs["estacionalidad_semanal"] = bool(
        out.attrs["celdas_sobre_ruido"] > 0 and out.attrs["replica"]
    )
    out.attrs["lectura"] = (
        "{} de 7 posiciones se desvian mas de 2 errores estandar y la amplitud es {:.4f} "
        "sobre una tasa global de {:.4f}. Pero los perfiles de los dos anios del panel "
        "correlacionan {:+.3f}: el patron {}. El trafico por posicion varia un {:.1%}, "
        "asi que tampoco es un efecto de exposicion."
    ).format(
        out.attrs["celdas_sobre_ruido"],
        out.attrs["amplitud"],
        out.attrs["tasa_global"],
        out.attrs["correlacion_entre_anios"],
        "se repite y es estacionalidad" if out.attrs["replica"] else "no se repite, asi que no es estacionalidad",
        visitas_cv,
    )
    return out


# --------------------------------------------------------------------------
# Monthly seasonality
# --------------------------------------------------------------------------


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase rate by position in the 12-block cycle, tested for replication."""
    bloque = (df[cfg.KEY_TIME] - 1) // DAYS_PER_MONTH
    out = _cycle_profile(df, (bloque % 12) + 1)
    out = out.rename(columns={"posicion": "mes_del_ciclo"})

    pico = int(out.loc[out["tasa_compra"].idxmax(), "mes_del_ciclo"])
    valle = int(out.loc[out["tasa_compra"].idxmin(), "mes_del_ciclo"])
    out.attrs["pico"] = pico
    out.attrs["valle"] = valle
    out.attrs["estacionalidad_mensual"] = bool(
        out.attrs["celdas_sobre_ruido"] > 0 and out.attrs["replica"]
    )
    out.attrs["lectura"] = (
        "{} de 12 posiciones superan el ruido, con pico en la {} ({:.4f}) y valle en la "
        "{} ({:.4f}) -- amplitud {:.4f}, tres veces la del ciclo semanal. La correlacion "
        "entre los perfiles de los dos anios es {:+.3f}: {}. El valle de la posicion 1 es "
        "el arranque del panel (ver warmup_profile), no un mes flojo."
    ).format(
        out.attrs["celdas_sobre_ruido"],
        pico,
        float(out["tasa_compra"].max()),
        valle,
        float(out["tasa_compra"].min()),
        out.attrs["amplitud"],
        out.attrs["correlacion_entre_anios"],
        "hay estacionalidad mensual"
        if out.attrs["replica"]
        else "el patron no se repite, asi que no es estacionalidad",
    )
    return out


# --------------------------------------------------------------------------
# Start-of-panel regime
# --------------------------------------------------------------------------


def warmup_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase rate per 30-day block against the share of visits with no history.

    The reference rate is computed on **train only** (blocks from
    ``FIRST_STABLE_BLOCK`` up to the frozen split boundary), so deciding where
    the warm-up ends never looks at test.
    """
    d = df.copy()
    d["bloque"] = ((d[cfg.KEY_TIME] - 1) // DAYS_PER_MONTH) + 1
    d["sin_historial"] = (d["ultima_marca_comprada"] == 0) & (d["ultima_cantidad_comprada"] == 0)

    out = (
        d.groupby("bloque", observed=True)
        .agg(
            dia_inicial=(cfg.KEY_TIME, "min"),
            dia_final=(cfg.KEY_TIME, "max"),
            visitas=(cfg.KEY_CLIENT, "size"),
            tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
            pct_sin_historial=("sin_historial", "mean"),
        )
        .reset_index()
    )

    train = out[(out["bloque"] >= FIRST_STABLE_BLOCK) & (out["dia_final"] <= cfg.SPLIT_DAY)]
    referencia = float(train["tasa_compra"].mean())
    out["umbral"] = referencia - 2 * _binomial_se(referencia, out["visitas"])
    out["en_warmup"] = False

    for i in out.index:
        if out.loc[i, "tasa_compra"] < out.loc[i, "umbral"]:
            out.loc[i, "en_warmup"] = True
        else:
            break

    n_warmup = int(out["en_warmup"].sum())
    dia_fin = int(out.loc[out["en_warmup"], "dia_final"].max()) if n_warmup else 0
    tasa_warmup = (
        float(
            df.loc[df[cfg.KEY_TIME] <= dia_fin, cfg.TARGET_INCIDENCE].mean()
        )
        if n_warmup
        else float("nan")
    )
    tasa_estable = float(df.loc[df[cfg.KEY_TIME] > dia_fin, cfg.TARGET_INCIDENCE].mean())

    for col in ("tasa_compra", "pct_sin_historial", "umbral"):
        out[col] = out[col].round(4)

    out.attrs["referencia_train"] = round(referencia, 4)
    out.attrs["bloques_warmup"] = n_warmup
    out.attrs["dia_fin_warmup"] = dia_fin
    out.attrs["tasa_warmup"] = round(tasa_warmup, 4) if n_warmup else None
    out.attrs["tasa_estable"] = round(tasa_estable, 4)
    out.attrs["hay_warmup"] = n_warmup > 0
    out.attrs["lectura"] = (
        "Los primeros {} bloques (hasta el dia {}) estan por debajo de la referencia de "
        "train ({:.4f}): tasa {:.4f} frente a {:.4f} despues. En el bloque 1 el {:.1%} de "
        "las visitas no tiene historial de compra, y esa proporcion cae bloque a bloque. "
        "No es estacionalidad ni tendencia: es el panel llenandose."
    ).format(
        n_warmup,
        dia_fin,
        referencia,
        tasa_warmup,
        tasa_estable,
        float(out.loc[0, "pct_sin_historial"]),
    ) if n_warmup else (
        "Ningun bloque inicial queda por debajo de la referencia de train ({:.4f}): "
        "no hay regimen de arranque.".format(referencia)
    )
    return out


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------


def _ols_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Slope, its standard error, t and R2 -- enough to call a trend or not."""
    pendiente, intercepto = np.polyfit(x, y, 1)
    ajuste = intercepto + pendiente * x
    residuos = y - ajuste
    var_resid = float((residuos**2).sum() / (len(x) - 2))
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt(var_resid / sxx))
    return {
        "pendiente": float(pendiente),
        "se": se,
        "t": float(pendiente / se) if se else float("nan"),
        "r2": float(1 - (residuos**2).sum() / ((y - y.mean()) ** 2).sum()),
        "ajuste": ajuste,
    }


def purchase_rate_trend(df: pd.DataFrame, desde_bloque: int | None = None) -> pd.DataFrame:
    """Purchase rate per 30-day block with its slope, before and after the warm-up.

    Fitting one line through the whole horizon mixes two things: the panel
    filling up and whatever happens afterwards. The second slope is the one
    that answers whether train and test measure the same regime.
    """
    d = df.copy()
    d["bloque"] = ((d[cfg.KEY_TIME] - 1) // DAYS_PER_MONTH) + 1

    out = (
        d.groupby("bloque", observed=True)
        .agg(
            dia_inicial=(cfg.KEY_TIME, "min"),
            dia_final=(cfg.KEY_TIME, "max"),
            visitas=(cfg.KEY_CLIENT, "size"),
            tasa_compra=(cfg.TARGET_INCIDENCE, "mean"),
        )
        .reset_index()
    )

    if desde_bloque is None:
        desde_bloque = int(warmup_profile(df).attrs["bloques_warmup"]) + 1

    total = _ols_slope(out["bloque"].to_numpy(float), out["tasa_compra"].to_numpy(float))
    estable = out[out["bloque"] >= desde_bloque]
    post = _ols_slope(estable["bloque"].to_numpy(float), estable["tasa_compra"].to_numpy(float))

    out["tendencia_completa"] = np.round(total["ajuste"], 4)
    out["tendencia_post_warmup"] = np.nan
    out.loc[out["bloque"] >= desde_bloque, "tendencia_post_warmup"] = np.round(post["ajuste"], 4)
    out["tasa_compra"] = out["tasa_compra"].round(4)

    out.attrs["desde_bloque"] = desde_bloque
    out.attrs["pendiente_completa"] = round(total["pendiente"], 6)
    out.attrs["t_completa"] = round(total["t"], 2)
    out.attrs["pendiente_post_warmup"] = round(post["pendiente"], 6)
    out.attrs["error_estandar_post_warmup"] = round(post["se"], 6)
    out.attrs["t_post_warmup"] = round(post["t"], 2)
    out.attrs["r2_post_warmup"] = round(post["r2"], 4)
    out.attrs["hay_tendencia"] = abs(post["t"]) > 2
    out.attrs["lectura"] = (
        "Sobre los 25 bloques completos la pendiente es {:+.6f} (t={:+.2f}), pero esa "
        "recta esta arrastrada por el arranque. Desde el bloque {} la pendiente es "
        "{:+.6f} por bloque de 30 dias (error estandar {:.6f}, t={:+.2f}, R2={:.4f}): "
        "{}. Es la condicion que vuelve comparables train y test."
    ).format(
        total["pendiente"],
        total["t"],
        desde_bloque,
        post["pendiente"],
        post["se"],
        post["t"],
        post["r2"],
        "hay tendencia" if abs(post["t"]) > 2 else "sin tendencia distinguible del ruido",
    )
    return out


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def temporal_dynamics_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One verdict per temporal question, each with its consequence declared."""
    sem = weekly_seasonality(df)
    mes = monthly_seasonality(df)
    cal = warmup_profile(df)
    ten = purchase_rate_trend(df)

    filas = [
        {
            "pregunta": "estacionalidad semanal",
            "medida": "amplitud {:.4f}; {}/7 celdas sobre el ruido; correlacion entre anios {:+.3f}".format(
                sem.attrs["amplitud"],
                sem.attrs["celdas_sobre_ruido"],
                sem.attrs["correlacion_entre_anios"],
            ),
            "veredicto": "si" if sem.attrs["estacionalidad_semanal"] else "no",
            "consecuencia": (
                "entra un indicador de posicion del ciclo como feature"
                if sem.attrs["estacionalidad_semanal"]
                else "no se crea feature de dia de la semana: el patron no se repite entre anios"
            ),
        },
        {
            "pregunta": "estacionalidad mensual",
            "medida": "amplitud {:.4f}; {}/12 celdas sobre el ruido; correlacion entre anios {:+.3f}".format(
                mes.attrs["amplitud"],
                mes.attrs["celdas_sobre_ruido"],
                mes.attrs["correlacion_entre_anios"],
            ),
            "veredicto": "si" if mes.attrs["estacionalidad_mensual"] else "no",
            "consecuencia": (
                "el calendario promocional de P2 debe respetar el ciclo"
                if mes.attrs["estacionalidad_mensual"]
                else "P2 puede reasignar dias de promocion sin restriccion estacional"
            ),
        },
        {
            "pregunta": "regimen de arranque del panel",
            "medida": "{} bloques hasta el dia {}; tasa {} vs {} despues".format(
                cal.attrs["bloques_warmup"],
                cal.attrs["dia_fin_warmup"],
                cal.attrs["tasa_warmup"],
                cal.attrs["tasa_estable"],
            ),
            "veredicto": "si" if cal.attrs["hay_warmup"] else "no",
            "consecuencia": (
                "los primeros {} dias son otro regimen; entran al entrenamiento pero el "
                "indicador de ausencia de historial es lo que debe explicarlos, y las "
                "metricas se reportan tambien sin ellos".format(cal.attrs["dia_fin_warmup"])
                if cal.attrs["hay_warmup"]
                else "no hay regimen de arranque que aislar"
            ),
        },
        {
            "pregunta": "tendencia de la tasa de compra",
            "medida": "pendiente {:+.6f} por bloque tras el arranque (t={:+.2f})".format(
                ten.attrs["pendiente_post_warmup"], ten.attrs["t_post_warmup"]
            ),
            "veredicto": "si" if ten.attrs["hay_tendencia"] else "no",
            "consecuencia": (
                "el split temporal mide un regimen distinto: hay que reponderar o detrendear"
                if ten.attrs["hay_tendencia"]
                else "el split temporal es comparable; semana y mes como indices de tendencia "
                "no aportan senal y su extrapolacion en test es riesgo sin beneficio "
                "(decision de la Fase 4)"
            ),
        },
    ]
    out = pd.DataFrame(filas)
    hay_estacionalidad = (
        sem.attrs["estacionalidad_semanal"] or mes.attrs["estacionalidad_mensual"]
    )
    out.attrs["hay_estacionalidad"] = hay_estacionalidad
    out.attrs["lectura"] = (
        "Lo que parecia estacionalidad semanal y mensual no se repite entre los dos anios "
        "del panel: la unica dinamica temporal real son los primeros {} dias de arranque. "
        "Despues de ellos la tasa de compra es plana, y eso es exactamente lo que el split "
        "temporal del dia {} necesita para ser valido."
    ).format(cal.attrs["dia_fin_warmup"], cfg.SPLIT_DAY) if not hay_estacionalidad else (
        "Al menos una estacionalidad replica entre anios y tiene consecuencia sobre el modelado."
    )
    return out
