"""Figures for the EDA.

Every function takes the output of an ``eda.*`` function and returns a
Matplotlib ``Axes``. None of them computes anything: the numbers come from the
analysis modules, so a figure can never disagree with the table beside it.

Each figure carries the decision it informs in its title or annotation -- the
alignment brief rejects plots that do not state their consequence.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

PALETTE = {
    "primary": "#2B6CB0",
    "accent": "#DD6B20",
    "muted": "#A0AEC0",
    "good": "#38A169",
    "bad": "#E53E3E",
}


def set_style() -> None:
    """Apply a consistent, readable style across the notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def _annotate_bars(ax, values, fmt="{:.3f}", offset=0.002):
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        ax.text(i, v + offset, fmt.format(v), ha="center", va="bottom", fontsize=8)


# --------------------------------------------------------------------------
# Block A -- objective parameters
# --------------------------------------------------------------------------


def plot_rate_stability(by_period: pd.DataFrame, global_rate: float, ax=None):
    """A1: the base rate over time, against the global average."""
    ax = ax or plt.gca()
    ax.bar(by_period["periodo"].astype(str), by_period["tasa_compra"], color=PALETTE["primary"])
    ax.axhline(global_rate, color=PALETTE["accent"], linestyle="--", label=f"global {global_rate:.4f}")
    _annotate_bars(ax, by_period["tasa_compra"])
    ax.set_title("A1 · Tasa de compra estable por periodo → el split temporal es valido")
    ax.set_ylabel("tasa de compra")
    ax.set_ylim(0, by_period["tasa_compra"].max() * 1.25)
    ax.legend()
    return ax


def plot_price_elasticity(by_quintile: pd.DataFrame, ax=None):
    """A2: incidence against the day's cheapest price -- the elasticity signal."""
    ax = ax or plt.gca()
    labels = [f"Q{i + 1}\n{p:.2f}" for i, p in enumerate(by_quintile["precio_min_medio"])]
    ax.plot(labels, by_quintile["tasa_compra"], marker="o", color=PALETTE["primary"], linewidth=2)
    for i, v in enumerate(by_quintile["tasa_compra"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    drop = by_quintile.attrs.get("caida_relativa_pct")
    ax.set_title(f"A2 · Incidencia vs precio minimo del dia → caida relativa {drop}%")
    ax.set_xlabel("quintil de precio minimo (precio medio del quintil)")
    ax.set_ylabel("tasa de compra")
    return ax


def plot_promo_incidence(by_n_promos: pd.DataFrame, ax=None):
    """A3: incidence by number of brands on promotion -- note the non-monotonicity."""
    ax = ax or plt.gca()
    colors = [PALETTE["primary"]] * len(by_n_promos)
    peak = int(by_n_promos["tasa_compra"].idxmax())
    colors[peak] = PALETTE["good"]
    ax.bar(by_n_promos["n_promos_dia"].astype(str), by_n_promos["tasa_compra"], color=colors)
    _annotate_bars(ax, by_n_promos["tasa_compra"])
    ax.set_title("A3 · Incidencia por n de marcas en promocion → el efecto no es monotono")
    ax.set_xlabel("marcas en promocion ese dia")
    ax.set_ylabel("tasa de compra")
    ax.set_ylim(0, by_n_promos["tasa_compra"].max() * 1.2)
    return ax


def plot_brand_share(brand_share: pd.DataFrame, ax=None):
    """A4: share of purchases versus share of revenue, per brand."""
    ax = ax or plt.gca()
    x = np.arange(len(brand_share))
    width = 0.38
    ax.bar(x - width / 2, brand_share["share_compras_pct"], width,
           label="share de compras", color=PALETTE["muted"])
    ax.bar(x + width / 2, brand_share["share_ingreso_pct"], width,
           label="share de ingreso", color=PALETTE["primary"])
    ax.set_xticks(x, [f"marca {b}" for b in brand_share[cfg.TARGET_BRAND]])
    ax.set_title("A4 · Share de compras vs share de ingreso → la marca 5 concentra valor")
    ax.set_ylabel("%")
    ax.legend()
    return ax


def plot_promo_lift(lift: pd.DataFrame, ax=None):
    """A5: the central finding -- promotional investment against response."""
    ax = ax or plt.gca()
    colors = [PALETTE["bad"] if "SOBRE" in d else PALETTE["good"] if "SUB" in d else PALETTE["muted"]
              for d in lift["respuesta_vs_inversion"]]
    ax.scatter(lift["frecuencia_promo"], lift["lift"], s=180, c=colors, zorder=3)
    for _, r in lift.iterrows():
        ax.annotate(f"marca {int(r['marca'])}", (r["frecuencia_promo"], r["lift"]),
                    textcoords="offset points", xytext=(10, 4), fontsize=9)
    ax.axhline(lift["lift"].median(), color=PALETTE["muted"], linestyle=":", linewidth=1)
    ax.axvline(lift["frecuencia_promo"].median(), color=PALETTE["muted"], linestyle=":", linewidth=1)
    ax.set_title("A5 · Inversion promocional vs respuesta → las marcas 3 y 5 estan sub-promocionadas")
    ax.set_xlabel("frecuencia de promocion (dias en promo / total)")
    ax.set_ylabel("lift del share cuando esta en promo")
    return ax


def plot_switching_matrix(matrix: pd.DataFrame, ax=None):
    """A5: loyalty on the diagonal, substitution off it."""
    ax = ax or plt.gca()
    im = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=matrix.to_numpy().max())
    ax.set_xticks(range(len(matrix.columns)), [f"→{c}" for c in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [f"{i}" for i in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > matrix.to_numpy().max() * 0.6 else "black")
    ax.set_title("A5 · Matriz de cambio de marca → diagonal = lealtad, resto = sustitucion")
    ax.set_xlabel("marca actual")
    ax.set_ylabel("marca anterior")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04)
    return ax


def plot_quantity_distribution(qty: pd.DataFrame, ax=None):
    """A6: quantity given purchase -- the shape model M3 must reproduce."""
    ax = ax or plt.gca()
    ax.bar(qty["cantidad"].astype(str), qty["compras"], color=PALETTE["primary"])
    ax.set_title("A6 · Cantidad dado compra → concentrada en 1-3 unidades")
    ax.set_xlabel("unidades compradas")
    ax.set_ylabel("n de compras")
    return ax


# --------------------------------------------------------------------------
# Block B -- decision variables
# --------------------------------------------------------------------------


def plot_price_series(daily: pd.DataFrame, ax=None):
    """B2: the five price trajectories -- the feasible set of ``y_jtk`` over time."""
    ax = ax or plt.gca()
    for j in cfg.BRANDS:
        ax.plot(daily[cfg.KEY_TIME], daily[f"precio_marca_{j}"],
                label=f"marca {j}", color=cfg.BRAND_COLORS[j], linewidth=1)
    ax.set_title("B2 · Trayectoria de precios por marca → escalera ordenada en medias, rangos solapados")
    ax.set_xlabel("dia")
    ax.set_ylabel("precio")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return ax


def plot_price_ladder(ladder: pd.DataFrame, ax=None):
    """B1: the discrete price menu per brand, which makes the MILP tractable."""
    ax = ax or plt.gca()
    for i, r in ladder.iterrows():
        j = int(r["marca"])
        ax.plot([r["precio_min"], r["precio_max"]], [j, j], color=cfg.BRAND_COLORS[j], linewidth=6, alpha=0.7)
        ax.scatter([r["precio_medio"]], [j], color="black", zorder=3, s=25)
        ax.text(r["precio_max"] + 0.03, j, f"{int(r['n_puntos_precio'])} puntos", va="center", fontsize=8)
    ax.set_yticks(list(cfg.BRANDS), [f"marca {j}" for j in cfg.BRANDS])
    ax.set_title("B1 · Escalera de precios observada → el espacio de decision es finito")
    ax.set_xlabel("precio")
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_promo_calendar(df: pd.DataFrame, ax=None, max_days: int = 365):
    """B3: when each brand is on promotion -- the shape of ``z_jt``."""
    ax = ax or plt.gca()
    daily = df.groupby(cfg.KEY_TIME)[cfg.PROMO_COLS].first().head(max_days)
    for k, j in enumerate(cfg.BRANDS):
        on = daily.index[daily[f"promo_marca_{j}"] == 1]
        ax.scatter(on, np.full(len(on), j), marker="|", s=60,
                   color=cfg.BRAND_COLORS[j], linewidths=1.2)
    ax.set_yticks(list(cfg.BRANDS), [f"marca {j}" for j in cfg.BRANDS])
    ax.set_title(f"B3 · Calendario promocional (primeros {max_days} dias) → marcas 1-2 saturadas")
    ax.set_xlabel("dia")
    ax.grid(axis="x", alpha=0.2)
    return ax


def plot_client_heterogeneity(client_prof: pd.DataFrame, ax=None):
    """B4: the distribution of purchase rate across clients."""
    ax = ax or plt.gca()
    ax.hist(client_prof["tasa_compra"], bins=40, color=PALETTE["primary"], edgecolor="white")
    median = client_prof["tasa_compra"].median()
    ax.axvline(median, color=PALETTE["accent"], linestyle="--", label=f"mediana {median:.3f}")
    ax.set_title("B4 · Tasa de compra por cliente → heterogeneidad que justifica el targeting")
    ax.set_xlabel("tasa de compra del cliente")
    ax.set_ylabel("n de clientes")
    ax.legend()
    return ax


def plot_pareto(curve: pd.DataFrame, concentration: pd.DataFrame, ax=None):
    """B5: the revenue concentration curve, with the top-decile marker."""
    ax = ax or plt.gca()
    ax.plot(curve["pct_clientes"], curve["pct_ingreso_acumulado"],
            color=PALETTE["primary"], linewidth=2)
    ax.plot([0, 100], [0, 100], color=PALETTE["muted"], linestyle=":", label="distribucion uniforme")
    for _, r in concentration.iterrows():
        ax.scatter([r["top_pct_clientes"]], [r["share_ingreso_pct"]], color=PALETTE["accent"], zorder=3)
        ax.annotate(f"{r['share_ingreso_pct']}%", (r["top_pct_clientes"], r["share_ingreso_pct"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_title("B5 · Concentracion de ingreso → el top 20% aporta mas de la mitad")
    ax.set_xlabel("% de clientes (ordenados por gasto)")
    ax.set_ylabel("% de ingreso acumulado")
    ax.legend()
    return ax


def plot_targeting_baseline(baseline: pd.DataFrame, ax=None):
    """B6: the lift curve the ML model must beat."""
    ax = ax or plt.gca()
    base = baseline.attrs.get("tasa_base", 0)
    ax.bar(baseline["top_pct_clientes"].astype(str), baseline["tasa_compra_del_grupo"],
           color=PALETTE["primary"])
    ax.axhline(base, color=PALETTE["accent"], linestyle="--", label=f"tasa base {base:.4f}")
    for i, r in baseline.iterrows():
        ax.text(i, r["tasa_compra_del_grupo"] + 0.01, f"x{r['lift_vs_base']}", ha="center", fontsize=8)
    ax.set_title("B6 · Lift del targeting historico → piso que M1 debe superar")
    ax.set_xlabel("top % de clientes por tasa historica")
    ax.set_ylabel("tasa de compra del grupo")
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Block D -- segmentation
# --------------------------------------------------------------------------


def plot_demographic_spread(spread: pd.DataFrame, client_spread: float, ax=None):
    """D3: demographic spread against client-level spread."""
    ax = ax or plt.gca()
    ax.barh(spread["variable"], spread["amplitud"], color=PALETTE["muted"])
    ax.axvline(client_spread, color=PALETTE["accent"], linestyle="--",
               label=f"amplitud entre clientes {client_spread:.3f}")
    for i, v in enumerate(spread["amplitud"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title("D3 · Amplitud de tasa explicada → la demografia segmenta, no predice")
    ax.set_xlabel("amplitud de la tasa de compra (max - min)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Fase 1 of the methodology brief -- dynamics, price gap, promo, anomalies
# --------------------------------------------------------------------------


def plot_weekly_seasonality(weekly: pd.DataFrame, global_rate: float, ax=None):
    """Purchase rate by position in the 7-day cycle, against the noise band.

    The band is +/- 2 binomial standard errors: bars inside it are sampling
    noise, so the figure answers "is there a cycle" instead of showing seven
    bars of different heights and leaving the reader to guess.
    """
    ax = ax or plt.gca()
    se = float(weekly["error_estandar"].mean())
    colors = [
        PALETTE["accent"] if flag else PALETTE["primary"] for flag in weekly["supera_ruido"]
    ]
    ax.bar(weekly["dia_del_ciclo"].astype(str), weekly["tasa_compra"], color=colors)
    ax.axhline(global_rate, color=PALETTE["bad"], linestyle="--", label=f"global {global_rate:.4f}")
    ax.axhspan(global_rate - 2 * se, global_rate + 2 * se, color=PALETTE["muted"], alpha=0.25,
               label="+/- 2 errores estandar")
    _annotate_bars(ax, weekly["tasa_compra"], offset=0.003)
    n_sig = weekly.attrs.get("celdas_sobre_ruido", 0)
    veredicto = "hay ciclo semanal" if n_sig else "sin ciclo semanal"
    ax.set_title(f"Estacionalidad semanal → {veredicto} ({n_sig}/7 celdas sobre el ruido)")
    ax.set_xlabel("posicion en el ciclo de 7 dias de dia_visita")
    ax.set_ylabel("tasa de compra")
    ax.set_ylim(0, weekly["tasa_compra"].max() * 1.25)
    ax.legend(loc="lower right")
    return ax


def plot_rate_trend(trend: pd.DataFrame, ax=None):
    """Purchase rate per 30-day block, with the full-horizon and post-warm-up fits.

    Two lines on purpose: the first one is what you get by fitting through the
    start-of-panel ramp, the second is the regime the split actually compares.
    """
    ax = ax or plt.gca()
    desde = trend.attrs.get("desde_bloque", 1)
    ax.plot(trend["bloque"], trend["tasa_compra"], marker="o", markersize=4,
            color=PALETTE["primary"], linewidth=1.5, label="tasa observada")
    ax.plot(trend["bloque"], trend["tendencia_completa"], color=PALETTE["muted"],
            linestyle="--", linewidth=1.5,
            label="ajuste sobre todo el horizonte (t={:+.2f})".format(trend.attrs["t_completa"]))
    ax.plot(trend["bloque"], trend["tendencia_post_warmup"], color=PALETTE["accent"],
            linestyle="--", linewidth=2,
            label="ajuste tras el arranque (t={:+.2f})".format(trend.attrs["t_post_warmup"]))
    ax.axvspan(0.5, desde - 0.5, color=PALETTE["bad"], alpha=0.08, label="arranque del panel")
    ax.axvline(cfg.SPLIT_DAY / 30, color=PALETTE["bad"], linestyle=":", linewidth=1.5,
               label=f"frontera del split (dia {cfg.SPLIT_DAY})")
    veredicto = "hay tendencia" if trend.attrs.get("hay_tendencia") else "sin tendencia"
    ax.set_title("Tendencia de la tasa de compra → {} tras el arranque".format(veredicto))
    ax.set_xlabel("bloque de 30 dias")
    ax.set_ylabel("tasa de compra")
    ax.legend(loc="lower right", ncol=2)
    return ax


def plot_warmup(warm: pd.DataFrame, ax=None):
    """The start-of-panel regime: purchase rate against visits with no history."""
    ax = ax or plt.gca()
    ax.plot(warm["bloque"], warm["tasa_compra"], marker="o", markersize=4,
            color=PALETTE["primary"], linewidth=1.5, label="tasa de compra")
    ax.axhline(warm.attrs["referencia_train"], color=PALETTE["accent"], linestyle="--",
               linewidth=1.5, label="referencia de train {:.4f}".format(warm.attrs["referencia_train"]))
    n = warm.attrs.get("bloques_warmup", 0)
    if n:
        ax.axvspan(0.5, n + 0.5, color=PALETTE["bad"], alpha=0.1,
                   label="arranque (hasta el dia {})".format(warm.attrs["dia_fin_warmup"]))
    ax.set_xlabel("bloque de 30 dias")
    ax.set_ylabel("tasa de compra")
    ax.set_title("Arranque del panel → el valle inicial es historial ausente, no demanda baja")

    ax2 = ax.twinx()
    ax2.plot(warm["bloque"], 100 * warm["pct_sin_historial"], color=PALETTE["muted"],
             linewidth=1.5, linestyle=":", label="% visitas sin historial")
    ax2.set_ylabel("% de visitas sin historial")
    ax2.grid(False)

    lineas = ax.get_legend_handles_labels()
    lineas2 = ax2.get_legend_handles_labels()
    ax.legend(lineas[0] + lineas2[0], lineas[1] + lineas2[1], loc="lower right")
    return ax


def plot_relative_price_gap(gap: pd.DataFrame, ax=None):
    """Relative price of each brand: level (the position) and range (the lever)."""
    ax = ax or plt.gca()
    x = range(len(gap))
    ax.vlines(x, gap["precio_relativo_min"], gap["precio_relativo_max"],
              color=PALETTE["muted"], linewidth=6, label="rango observado")
    ax.scatter(x, gap["precio_relativo_medio"], color=PALETTE["primary"], zorder=3, s=55,
               label="precio relativo medio")
    ax.axhline(1.0, color=PALETTE["bad"], linestyle="--", linewidth=1,
               label="promedio del dia")
    for i, (m, v) in enumerate(zip(gap["precio_relativo_medio"], gap["pct_dias_mas_barata"])):
        ax.text(i, m + 0.03, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i, gap["precio_relativo_min"].iloc[i] - 0.05, f"mas barata\n{v:.0f}% dias",
                ha="center", fontsize=7, color=PALETTE["muted"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"marca {j}" for j in gap["marca"]])
    ax.set_title("Brecha de precio relativo → posicion de marca y margen de maniobra de P3")
    ax.set_ylabel("precio / promedio del dia")
    return ax


def plot_promo_incidence_by_brand(tab: pd.DataFrame, ax=None):
    """Purchase rate with and without promotion, brand by brand."""
    ax = ax or plt.gca()
    x = np.arange(len(tab))
    w = 0.38
    ax.bar(x - w / 2, tab["tasa_sin_promo"], width=w, color=PALETTE["muted"], label="sin promo")
    ax.bar(x + w / 2, tab["tasa_con_promo"], width=w,
           color=[PALETTE["good"] if e == "expande el mercado" else PALETTE["primary"]
                  for e in tab["efecto"]],
           label="con promo")
    for i, row in tab.reset_index(drop=True).iterrows():
        ax.text(i, max(row["tasa_con_promo"], row["tasa_sin_promo"]) + 0.006,
                f"{row['delta_pp']:+.2f} pp", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"marca {j}" for j in tab["marca"]])
    ax.set_title("Efecto bruto de promocion sobre la incidencia → expansion vs canibalizacion")
    ax.set_ylabel("tasa de compra")
    ax.set_ylim(0, tab[["tasa_con_promo", "tasa_sin_promo"]].to_numpy().max() * 1.25)
    ax.legend(loc="upper right")
    return ax


def plot_quantity_tail(qty: pd.DataFrame, ax=None):
    """Share of purchases versus share of units for each quantity value.

    The gap between the two series is the whole point: the tail is a small
    fraction of the purchases and a large fraction of the volume, which is what
    decides how M3 is evaluated.
    """
    ax = ax or plt.gca()
    x = np.arange(len(qty))
    w = 0.38
    umbral = qty.attrs.get("umbral_cola")
    ax.bar(x - w / 2, qty["pct_compras"], width=w, color=PALETTE["primary"], label="% de compras")
    ax.bar(x + w / 2, qty["pct_unidades"], width=w, color=PALETTE["accent"], label="% de unidades")
    if umbral is not None and (qty["cantidad"] >= umbral).any():
        corte = float(np.argmax(qty["cantidad"].to_numpy() >= umbral)) - 0.5
        ax.axvline(corte, color=PALETTE["bad"], linestyle=":", linewidth=1.5,
                   label=f"cola (cantidad >= {umbral})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(qty["cantidad"].astype(int))
    ax.set_title("Cantidades extremas → {:.1f}% de las compras y {:.1f}% de las unidades".format(
        qty.attrs.get("pct_compras_en_cola", float("nan")),
        qty.attrs.get("pct_unidades_en_cola", float("nan")),
    ))
    ax.set_xlabel("unidades por compra")
    ax.set_ylabel("% del total")
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Fase 3 -- model M0: customer segmentation
# --------------------------------------------------------------------------


def plot_k_selection(seleccion: pd.DataFrame, ax=None):
    """Elbow and silhouette on the same axes, with the chosen k marked.

    Both criteria the brief names, side by side, because the decision is only
    defensible if you can see that they disagree and how the tie was broken.
    """
    ax = ax or plt.gca()
    k_rec = seleccion.attrs.get("k_recomendado")
    ax.plot(seleccion["k"], seleccion["inercia"], marker="o", color=PALETTE["primary"],
            linewidth=2, label="inercia (codo)")
    ax.set_xlabel("k")
    ax.set_ylabel("inercia", color=PALETTE["primary"])
    ax.set_xticks(list(seleccion["k"]))

    ax2 = ax.twinx()
    ax2.plot(seleccion["k"], seleccion["silhouette"], marker="s", markersize=4,
             color=PALETTE["accent"], linewidth=2, label="silhouette")
    ax2.set_ylabel("silhouette", color=PALETTE["accent"])
    ax2.grid(False)

    if k_rec is not None:
        ax.axvline(k_rec, color=PALETTE["bad"], linestyle=":", linewidth=1.5,
                   label=f"k elegido = {k_rec}")
    h1 = ax.get_legend_handles_labels()
    h2 = ax2.get_legend_handles_labels()
    ax.legend(h1[0] + h2[0], h1[1] + h2[1], loc="upper right")
    ax.set_title("M0 · Selección de k → codo en k={}, silhouette máxima en k={}".format(
        seleccion.attrs.get("k_codo"), seleccion.attrs.get("k_silhouette")))
    return ax


def plot_pca_segments(projection: pd.DataFrame, ax=None):
    """Clients on the first two components, coloured by segment.

    PCA is a window here, not the model: K-Means ran on the full standardized
    space, so what this figure shows is how much of the partition survives a
    two-dimensional view.
    """
    ax = ax or plt.gca()
    var = projection.attrs.get("varianza_explicada", [float("nan"), float("nan")])
    colores = [PALETTE["primary"], PALETTE["accent"], PALETTE["good"], PALETTE["bad"],
               PALETTE["muted"], "#6B46C1", "#D53F8C", "#2C7A7B", "#975A16", "#4A5568"]
    for i, (seg, part) in enumerate(projection.groupby("segmento")):
        nombre = part["nombre"].iloc[0] if "nombre" in part.columns else f"segmento {seg}"
        ax.scatter(part["pc1"], part["pc2"], s=18, alpha=0.75,
                   color=colores[i % len(colores)], label=f"{seg} · {nombre}")
        ax.scatter(part["pc1"].mean(), part["pc2"].mean(), s=220, marker="X",
                   color=colores[i % len(colores)], edgecolor="white", linewidth=1.5, zorder=5)
    ax.set_xlabel(f"PC1 ({var[0]:.1%} de la varianza)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} de la varianza)")
    ax.set_title("M0 · Segmentos sobre las dos primeras componentes (X = centroide)")
    ax.legend(loc="best", fontsize=7)
    return ax


def plot_segment_behaviour(behaviour: pd.DataFrame, global_rate: float, ax=None):
    """Purchase rate per segment with 95% intervals, against the global rate.

    The intervals are the argument: they are what turns "these bars differ"
    into "the segment separates behaviour".
    """
    ax = ax or plt.gca()
    x = np.arange(len(behaviour))
    err = 1.96 * behaviour["error_estandar"]
    ax.bar(x, behaviour["tasa_compra"], color=PALETTE["primary"], yerr=err, capsize=5,
           error_kw={"ecolor": PALETTE["bad"], "elinewidth": 1.4})
    ax.axhline(global_rate, color=PALETTE["accent"], linestyle="--",
               label=f"global {global_rate:.4f}")
    for i, row in behaviour.reset_index(drop=True).iterrows():
        ax.text(i, row["tasa_compra"] + 1.96 * row["error_estandar"] + 0.004,
                "{:.4f}\n{:.1f}% del ingreso".format(row["tasa_compra"], row["pct_ingreso"]),
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["{}\n{}".format(s, n) for s, n in zip(behaviour["segmento"], behaviour["nombre"])],
        fontsize=8,
    )
    ax.set_ylabel("tasa de compra")
    ax.set_ylim(0, behaviour["tasa_compra"].max() * 1.45)
    ax.set_title("M0 · Comportamiento por segmento → amplitud {:.4f}, intervalos al 95%".format(
        behaviour.attrs.get("amplitud_tasa", float("nan"))))
    ax.legend(loc="lower right")
    return ax


def plot_segment_brand_share(brand_share: pd.DataFrame, ax=None):
    """Deviation of each segment's brand mix from the global mix, in points.

    Plotting the deviation rather than the share is deliberate: the decision
    this figure informs is whether to differentiate price and promotion by
    segment, and that depends on the gap against the average, not the level.
    """
    ax = ax or plt.gca()
    cols = [c for c in brand_share.columns if c.startswith("desvio_")]
    M = brand_share.set_index("segmento")[cols]
    lim = float(np.abs(M.to_numpy()).max())
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)

    ax.set_xticks(range(len(cols)), [c.replace("desvio_marca_", "marca ") for c in cols])
    etiquetas = [
        "{} · {}".format(s, brand_share.loc[brand_share["segmento"] == s, "nombre"].iloc[0])
        for s in M.index
    ]
    ax.set_yticks(range(len(M.index)), etiquetas, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iloc[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > lim * 0.6 else "black")
    ax.set_title("M0 · Preferencia de marca por segmento (desvío en pp vs. share global)")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04, label="puntos porcentuales")
    return ax


def plot_segment_promo_response(promo: pd.DataFrame, ax=None):
    """Incidence lift from any promotion, per segment, with its z."""
    ax = ax or plt.gca()
    x = np.arange(len(promo))
    colores = [PALETTE["good"] if r == "responde" else PALETTE["muted"] for r in promo["responde"]]
    ax.bar(x, promo["delta_pp"], color=colores)
    for i, row in promo.reset_index(drop=True).iterrows():
        ax.text(i, row["delta_pp"] + 0.1, "z={:.1f}".format(row["z"]), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["{}\n{}".format(s, n) for s, n in zip(promo["segmento"], promo["nombre"])], fontsize=8
    )
    ax.set_ylabel("Δ incidencia con promoción (pp)")
    ax.set_ylim(0, promo["delta_pp"].max() * 1.3)
    ax.set_title("M0 · Respuesta a promoción por segmento → a quién dirige P2 el calendario")
    return ax


# --------------------------------------------------------------------------
# Fase 4 -- models M1, M2, M3
# --------------------------------------------------------------------------


def plot_pr_roc(y, proba, ax=None):
    """Precision-recall and ROC side by side, each against its own baseline.

    PR is the one that decides (25% positives); ROC is shown because it is the
    number everyone asks for, and the gap between the two is itself the
    argument for not reading ROC alone under class imbalance.
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    ax = ax or plt.gca()
    base = float(np.mean(y))
    p, r, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    ax.plot(r, p, color=PALETTE["primary"], linewidth=2, label=f"PR-AUC {ap:.4f}")
    ax.axhline(base, color=PALETTE["bad"], linestyle="--", linewidth=1.2,
               label=f"azar {base:.4f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"M1 · Curva precision-recall → PR-AUC {ap:.4f} vs azar {base:.4f}")
    ax.legend(loc="upper right")

    fpr, tpr, _ = roc_curve(y, proba)
    inset = ax.inset_axes([0.58, 0.45, 0.38, 0.42])
    inset.plot(fpr, tpr, color=PALETTE["accent"], linewidth=1.6)
    inset.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle=":", linewidth=1)
    inset.set_title(f"ROC {roc_auc_score(y, proba):.4f}", fontsize=8)
    inset.tick_params(labelsize=6)
    inset.grid(False)
    return ax


def plot_gains(contra_baseline: pd.DataFrame, ax=None):
    """Lift by budget, model against the policy already in use."""
    ax = ax or plt.gca()
    pivote = contra_baseline.pivot(index="top", columns="politica", values="lift")
    orden = sorted(pivote.index, key=lambda s: int(s.rstrip("%")))
    pivote = pivote.loc[orden]
    x = np.arange(len(pivote))
    w = 0.38
    ax.bar(x - w / 2, pivote["politica historica"], width=w, color=PALETTE["muted"],
           label="politica historica")
    ax.bar(x + w / 2, pivote["modelo M1"], width=w, color=PALETTE["primary"], label="modelo M1")
    for i, (a, b) in enumerate(zip(pivote["politica historica"], pivote["modelo M1"])):
        ax.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.03, f"{b:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color=PALETTE["bad"], linestyle="--", linewidth=1, label="azar")
    ax.set_xticks(x, pivote.index)
    ax.set_xlabel("presupuesto de contacto (top % del score)")
    ax.set_ylabel("lift sobre la tasa base")
    ax.set_title("M1 · Lift por presupuesto → el modelo contra la politica vigente")
    ax.legend(loc="upper right")
    return ax


def plot_threshold_value(y, proba, ratios=(0.1, 0.2, 0.3), ax=None):
    """Net value as a function of the threshold, one curve per cost ratio.

    The vertical marks are the theoretical optimum ``umbral = costo/margen``.
    Seeing the empirical maximum land on them is what turns the rule from an
    assertion into a measurement -- and seeing where 0.5 falls is what retires
    the convention.
    """
    ax = ax or plt.gca()
    yv = np.asarray(y)
    rejilla = np.linspace(0.02, 0.95, 187)
    colores = [PALETTE["primary"], PALETTE["accent"], PALETTE["good"]]
    for i, r in enumerate(ratios):
        valores = [float(yv[proba >= t].sum()) - r * int((proba >= t).sum()) for t in rejilla]
        ax.plot(rejilla, valores, color=colores[i % 3], linewidth=1.8,
                label=f"costo/margen = {r:.2f}")
        ax.axvline(r, color=colores[i % 3], linestyle=":", linewidth=1.2)
    ax.axvline(0.5, color=PALETTE["bad"], linestyle="--", linewidth=1.5,
               label="convencion 0.5")
    ax.set_xlabel("umbral de probabilidad")
    ax.set_ylabel("valor neto en test (unidades de margen)")
    ax.set_title("Umbral de operacion · el optimo es costo/margen, no 0.5")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_calibration(y, proba, bins: int = 10, ax=None):
    """Observed frequency against predicted probability, by decile of the score.

    The cost-based threshold only means what it says if the probabilities are
    calibrated: ``p * margen > costo`` compares a probability with money.
    """
    ax = ax or plt.gca()
    d = pd.DataFrame({"p": proba, "y": np.asarray(y)})
    d["bin"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(predicha=("p", "mean"), observada=("y", "mean"),
                                            n=("y", "size"))
    ax.plot([0, g["predicha"].max() * 1.05], [0, g["predicha"].max() * 1.05],
            color=PALETTE["muted"], linestyle=":", linewidth=1.2, label="calibracion perfecta")
    ax.plot(g["predicha"], g["observada"], marker="o", color=PALETTE["primary"], linewidth=2,
            label="observada")
    ax.set_xlabel("probabilidad predicha (media del decil)")
    ax.set_ylabel("frecuencia observada de compra")
    error = float((g["observada"] - g["predicha"]).abs().mean())
    ax.set_title(f"M1 · Calibracion por decil → desvio absoluto medio {error:.4f}")
    ax.legend(loc="upper left")
    return ax


def plot_shap_global(shap_global: pd.DataFrame, ax=None):
    """Driver ranking by mean |SHAP|, coloured by the direction of the effect."""
    ax = ax or plt.gca()
    d = shap_global.iloc[::-1]
    colores = [PALETTE["primary"] if e >= 0 else PALETTE["accent"] for e in d["efecto_medio"]]
    ax.barh(d["feature"], d["importancia_shap"], color=colores)
    for i, v in enumerate(d["importancia_shap"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("|SHAP| medio")
    ax.set_title("M1 · Drivers globales (SHAP) → el historial manda, no la demografia")
    ax.set_xlim(0, float(d["importancia_shap"].max()) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    return ax


def plot_elasticities(E: pd.DataFrame, ax=None):
    """The 5x5 elasticity matrix; the diagonal is what pricing depends on."""
    ax = ax or plt.gca()
    M = E.to_numpy(dtype=float)
    lim = float(np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(E.shape[1]), [c.replace("precio marca ", "p") for c in E.columns])
    ax.set_yticks(range(E.shape[0]), [i.replace("share marca ", "share ") for i in E.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            marco = i == j
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9 if not marco else 10,
                    fontweight="bold" if marco else "normal",
                    color="white" if abs(M[i, j]) > lim * 0.6 else "black")
    ax.set_title("M2 · Elasticidades propias (diagonal) y cruzadas → insumo de P3")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.045, label="% de share por % de precio")
    return ax


def plot_value_capture(captura: pd.DataFrame, ax=None):
    """Share of real test revenue captured by each targeting policy."""
    ax = ax or plt.gca()
    pivote = captura.pivot(index="presupuesto", columns="politica", values="pct_del_ingreso")
    orden = sorted(pivote.index, key=lambda s: int(s.split()[-1].rstrip("%")))
    pivote = pivote.loc[orden]
    politicas = ["azar (proporcional)", "politica historica", "solo propension (M1)",
                 "valor esperado (M1xM2xM3)"]
    colores = [PALETTE["muted"], "#A0AEC0", PALETTE["accent"], PALETTE["primary"]]
    x = np.arange(len(pivote))
    w = 0.2
    for i, (pol, color) in enumerate(zip(politicas, colores)):
        if pol not in pivote.columns:
            continue
        pos = x + (i - 1.5) * w
        ax.bar(pos, pivote[pol], width=w, color=color, label=pol)
        for xi, v in zip(pos, pivote[pol]):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, pivote.index)
    ax.set_xlabel("presupuesto de contacto")
    ax.set_ylabel("% del ingreso real de test capturado")
    ax.set_title("Captura de valor → encadenar M2 y M3 paga donde el presupuesto aprieta")
    ax.legend(loc="upper left", fontsize=7)
    return ax


# --------------------------------------------------------------------------
# Fase 5 -- the prescriptive layer: P1, P2, P3
# --------------------------------------------------------------------------


def plot_propensity_curves(curvas: dict, ax=None):
    """Margin response to moving one brand's price, one line per brand.

    A brand whose own-price elasticity Fase 4 could not identify is drawn
    dashed and grey: the curve exists, but it describes observational data, not
    a lever anyone should pull.
    """
    ax = ax or plt.gca()
    colores = [PALETTE["primary"], PALETTE["accent"], PALETTE["good"], "#6B46C1", "#D53F8C"]
    for i, (marca, curva) in enumerate(sorted(curvas.items())):
        identificada = curva.attrs.get("identificada", True)
        ax.plot(
            curva["cambio_precio_pct"],
            curva["delta_margen_total"],
            marker="o",
            markersize=4,
            linewidth=2 if identificada else 1.4,
            linestyle="-" if identificada else "--",
            color=colores[i % len(colores)] if identificada else PALETTE["muted"],
            label="marca {}{}".format(marca, "" if identificada else " (no identificada)"),
        )
    ax.axhline(0, color=PALETTE["muted"], linewidth=1)
    ax.axvline(0, color=PALETTE["muted"], linestyle=":", linewidth=1)
    ax.set_xlabel("cambio de precio (%)")
    ax.set_ylabel("Δ margen esperado del horizonte")
    ax.set_title("Curvas de propension frente al precio → la pregunta comercial, respondida")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_p1_frontier(frontera: pd.DataFrame, ax=None):
    """Net objective against budget, with the point where contacting stops paying."""
    ax = ax or plt.gca()
    ax.plot(frontera["contactos_usados"], frontera["objetivo_neto"], marker="o",
            color=PALETTE["primary"], linewidth=2, label="objetivo neto")
    tope = frontera.attrs.get("contactos_utiles")
    if tope:
        ax.axvline(tope, color=PALETTE["bad"], linestyle="--", linewidth=1.5,
                   label=f"saturacion: {tope} contactos ({frontera.attrs['pct_visitas_utiles']}%)")
    for _, fila in frontera.iterrows():
        ax.annotate(
            "{:.0f}%".format(fila["presupuesto_pct_visitas"]),
            (fila["contactos_usados"], fila["objetivo_neto"]),
            textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7,
        )
    ax.set_xlabel("contactos usados")
    ax.set_ylabel("objetivo neto (unidades de margen)")
    ax.set_title("P1 · Frontera de presupuesto → contactar mas no es contactar mejor")
    ax.legend(loc="lower right")
    return ax


def plot_promo_reallocation(p2: pd.DataFrame, ax=None):
    """Promotion days per brand, observed against optimized, with value per day."""
    ax = ax or plt.gca()
    x = np.arange(len(p2))
    w = 0.38
    ax.bar(x - w / 2, p2["dias_observados"], width=w, color=PALETTE["muted"], label="observado")
    colores = [PALETTE["good"] if v > 0 else PALETTE["bad"] for v in p2["valor_medio_por_dia"]]
    ax.bar(x + w / 2, p2["dias_optimos"], width=w, color=colores, label="optimo (P2)")
    for i, (_, fila) in enumerate(p2.iterrows()):
        ax.text(i + w / 2, fila["dias_optimos"] + 1.5,
                "{:+.2f}/dia".format(fila["valor_medio_por_dia"]), ha="center", fontsize=8)
    ax.set_xticks(x, [f"marca {int(j)}" for j in p2.index])
    ax.set_ylabel("dias en promocion")
    ax.set_title("P2 · Reasignar el mismo presupuesto promocional ({:+.2f} de margen)".format(
        p2.attrs.get("ganancia", float("nan"))))
    ax.legend(loc="upper left")
    return ax


def plot_price_response(respuesta: pd.DataFrame, ax=None):
    """Margin response per brand over its candidate price points."""
    ax = ax or plt.gca()
    colores = [PALETTE["primary"], PALETTE["accent"], PALETTE["muted"], "#6B46C1", "#D53F8C"]
    for i, marca in enumerate(sorted(respuesta["marca"].unique())):
        sub = respuesta[respuesta["marca"] == marca].sort_values("precio")
        ax.plot(sub["precio"], sub["delta_vs_observado"], marker="o", markersize=4,
                linewidth=1.8, color=colores[i % len(colores)], label=f"marca {int(marca)}")
        obs = sub[sub["es_observado"]]
        if len(obs):
            ax.scatter(obs["precio"], obs["delta_vs_observado"], s=120, marker="s",
                       facecolor="none", edgecolor=colores[i % len(colores)], linewidth=1.6)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1)
    ax.set_xlabel("punto de la escalera de precios")
    ax.set_ylabel("Δ margen vs. precio observado")
    ax.set_title("P3 · Respuesta de margen por punto de precio (□ = precio observado)")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_value_waterfall(base: float, componentes: dict, ax=None):
    """Base margin plus each lever, in the order the levers get executed."""
    ax = ax or plt.gca()
    etiquetas = ["margen base"] + list(componentes) + ["total proyectado"]
    valores = [base] + list(componentes.values())
    total = base + sum(componentes.values())

    acumulado = base
    ax.bar(0, base, color=PALETTE["muted"])
    ax.text(0, base / 2, f"{base:.0f}", ha="center", va="center", fontsize=9, color="white")
    for i, (nombre, v) in enumerate(componentes.items(), start=1):
        color = PALETTE["good"] if v >= 0 else PALETTE["bad"]
        ax.bar(i, v, bottom=acumulado, color=color)
        ax.text(i, acumulado + v / 2, f"{v:+.0f}", ha="center", va="center", fontsize=9,
                color="white")
        ax.plot([i - 0.4, i + 0.4], [acumulado + v, acumulado + v], color=PALETTE["muted"],
                linewidth=0.8, linestyle=":")
        acumulado += v
    ax.bar(len(componentes) + 1, total, color=PALETTE["primary"])
    ax.text(len(componentes) + 1, total / 2, f"{total:.0f}", ha="center", va="center",
            fontsize=9, color="white")

    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("unidades de margen en el horizonte de test")
    ax.set_title("Captura de valor proyectada → {:+.1f}% sobre el margen base".format(
        100 * sum(componentes.values()) / base))
    return ax
