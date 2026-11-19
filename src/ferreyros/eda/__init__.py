"""EDA modules, one per block of the alignment matrix.

* :mod:`objective`   -- block A: parameters of the objective function (pi, q, lambda, m)
* :mod:`decisions`   -- block B: decision variables (y_jtk, z_jt, x_ijt)
* :mod:`constraints` -- block C: bounds for the constraints (R3-R8)
* :mod:`segments`    -- block D: demographics and the segmentation model M0
* :mod:`dynamics`    -- temporal dynamics: seasonality and trend (Fase 1)
* :mod:`anomalies`   -- coherence, extreme quantities, atypical clients (Fase 1)
"""

from . import anomalies, constraints, decisions, dynamics, objective, segments

__all__ = [
    "objective",
    "decisions",
    "constraints",
    "segments",
    "dynamics",
    "anomalies",
]
