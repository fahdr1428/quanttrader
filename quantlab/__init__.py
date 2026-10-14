"""Quant Lab Pro — quantitative research and backtesting core.

The ``quantlab`` package contains everything that computes something: data
access, validation, strategies, execution, portfolio accounting, metrics and
research tooling. It deliberately has no dependency on Streamlit, so every
calculation can be tested, scripted and reproduced without the user interface.
"""

from quantlab.config import APP_VERSION, METHODOLOGY_VERSION

__all__ = ["APP_VERSION", "METHODOLOGY_VERSION"]
