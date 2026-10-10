"""Execution model: turning an order into a fill.

Conventions (default daily model)
---------------------------------
* **Timing** — an order generated from the close of session ``t`` fills at the
  *open* of session ``t + 1``, the next row in the data. If the market gapped
  overnight, the fill reflects the gap; there is no guarantee of trading near
  the previous close.
* **Reference price** — the session's opening price.
* **Slippage** — the fill is worse than the reference by ``slippage_rate``:
  buys fill at ``open * (1 + s)``, sells at ``open * (1 - s)``. The slippage
  *cost* reported is ``quantity * |execution price - reference price|``.
* **Fees** — ``fee_rate * quantity * execution price``, i.e. charged on the
  actual traded notional, for buys and sells alike.
* **Sizing** — all-in: a buy spends the available cash such that
  ``notional + fee <= cash``. With whole shares, the quantity is rounded down
  and the remainder stays in cash. A sell always closes the entire position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Fill:
    side: Side
    quantity: float
    reference_price: float
    execution_price: float
    gross_notional: float
    fee: float
    slippage_cost: float

    @property
    def cash_flow(self) -> float:
        """Signed change in cash: negative for a buy, positive for a sell."""
        if self.side is Side.BUY:
            return -(self.gross_notional + self.fee)
        return self.gross_notional - self.fee


class ExecutionModel:
    def __init__(self, fee_rate: float, slippage_rate: float, fractional_shares: bool = True):
        if not (0 <= fee_rate < 1) or not (0 <= slippage_rate < 1):
            raise ValueError("fee and slippage rates must be in [0, 1)")
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)
        self.fractional_shares = fractional_shares

    def execution_price(self, side: Side, reference_price: float) -> float:
        if not (reference_price > 0 and math.isfinite(reference_price)):
            raise ValueError(f"invalid reference price {reference_price!r}")
        factor = 1 + self.slippage_rate if side is Side.BUY else 1 - self.slippage_rate
        return reference_price * factor

    def buy_quantity(self, cash: float, reference_price: float) -> float:
        """Largest quantity whose notional plus fee fits in ``cash``."""
        if cash <= 0:
            return 0.0
        price = self.execution_price(Side.BUY, reference_price)
        quantity = cash / (price * (1 + self.fee_rate))
        if not self.fractional_shares:
            # Small epsilon guards against 99.9999999 being floored to 99.
            quantity = float(math.floor(quantity + 1e-9))
            while quantity > 0 and quantity * price * (1 + self.fee_rate) > cash:
                quantity -= 1.0
        return max(quantity, 0.0)

    def fill(self, side: Side, quantity: float, reference_price: float) -> Fill:
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        price = self.execution_price(side, reference_price)
        notional = quantity * price
        return Fill(
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            execution_price=price,
            gross_notional=notional,
            fee=self.fee_rate * notional,
            slippage_cost=quantity * abs(price - reference_price),
        )
