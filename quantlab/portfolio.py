"""Portfolio state: cash and shares, updated only through fills."""

from __future__ import annotations

from dataclasses import dataclass

from quantlab.execution import Fill, Side


class AccountingError(RuntimeError):
    """Raised when a fill would violate a portfolio constraint (e.g. negative cash)."""


@dataclass
class Portfolio:
    cash: float
    shares: float = 0.0

    # Floating-point noise below this fraction of the trade value is treated as zero.
    _REL_TOL = 1e-9

    def value(self, price: float) -> float:
        return self.cash + self.shares * price

    def apply(self, fill: Fill) -> tuple[float, float, float, float]:
        """Apply a fill; return ``(cash_before, cash_after, shares_before, shares_after)``."""
        cash_before, shares_before = self.cash, self.shares
        tol = self._REL_TOL * max(1.0, fill.gross_notional)
        if fill.side is Side.BUY:
            cost = fill.gross_notional + fill.fee
            if cost > self.cash + tol:
                raise AccountingError(
                    f"buy of {fill.quantity:.6f} shares costs {cost:.2f} but only {self.cash:.2f} cash is available"
                )
            self.cash -= cost
            self.shares += fill.quantity
        else:
            if fill.quantity > self.shares + self._REL_TOL * max(1.0, self.shares):
                raise AccountingError(
                    f"sell of {fill.quantity:.6f} shares exceeds the {self.shares:.6f} held (no short selling)"
                )
            self.cash += fill.gross_notional - fill.fee
            self.shares -= fill.quantity
        if abs(self.cash) < tol:
            self.cash = 0.0
        if abs(self.shares) < self._REL_TOL:
            self.shares = 0.0
        return cash_before, self.cash, shares_before, self.shares
