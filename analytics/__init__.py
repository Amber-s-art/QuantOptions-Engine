"""Higher-level analytics for QuantOptions-Engine.

The analytics package builds on the project's core pricing and simulation
modules. It deliberately does not fetch market data or duplicate model code.
"""

from .arbitrage import MispricingFilters, scan_mispricing
from .greeks_matrix import build_greeks_matrix
from .risk_engine import (
    RiskSummary,
    calculate_var_cvar,
    calculate_portfolio_pnl,
    run_stress_test,
)

__all__ = [
    "MispricingFilters",
    "RiskSummary",
    "build_greeks_matrix",
    "calculate_portfolio_pnl",
    "calculate_var_cvar",
    "run_stress_test",
    "scan_mispricing",
]
