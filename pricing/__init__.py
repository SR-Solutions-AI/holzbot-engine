from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jobs import PricingJobResult

__all__ = ["run_pricing_for_run", "PricingJobResult"]


def __getattr__(name: str):
    if name in {"run_pricing_for_run", "PricingJobResult"}:
        from .jobs import PricingJobResult, run_pricing_for_run

        return {
            "run_pricing_for_run": run_pricing_for_run,
            "PricingJobResult": PricingJobResult,
        }[name]
    raise AttributeError(f"module 'pricing' has no attribute {name!r}")