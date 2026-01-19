from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PricingMode(Protocol):
    """
    A pricing mode can customize how coefficients are loaded and/or how pricing is computed.
    Start simple (default mode == existing behavior) and extend with additional modes later.
    """

    key: str

    def normalize_frontend_input(self, frontend_data: dict) -> dict:
        ...


@dataclass(frozen=True)
class DefaultPricingMode:
    key: str = "default"

    def normalize_frontend_input(self, frontend_data: dict) -> dict:
        # Default behavior: use frontend_data as-is.
        return frontend_data


def get_pricing_mode(calc_mode: str | None) -> PricingMode:
    mode = (calc_mode or "default").strip() or "default"
    # Future: map other modes here, e.g. "tenantX_v2" -> TenantXv2Mode()
    return DefaultPricingMode(key=mode if mode == "default" else mode)













