"""Transparent $ cost model for Spark and DuckDB runs.

Cost is computed from resource-hours so both engines use the SAME function:
    cost = vcpu_hours * $/vCPU-h + ram_gb_hours * $/GB-h
Reference prices approximate AWS general-purpose on-demand vs spot (us-east-1,
m6i family, decomposed into per-vCPU and per-GB rates). These are documented
inputs, not authoritative quotes — report a sensitivity range in the paper.
"""
from typing import Tuple

# $/vCPU-hour and $/GB-RAM-hour. m6i.xlarge ~= $0.192/h for 4 vCPU + 16 GB.
# Decomposed ~70% compute / 30% memory: vCPU ~0.0336/h, GB ~0.0036/h.
PRICES = {
    "ondemand": {"vcpu_h": 0.0336, "gb_h": 0.0036},
    "spot":     {"vcpu_h": 0.0118, "gb_h": 0.0013},  # ~35% of on-demand
}

# Multipliers for the sensitivity band reported alongside the point estimate.
SENSITIVITY = (0.6, 1.0, 1.6)


def run_cost_usd(vcpu_hours: float, ram_gb_hours: float,
                 price: str = "ondemand") -> float:
    p = PRICES[price]
    return round(vcpu_hours * p["vcpu_h"] + ram_gb_hours * p["gb_h"], 4)


def cost_sensitivity(vcpu_hours: float, ram_gb_hours: float,
                     price: str = "ondemand") -> Tuple[float, float, float]:
    base = run_cost_usd(vcpu_hours, ram_gb_hours, price)
    lo, mid, hi = SENSITIVITY
    return (round(base * lo, 4), round(base * mid, 4), round(base * hi, 4))
