"""Common parsing, wavelength matching, and interpolation helpers for ljn workflows."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List, Set


def parse_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or number <= -900:
        return None
    return number


def parse_datetime(date_text: str, time_text: str) -> datetime:
    date_text = date_text.strip()
    time_text = time_text.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromisoformat(f"{date_text}T{time_text}").replace(tzinfo=timezone.utc)

def tuple_to_set(data: Optional[Tuple[int, int]]) -> Optional[Set[int]]:
    if data is not None:
        data_min, data_max = data
        # 生成闭区间所有数字集合
        data_set = set(range(data_min, data_max + 1))
        return data_set
    return None


def nearest_wavelength_value(
    values: Dict[float, float],
    target: float,
    max_delta: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    wl, value = min(values.items(), key=lambda item: abs(item[0] - target))
    if max_delta is not None and abs(wl - target) > max_delta:
        return None
    return wl, value


def nearest_oc_products(
    products: Dict[float, Dict[str, float]],
    target: float,
    max_delta: float = 0.006,
) -> Tuple[Optional[float], Dict[str, float]]:
    if not products:
        return None, {}
    wavelength_um, values = min(products.items(), key=lambda item: abs(item[0] - target))
    if abs(wavelength_um - target) > max_delta:
        return None, {}
    return wavelength_um, values


def interpolate_linear(values: Dict[float, float], target: float) -> float:
    pairs = sorted((wl, val) for wl, val in values.items() if math.isfinite(wl) and math.isfinite(val))
    if not pairs:
        raise ValueError("No values available for interpolation")
    if target <= pairs[0][0]: # 目标波长超出输入范围时不做线性外推，而是使用最近端点值
        return pairs[0][1]
    if target >= pairs[-1][0]:
        return pairs[-1][1]
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x0 <= target <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * ((target - x0) / (x1 - x0)) # 线性插值
    return pairs[-1][1]
