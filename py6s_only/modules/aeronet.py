"""AERONET ljn CSV reading and record conversion helpers."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

from modules.common import nearest_wavelength_value, parse_datetime, parse_float


@dataclass
class LjnAeronetRecord:
    oc_id: str
    site: str
    dt_utc: datetime
    lat: float
    lon: float
    elevation_m: float
    aod_by_um: Dict[float, float]
    ordinary_aod_by_um: Dict[float, float]
    inv_total_aod_by_um: Dict[float, float]
    inv_fine_aod_by_um: Dict[float, float]
    inv_coarse_aod_by_um: Dict[float, float]
    size_distribution: Dict[float, float]
    refr_real_by_um: Dict[float, float]
    refr_imag_by_um: Dict[float, float]
    ssa_by_um: Dict[float, float]
    absorption_aod_by_um: Dict[float, float]
    oc_products_by_um: Dict[float, Dict[str, float]]
    water_cm: Optional[float]
    ozone_dobson: Optional[float]
    no2_dobson: Optional[float]
    wind_speed_ms: Optional[float]
    chlorophyll_a_mg_m3: Optional[float]
    oc_quality_level: str
    inversion_quality_level: str
    inv_time_diff_minutes: Optional[float]

    def angstrom(self) -> Optional[float]:
        pairs = sorted((wl, aod) for wl, aod in self.aod_by_um.items() if wl > 0 and aod > 0)
        if len(pairs) < 2:
            return None
        preferred = [(0.440, 0.870), (0.443, 0.865), (0.412, 0.865)]
        for lo_target, hi_target in preferred:
            lo = nearest_wavelength_value(self.aod_by_um, lo_target, max_delta=0.015)
            hi = nearest_wavelength_value(self.aod_by_um, hi_target, max_delta=0.020)
            if lo and hi and lo[1] > 0 and hi[1] > 0 and lo[0] != hi[0]:
                return -math.log(lo[1] / hi[1]) / math.log(lo[0] / hi[0])
        lo_wl, lo_aod = pairs[0]
        hi_wl, hi_aod = pairs[-1]
        if lo_wl == hi_wl:
            return None
        return -math.log(lo_aod / hi_aod) / math.log(lo_wl / hi_wl)

    def fine_coarse_at_nearest_550(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        keys = sorted(set(self.inv_fine_aod_by_um) & set(self.inv_coarse_aod_by_um), key=lambda wl: abs(wl - 0.55))
        for wl in keys:
            fine = self.inv_fine_aod_by_um.get(wl)
            coarse = self.inv_coarse_aod_by_um.get(wl)
            if fine is None or coarse is None:
                continue
            total = fine + coarse
            if total > 0:
                return fine / total, fine, coarse
        return None, None, None

    def has_sunphotometer_inputs(self) -> bool:
        return (
            len(self.size_distribution) >= 10
            and len(self.refr_real_by_um) >= 2
            and len(self.refr_imag_by_um) >= 2
        )

    def angstrom_or_default(self, default: float = 1.3) -> float:
        return self.angstrom() or default


def parse_lwn_datetime(row: Dict[str, str]) -> datetime:
    value = (row.get("lwn_datetime") or "").strip()
    if value:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return parse_datetime(row.get("Date(dd-mm-yyyy)", ""), row.get("Time(hh:mm:ss)", ""))


def passes_ljn_quality_control(record: LjnAeronetRecord, max_inv_time_diff_minutes: float = 180.0) -> bool:
    if record.inv_time_diff_minutes is None:
        return False
    return abs(record.inv_time_diff_minutes) < max_inv_time_diff_minutes


def extract_wavelength_columns(row: Dict[str, str], pattern: str) -> Dict[float, float]:
    out: Dict[float, float] = {}
    regex = re.compile(pattern)
    for key, value in row.items():
        match = regex.fullmatch(key)
        if not match:
            continue
        number = parse_float(value)
        if number is None:
            continue
        out[float(match.group(1)) / 1000.0] = number
    return out


def extract_size_distribution(row: Dict[str, str]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    regex = re.compile(r"inv_(\d+\.\d+)")
    for key, value in row.items():
        match = regex.fullmatch(key)
        if not match:
            continue
        radius = float(match.group(1))
        number = parse_float(value)
        if number is not None and number >= 0:
            out[radius] = number
    return out


def extract_oc_products(row: Dict[str, str]) -> Dict[float, Dict[str, float]]:
    products = {
        "Rho": "oc_rho",
        "Lw": "oc_lw",
        "LwQ": "oc_lwq",
        "Lwn": "oc_lwn",
        "Lwn_f/Q": "oc_lwn_fq",
        "Lwn_IOP": "oc_lwn_iop",
    }
    out: Dict[float, Dict[str, float]] = {}
    for source_name, output_name in products.items():
        regex = re.compile(re.escape(source_name) + r"\[(\d+)nm\]")
        for key, value in row.items():
            match = regex.fullmatch(key)
            if not match:
                continue
            number = parse_float(value)
            if number is None:
                continue
            wavelength_um = float(match.group(1)) / 1000.0
            out.setdefault(wavelength_um, {})[output_name] = number
    return out


def read_aeronet_index(
    path: Path,
    wanted_oc_ids: Optional[Union[Set[str], Tuple[int, int]]] = None,
) -> Dict[str, LjnAeronetRecord]:
    index: Dict[str, LjnAeronetRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oc_id = (row.get("oc_id") or "").strip()
            if not oc_id:
                continue
            # if wanted_oc_ids is not None and oc_id not in wanted_oc_ids:
            #     continue
            if wanted_oc_ids is not None:
                if isinstance(wanted_oc_ids, tuple):
                    # 区间模式：min <= 当前ID <= max
                    oc_min, oc_max = wanted_oc_ids
                    if not (oc_min <= int(oc_id) <= oc_max):
                        continue
                else:
                    # 原有集合匹配模式（兼容旧逻辑）
                    if oc_id not in wanted_oc_ids:
                        continue
            '''
            ordinary_aod（Direct Sun AOT）：来自太阳直射辐射衰减，是直接观测量，信噪比高、模型假设少，是 AERONET 公认的基准真值；
            inv_total（inv_AOD_Extinction-Total）：是天空散射反演迭代后，由气溶胶粒径 / 折射率模型正向计算出的理论消光厚度，依赖气溶胶双模态、垂直廓线等先验假设，存在模型偏差；
            当两者同波长共存时，直射观测更可靠，优先采信直射值。
            为填充波长缺口，保证波长覆盖完整，先用inv_total打底，再用ordinary_aod覆盖
            '''
            ordinary_aod = extract_wavelength_columns(row, r"Aerosol_Optical_Depth\[(\d+)nm\]") #
            inv_total = extract_wavelength_columns(row, r"inv_AOD_Extinction-Total\[(\d+)nm\]") # 仅四个波段
            aod_by_um = dict(inv_total)
            aod_by_um.update(ordinary_aod) # 普通 AOD 覆盖反演总 AOD
            lat = parse_float(row.get("Site_Latitude(Degrees)"))
            lon = parse_float(row.get("Site_Longitude(Degrees)"))
            if lat is None or lon is None or not aod_by_um:
                continue
            index[oc_id] = LjnAeronetRecord(
                oc_id=oc_id,
                site=(row.get("AERONET_Site_Name") or row.get("site") or "").strip(),
                dt_utc=parse_lwn_datetime(row),
                lat=lat,
                lon=lon,
                elevation_m=parse_float(row.get("Site_Elevation(m)")) or 0.0,
                aod_by_um=aod_by_um,
                ordinary_aod_by_um=ordinary_aod,
                inv_total_aod_by_um=inv_total,
                inv_fine_aod_by_um=extract_wavelength_columns(row, r"inv_AOD_Extinction-Fine\[(\d+)nm\]"),
                inv_coarse_aod_by_um=extract_wavelength_columns(row, r"inv_AOD_Extinction-Coarse\[(\d+)nm\]"),
                size_distribution=extract_size_distribution(row),
                refr_real_by_um=extract_wavelength_columns(row, r"inv_Refractive_Index-Real_Part\[(\d+)nm\]"),
                refr_imag_by_um=extract_wavelength_columns(row, r"inv_Refractive_Index-Imaginary_Part\[(\d+)nm\]"),
                ssa_by_um=extract_wavelength_columns(row, r"inv_Single_Scattering_Albedo\[(\d+)nm\]"),
                absorption_aod_by_um=extract_wavelength_columns(row, r"inv_Absorption_AOD\[(\d+)nm\]"),
                oc_products_by_um=extract_oc_products(row),
                water_cm=parse_float(row.get("Total_Precipitable_Water(cm)")),
                ozone_dobson=parse_float(row.get("Total_Ozone(Du)")),
                no2_dobson=parse_float(row.get("Total_NO2(DU)")),
                wind_speed_ms=parse_float(row.get("Wind_Speed(m/s)")),
                chlorophyll_a_mg_m3=parse_float(row.get("Chlorophyll-a")),
                oc_quality_level=(row.get("Data_Quality_Level") or "").strip(),
                inversion_quality_level=(row.get("inv_Inversion_Data_Quality_Level") or "").strip(),
                inv_time_diff_minutes=parse_float(row.get("inv_time_diff_minutes")),
            )
            # if wanted_oc_ids is not None and len(index) == len(wanted_oc_ids):
            #     break
    return index
