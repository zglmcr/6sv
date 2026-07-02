"""MODIS row parsing helpers for ljn oc_id correction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from modules.common import parse_datetime, parse_float


@dataclass
class ModisSample:
    sample_id: str
    dt_utc: datetime
    lat: float
    lon: float
    solar_z: float
    sensor_z: float
    relative_azimuth: float
    toa_by_band: Dict[str, float]
    raw: Dict[str, str]


def build_modis_sample(row: Dict[str, str], bands: Optional[set[str]]) -> Optional[ModisSample]:
    lat = parse_float(row.get("pixel_latitude"))
    lon = parse_float(row.get("pixel_longitude"))
    solar_z = parse_float(row.get("solar_zenith_deg"))
    sensor_z = parse_float(row.get("sensor_zenith_deg"))
    relaz = parse_float(row.get("relative_azimuth_abs_deg"))
    if None in (lat, lon, solar_z, sensor_z, relaz):
        return None
    toa_by_band: Dict[str, float] = {}
    for key, value in row.items():
        match = re.fullmatch(r"reflectance_(\d+)", key)
        if not match:
            continue
        band = match.group(1)
        if bands is not None and band not in bands:
            continue
        rho = parse_float(value)
        if rho is not None:
            toa_by_band[band] = rho
    if not toa_by_band:
        return None
    return ModisSample(
        sample_id=(row.get("oc_id") or "").strip(),
        dt_utc=parse_datetime(row.get("date", ""), row.get("Time_UTC", "")),
        lat=lat,
        lon=lon,
        solar_z=solar_z,
        sensor_z=sensor_z,
        relative_azimuth=relaz,
        toa_by_band=toa_by_band,
        raw=row,
    )
