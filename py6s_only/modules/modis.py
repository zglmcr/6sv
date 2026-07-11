"""MODIS row parsing helpers for ljn oc_id correction."""

from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import math

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


@dataclass
class ModisSupplementRecord:
    oc_id: str
    abs_time_diff_minutes: Optional[float]
    raw: Dict[str, str]


def glint_angle_deg(sample: ModisSample) -> float:
    solar_z = math.radians(sample.solar_z)
    sensor_z = math.radians(sample.sensor_z)
    relative_azimuth = math.radians(sample.relative_azimuth)
    cosine = (
        math.cos(solar_z) * math.cos(sensor_z)
        - math.sin(solar_z) * math.sin(sensor_z) * math.cos(relative_azimuth)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def default_modis_supplement_csv_path(modis_csv: Path) -> Path:
    return modis_csv.with_name("modis_l1b_result_v2_rayleigh_rhos_1.csv")


def oc_id_in_range(oc_id: str, oc_id_range: Optional[tuple[int, int]]) -> bool:
    if oc_id_range is None:
        return True
    try:
        value = int(oc_id)
    except ValueError:
        return False
    return oc_id_range[0] <= value <= oc_id_range[1]


def read_modis_supplement_index(
    path: Path,
    oc_id_range: Optional[tuple[int, int]] = None,
) -> Dict[str, ModisSupplementRecord]:
    index: Dict[str, ModisSupplementRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        required = {"oc_id", "abs_time_diff_minutes"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"MODIS supplement CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            oc_id = (row.get("oc_id") or "").strip()
            if not oc_id or not oc_id_in_range(oc_id, oc_id_range):
                continue
            time_diff = parse_float(row.get("abs_time_diff_minutes"))
            if time_diff is None:
                continue
            previous = index.get(oc_id)
            if previous is None or (
                previous.abs_time_diff_minutes is not None
                and time_diff < previous.abs_time_diff_minutes
            ):
                index[oc_id] = ModisSupplementRecord(
                    oc_id=oc_id,
                    abs_time_diff_minutes=time_diff,
                    raw=row,
                )
    return index


def passes_modis_match_quality_control(
    oc_id: str,
    supplement: Dict[str, ModisSupplementRecord],
    max_abs_time_diff_minutes: float = 60.0,
) -> bool:
    record = supplement.get(oc_id)
    if record is None or record.abs_time_diff_minutes is None:
        return False
    return record.abs_time_diff_minutes < max_abs_time_diff_minutes


def build_modis_sample(
    row: Dict[str, str],
    bands: Optional[set[str]],
    reflectance_input: str = "l1b-rho-cos",
) -> Optional[ModisSample]:
    lat = parse_float(row.get("pixel_latitude"))
    lon = parse_float(row.get("pixel_longitude"))
    solar_z = parse_float(row.get("solar_zenith_deg"))
    sensor_z = parse_float(row.get("sensor_zenith_deg"))
    relaz = parse_float(row.get("relative_azimuth_abs_deg"))
    if None in (lat, lon, solar_z, sensor_z, relaz):
        return None
    solar_zenith_cosine = math.cos(math.radians(solar_z))
    if solar_zenith_cosine <= 0:
        return None
    toa_by_band: Dict[str, float] = {}
    for key, value in row.items():
        match = re.fullmatch(r"reflectance_(\d+)", key)
        if not match:
            continue
        band = match.group(1)
        if bands is not None and band not in bands:
            continue
        reflectance = parse_float(value)
        if reflectance is not None:
            toa_by_band[band] = (
                reflectance / solar_zenith_cosine
                if reflectance_input == "l1b-rho-cos"
                else reflectance
            )
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
