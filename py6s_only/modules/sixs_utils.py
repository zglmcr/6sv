"""Small Py6S utilities used by the ljn oc_id correction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from Py6S import AtmosProfile, Geometry, GroundReflectance, SixS, Wavelength

from modules.aeronet import LjnAeronetRecord
from modules.modis import ModisSample


def ozone_cm_atm(record: LjnAeronetRecord) -> Optional[float]:
    if record.ozone_dobson is None:
        return None
    return record.ozone_dobson / 1000.0


def configure_sixs(
    sixs_path: Path,
    sample: ModisSample,
    record: LjnAeronetRecord,
    wavelength_um: float,
) -> SixS:
    s = SixS(path=str(sixs_path))
    s.geometry = Geometry.User()
    s.geometry.solar_z = sample.solar_z
    s.geometry.solar_a = 0.0
    s.geometry.view_z = sample.sensor_z
    s.geometry.view_a = sample.relative_azimuth
    s.geometry.month = sample.dt_utc.month
    s.geometry.day = sample.dt_utc.day

    if record.elevation_m > 0:
        s.altitudes.set_target_custom_altitude(record.elevation_m / 1000.0)
    else:
        s.altitudes.set_target_sea_level()
    s.altitudes.set_sensor_satellite_level()

    ozone = ozone_cm_atm(record)
    if record.water_cm is not None and ozone is not None:
        s.atmos_profile = AtmosProfile.UserWaterAndOzone(record.water_cm, ozone)
    else:
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)

    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.0)
    s.wavelength = Wavelength(wavelength_um)
    return s


def solve_lambertian_surface_reflectance(rho_toa: float, sca: Dict[str, float | str]) -> float:
    rho_path = float(sca["rho_path_total"])
    trans = float(sca["trans_total_scattering"])
    spherical_albedo = float(sca["spherical_albedo"])
    y = rho_toa - rho_path
    denom = trans + spherical_albedo * y
    if abs(denom) < 1e-12:
        return float("nan")
    return y / denom


def format_value(value: float | str) -> str:
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)
