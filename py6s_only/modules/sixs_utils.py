"""Small Py6S utilities used by the ljn oc_id correction entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from Py6S import AtmosCorr, AtmosProfile, Geometry, GroundReflectance, PredefinedWavelengths, SixS, Wavelength

from modules.aeronet import LjnAeronetRecord
from modules.modis import ModisSample


MODIS_BAND_NUMBER = {
    "412": 8,
    "443": 9,
    "469": 3,
    "488": 10,
    "531": 11,
    "547": 12,
    "555": 4,
    "645": 1,
    "667": 13,
    "678": 14,
    "748": 15,
    "859": 2,
    "869": 16,
    "1240": 5,
    "1640": 6,
    "2130": 7,
}


def ozone_cm_atm(record: LjnAeronetRecord) -> Optional[float]:
    if record.ozone_dobson is None:
        return None
    return record.ozone_dobson / 1000.0


def configure_sixs(
    sixs_path: Path,
    sample: ModisSample,
    record: LjnAeronetRecord,
    wavelength_um: float,
    toa_reflectance: Optional[float] = None,
    band: Optional[str] = None,
    wavelength_mode: str = "modis-rsr",
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
        date = sample.dt_utc.strftime("%d/%m/%Y")
        s.atmos_profile = AtmosProfile.FromLatitudeAndDate(record.lat, date)

    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.0)
    if wavelength_mode == "modis-rsr" and band in MODIS_BAND_NUMBER:
        platform = "TERRA" if sample.raw.get("l1b_file", "").upper().startswith("T") else "AQUA"
        response = getattr(PredefinedWavelengths, f"ACCURATE_MODIS_{platform}_{MODIS_BAND_NUMBER[band]}")
        s.wavelength = Wavelength(response)
    else:
        s.wavelength = Wavelength(wavelength_um)
    if toa_reflectance is not None:
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(toa_reflectance)
    return s


def wavelength_configuration_name(sample: ModisSample, band: str, wavelength_mode: str) -> str:
    if wavelength_mode == "modis-rsr" and band in MODIS_BAND_NUMBER:
        platform = "Terra" if sample.raw.get("l1b_file", "").upper().startswith("T") else "Aqua"
        return f"MODIS_{platform}_Band_{MODIS_BAND_NUMBER[band]}_RSR"
    return f"Point_{float(band) / 1000.0:.6f}_um"


def configure_ocean_surface(s: SixS, record: LjnAeronetRecord, toa_reflectance: float) -> None:
    '''配置 6SV 内置 Cox-Munk 粗糙海面 BRDF 模型'''
    wind_speed = record.wind_speed_ms if record.wind_speed_ms is not None else 2.0
    pigment = record.chlorophyll_a_mg_m3 if record.chlorophyll_a_mg_m3 is not None else 0.2
    s.ground_reflectance = GroundReflectance.HomogeneousOcean(
        max(0.0, wind_speed),
        0.0,
        -1.0,
        max(0.0, pigment),
    )
    s.atmos_corr = AtmosCorr.AtmosCorrBRDFFromReflectance(toa_reflectance)


def atmosphere_profile_name(record: LjnAeronetRecord, sample: ModisSample) -> str:
    ozone = ozone_cm_atm(record)
    if record.water_cm is not None and ozone is not None:
        return f"LJN_UserWaterAndOzone(water_gcm2={record.water_cm:.3f},ozone_cmatm={ozone:.4f})"
    return f"FromLatitudeAndDate(latitude={record.lat:.4f},date={sample.dt_utc.date().isoformat()})"


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
