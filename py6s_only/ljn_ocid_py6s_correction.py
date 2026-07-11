#!/usr/bin/env python
"""Run Py6S correction for ljn MODIS/AERONET rows matched by oc_id."""

from __future__ import annotations

import argparse
import os
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple
from itertools import islice

from Py6S import AeroProfile
from Py6S.sixs_exceptions import OutputParsingError

from modules.aeronet import LjnAeronetRecord, passes_ljn_quality_control, read_aeronet_index
from modules.common import interpolate_linear, nearest_oc_products, nearest_wavelength_value, parse_float, tuple_to_set
from modules.modis import (
    ModisSample,
    ModisSupplementRecord,
    build_modis_sample,
    default_modis_supplement_csv_path,
    glint_angle_deg,
    passes_modis_match_quality_control,
    read_modis_supplement_index,
)
from modules.sixs_utils import (
    atmosphere_profile_name,
    configure_ocean_surface,
    configure_sixs,
    format_value,
    ozone_cm_atm,
    solve_lambertian_surface_reflectance,
    wavelength_configuration_name,
)


PY6S_REFRACTIVE_WAVELENGTHS_UM = [
    0.350,
    0.400,
    0.412,
    0.443,
    0.470,
    0.488,
    0.515,
    0.550,
    0.590,
    0.633,
    0.670,
    0.694,
    0.760,
    0.860,
    1.240,
    1.536,
    1.650,
    1.950,
    2.250,
    3.750,
]

OUTPUT_FIELDS = [
    "sample_id",
    "oc_id",
    "aeronet_site",
    "modis_datetime_utc",
    "aeronet_datetime_utc",
    "time_delta_min",
    "distance_km",
    "l1b_file",
    "geo_file",
    "row_0based",
    "col_0based",
    "request_latitude",
    "request_longitude",
    "pixel_latitude",
    "pixel_longitude",
    "distance_to_request_km",
    "height_m",
    "sensor_range_m",
    "sensor_zenith_deg",
    "sensor_azimuth_deg",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "relative_azimuth_deg",
    "relative_azimuth_abs_deg",
    "glint_angle_deg",
    "band",
    "wavelength_um",
    "wavelength_configuration",
    "oc_wavelength_um",
    "oc_rho",
    "oc_lw",
    "oc_lwq",
    "oc_lwn",
    "oc_lwn_fq",
    "oc_lwn_iop",
    "radiance_toa",
    "rho_l1b_reflectance_factor",
    "solar_zenith_cosine",
    "rho_toa",
    "rho_path_total",
    "rho_rayleigh",
    "rho_aerosol",
    "rho_surface_minus_path",
    "rho_toa_minus_atmosphere",
    "rho_surface_lambertian",
    "rho_surface_ocean_brdf",
    "rho_glint_6sv",
    "rho_whitecap_6sv",
    "rho_water_model_6sv",
    "rho_water_leaving_6sv",
    "rho_water_leaving_6sv_swir_corrected",
    "rho_swir_residual_6sv",
    "swir_residual_applied",
    "aod_band",
    "aod_band_6sv",
    "aod550",
    "aot550_6sv_input",
    "angstrom",
    "ssa_aeronet_band",
    "absorption_aod_aeronet_band",
    "single_scattering_albedo_6sv",
    "precipitable_water_cm",
    "ozone_dobson",
    "ozone_cm_atm",
    "no2_dobson",
    "wind_speed_ms",
    "chlorophyll_a_mg_m3",
    "ocean_surface_model",
    "oc_quality_level",
    "inversion_quality_level",
    "atmos_profile",
    "aerosol_model",
    "aerosol_source",
    "trans_rayleigh",
    "trans_aerosol",
    "trans_total_scattering",
    "spherical_albedo",
]

_WORKER_ARGS: Optional[argparse.Namespace] = None
_WORKER_AERONET: Optional[Dict[str, LjnAeronetRecord]] = None
_WORKER_MODIS_SUPPLEMENT: Optional[Dict[str, ModisSupplementRecord]] = None
_WORKER_BANDS: Optional[set[str]] = None


def init_worker(
    args: argparse.Namespace,
    aeronet: Dict[str, LjnAeronetRecord],
    modis_supplement: Dict[str, ModisSupplementRecord],
    bands: Optional[set[str]],
) -> None:
    global _WORKER_ARGS, _WORKER_AERONET, _WORKER_MODIS_SUPPLEMENT, _WORKER_BANDS
    _WORKER_ARGS = args
    _WORKER_AERONET = aeronet
    _WORKER_MODIS_SUPPLEMENT = modis_supplement
    _WORKER_BANDS = bands


def process_row_worker(modis_row: Dict[str, str]) -> Dict[str, object]:
    if _WORKER_ARGS is None or _WORKER_AERONET is None or _WORKER_MODIS_SUPPLEMENT is None:
        raise RuntimeError("worker was not initialized")
    return process_modis_row(
        _WORKER_ARGS,
        _WORKER_AERONET,
        _WORKER_MODIS_SUPPLEMENT,
        _WORKER_BANDS,
        modis_row,
    )


def modis_supplement_csv_path(args: argparse.Namespace) -> Path:
    if args.modis_supplement_csv is not None:
        return args.modis_supplement_csv
    return default_modis_supplement_csv_path(args.modis_csv)


def interpolate_aod(record: LjnAeronetRecord, wavelength_um: float) -> Tuple[float, float]:
    nearest = nearest_wavelength_value(record.aod_by_um, wavelength_um, max_delta=0.0005)
    alpha = record.angstrom_or_default()
    if nearest:
        return nearest[1], alpha
    if record.aod_by_um:
        nearest_550 = nearest_wavelength_value(record.aod_by_um, 0.55, max_delta=0.03)
        if nearest_550:
            return nearest_550[1] * (wavelength_um / nearest_550[0]) ** (-alpha), alpha # 使用离550nm最近波段的AOD，结合alpha，计算当前波段AOD
    nearest_any = nearest_wavelength_value(record.aod_by_um, wavelength_um)
    if nearest_any is None:
        raise ValueError("AOD spectrum is empty")
    return nearest_any[1] * (wavelength_um / nearest_any[0]) ** (-alpha), alpha # 使用任意最近波段的AOD，结合alpha，计算当前波段AOD


def get_global_550_aod(record: Optional[LjnAeronetRecord]) -> float:
    """每条AERONET记录仅执行一次，计算全局唯一550nm AOD基准"""
    if record is None:
        raise ValueError("Aeronet record cannot be None")
        # 或者兜底 return 0.0
    alpha = record.angstrom_or_default()
    aod_dict = record.aod_by_um
    if not aod_dict:
        raise ValueError("AOD spectrum is empty, cannot compute global 550 AOD")

    # 优先找最接近550nm的实测AOD，正向计算550nm基准
    nearest_550 = nearest_wavelength_value(aod_dict, 0.55, max_delta=0.03)
    if nearest_550 is not None:
        wl_ref, tau_ref = nearest_550
        tau_550 = tau_ref * (wl_ref / 0.55) ** alpha
        return max(0.0, tau_550)

    # 无近550波段时，取任意最近波长外推550
    any_wl, any_tau = min(aod_dict.items(), key=lambda x: abs(x[0] - 0.55))
    tau_550 = any_tau * (any_wl / 0.55) ** alpha
    return max(0.0, tau_550)

def user_components_from_fine_fraction(record: LjnAeronetRecord, aod550: float) -> Tuple[Dict[str, float], str]:
    # Set 6S to use a user-defined aerosol profile based on proportions of standard aerosol components.
    fine, _, _ = record.fine_coarse_at_nearest_550() # 找到最接近 550 nm（即 0.55 μm）波长的细模态和粗模态 AOD，并计算细模态占比
    if fine is None:
        alpha = record.angstrom_or_default()
        fine = max(0.0, min(1.0, (alpha - 0.4) / 1.6))
    fine = max(0.0, min(1.0, fine))
    coarse = 1.0 - fine
    alpha = record.angstrom_or_default()
    soot_fraction = 0.02 + 0.10 * min(max(aod550, 0.0), 0.8) / 0.8
    if alpha < 0.8:
        soot_fraction *= 0.5
    soot = fine * soot_fraction
    water = max(0.0, fine - soot)
    oceanic_fraction = 0.55 if alpha < 0.7 else 0.75
    oceanic = coarse * oceanic_fraction
    dust = max(0.0, coarse - oceanic)
    total = dust + water + oceanic + soot
    components = {
        "dust": dust / total,
        "water": water / total,
        "oceanic": oceanic / total,
        "soot": soot / total,
    }
    label = (
        "LJN_FineCoarse_User("
        f"fine={fine:.3f},dust={components['dust']:.3f},water={components['water']:.3f},"
        f"oceanic={components['oceanic']:.3f},soot={components['soot']:.3f})"
    )
    return components, label


def sunphotometer_profile(ljn: LjnAeronetRecord):
    r = [wl for wl, _ in sorted(ljn.size_distribution.items())]
    dvdlogr = [value for _, value in sorted(ljn.size_distribution.items())]
    refr_real = [interpolate_linear(ljn.refr_real_by_um, wl) for wl in PY6S_REFRACTIVE_WAVELENGTHS_UM]
    refr_imag = [interpolate_linear(ljn.refr_imag_by_um, wl) for wl in PY6S_REFRACTIVE_WAVELENGTHS_UM]
    return AeroProfile.SunPhotometerDistribution(r, dvdlogr, refr_real, refr_imag)


def run_scattering(
    sixs_path: Path,
    sample: ModisSample,
    ljn: LjnAeronetRecord,
    wavelength_um: float,
    aerosol_mode: str,
    rho_toa: float,
    band: str,
    wavelength_mode: str,
    global_aod550: float,
) -> Dict[str, float | str]:
    aod_band, alpha = interpolate_aod(ljn, wavelength_um)
    aod550 = global_aod550

    source = "aod_angstrom_fallback"
    model_name = "AOD_Angstrom_User"
    user_components: Optional[Dict[str, float]] = None
    profile = None

    if aerosol_mode in {"auto", "inv"} and ljn.has_sunphotometer_inputs():
        try:
            profile = sunphotometer_profile(ljn)
            source = "inv_sunphotometer"
            model_name = "INV_SunPhotometerDistribution"
        except Exception:
            if aerosol_mode == "inv":
                raise

    if profile is None:
        fine_fraction, _, _ = ljn.fine_coarse_at_nearest_550()
        user_components, model_name = user_components_from_fine_fraction(ljn, aod550)
        source = "inv_fine_coarse_user" if fine_fraction is not None else "aod_angstrom_fallback"

    total = configure_sixs(
        sixs_path,
        sample,
        ljn,
        wavelength_um,
        toa_reflectance=rho_toa,
        band=band,
        wavelength_mode=wavelength_mode,
    )
    if profile is not None:
        total.aero_profile = profile
    else:
        assert user_components is not None
        total.aero_profile = AeroProfile.User(**user_components)
    total.aot550 = max(0.0, aod550)
    total.run()
    first_aod_6sv = total.outputs.optical_depth_total.aerosol
    calibrated_aot550 = max(0.0, aod550)
    if first_aod_6sv > 0 and abs(first_aod_6sv - aod_band) / max(aod_band, 1e-12) > 0.005:
        calibrated_aot550 *= aod_band / first_aod_6sv
        total.aot550 = calibrated_aot550
        total.run() # 抵消 6SV 内置气溶胶散射系数与 AERONET 反演谱的系统偏差，让模拟 AOD 贴合实测

    ocean = configure_sixs(
        sixs_path, sample, ljn, wavelength_um, band=band, wavelength_mode=wavelength_mode
    )
    if profile is not None:
        ocean.aero_profile = profile
    else:
        assert user_components is not None
        ocean.aero_profile = AeroProfile.User(**user_components) # 必须使用user_components中的组分比例构建模型
    ocean.aot550 = calibrated_aot550
    configure_ocean_surface(ocean, ljn, rho_toa)
    ocean.run()

    rayleigh = configure_sixs(
        sixs_path, sample, ljn, wavelength_um, band=band, wavelength_mode=wavelength_mode
    )
    rayleigh.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    rayleigh.aot550 = 0.0
    rayleigh.run()

    rho_total = total.outputs.atmospheric_intrinsic_reflectance
    rho_rayleigh = rayleigh.outputs.atmospheric_intrinsic_reflectance
    try:
        rho_surface_py6s = total.outputs.atmos_corrected_reflectance_lambertian
    except OutputParsingError:
        rho_surface_py6s = float("nan")
    try:
        rho_surface_ocean = ocean.outputs.atmos_corrected_reflectance_brdf
        rho_glint = ocean.outputs.water_component_glint #太阳耀光
        rho_whitecap = ocean.outputs.water_component_foam # 白帽
        rho_water_model = ocean.outputs.water_component_water # 
        rho_water_leaving = rho_surface_ocean - rho_glint - rho_whitecap
    except OutputParsingError:
        rho_surface_ocean = float("nan")
        rho_glint = float("nan")
        rho_whitecap = float("nan")
        rho_water_model = float("nan")
        rho_water_leaving = float("nan")
    ozone = ozone_cm_atm(ljn)
    ssa_aeronet = nearest_wavelength_value(ljn.ssa_by_um, wavelength_um)
    absorption_aod = nearest_wavelength_value(ljn.absorption_aod_by_um, wavelength_um)
    atmos_profile = atmosphere_profile_name(ljn, sample)
    return {
        "rho_path_total": rho_total,
        "rho_rayleigh": rho_rayleigh,
        "rho_aerosol": rho_total - rho_rayleigh,
        "aod_band": aod_band,
        "aod_band_6sv": total.outputs.optical_depth_total.aerosol,
        "aod550": aod550,
        "aot550_6sv_input": calibrated_aot550,
        "angstrom": alpha,
        "ssa_aeronet_band": ssa_aeronet[1] if ssa_aeronet is not None else "",
        "absorption_aod_aeronet_band": absorption_aod[1] if absorption_aod is not None else "",
        "single_scattering_albedo_6sv": total.outputs.single_scattering_albedo.aerosol,
        "wavelength_configuration": wavelength_configuration_name(sample, band, wavelength_mode),
        "precipitable_water_cm": ljn.water_cm if ljn.water_cm is not None else "",
        "ozone_dobson": ljn.ozone_dobson if ljn.ozone_dobson is not None else "",
        "ozone_cm_atm": ozone if ozone is not None else "",
        "no2_dobson": ljn.no2_dobson if ljn.no2_dobson is not None else "",
        "wind_speed_ms": ljn.wind_speed_ms if ljn.wind_speed_ms is not None else "",
        "chlorophyll_a_mg_m3": ljn.chlorophyll_a_mg_m3 if ljn.chlorophyll_a_mg_m3 is not None else "",
        "ocean_surface_model": "6SV_OceanBRDF(wind_azimuth=0,salinity=34.3)",
        "oc_quality_level": ljn.oc_quality_level,
        "inversion_quality_level": ljn.inversion_quality_level,
        "atmos_profile": atmos_profile,
        "aerosol_model": model_name,
        "aerosol_source": source,
        "trans_rayleigh": total.outputs.transmittance_rayleigh_scattering.total,
        "trans_aerosol": total.outputs.transmittance_aerosol_scattering.total,
        "trans_total_scattering": total.outputs.transmittance_total_scattering.total,
        "spherical_albedo": total.outputs.spherical_albedo.total,
        "rho_surface_lambertian_py6s": rho_surface_py6s,
        "rho_surface_ocean_brdf": rho_surface_ocean,
        "rho_glint_6sv": rho_glint,
        "rho_whitecap_6sv": rho_whitecap,
        "rho_water_model_6sv": rho_water_model,
        "rho_water_leaving_6sv": rho_water_leaving,
    }


def output_base_row(modis_row: Dict[str, str], sample: ModisSample, ljn: LjnAeronetRecord) -> Dict[str, str]:
    dist = modis_row.get("distance_to_request_km", "")
    return {
        "sample_id": ljn.oc_id,
        "oc_id": ljn.oc_id,
        "aeronet_site": ljn.site,
        "modis_datetime_utc": sample.dt_utc.isoformat().replace("+00:00", "Z"),
        "aeronet_datetime_utc": ljn.dt_utc.isoformat().replace("+00:00", "Z"),
        "time_delta_min": "0.000",
        "distance_km": dist,
        "l1b_file": modis_row.get("l1b_file", ""),
        "geo_file": modis_row.get("geo_file", ""),
        "row_0based": modis_row.get("row_0based", ""),
        "col_0based": modis_row.get("col_0based", ""),
        "request_latitude": modis_row.get("request_latitude", ""),
        "request_longitude": modis_row.get("request_longitude", ""),
        "pixel_latitude": modis_row.get("pixel_latitude", ""),
        "pixel_longitude": modis_row.get("pixel_longitude", ""),
        "distance_to_request_km": dist,
        "height_m": modis_row.get("height_m", ""),
        "sensor_range_m": modis_row.get("sensor_range_m", ""),
        "sensor_zenith_deg": modis_row.get("sensor_zenith_deg", ""),
        "sensor_azimuth_deg": modis_row.get("sensor_azimuth_deg", ""),
        "solar_zenith_deg": modis_row.get("solar_zenith_deg", ""),
        "solar_azimuth_deg": modis_row.get("solar_azimuth_deg", ""),
        "relative_azimuth_deg": modis_row.get("relative_azimuth_deg", ""),
        "relative_azimuth_abs_deg": modis_row.get("relative_azimuth_abs_deg", ""),
        "glint_angle_deg": f"{glint_angle_deg(sample):.6f}",
    }


def process_rows_serial_legacy(
    args: argparse.Namespace,
    aeronet: Dict[str, LjnAeronetRecord],
    modis_supplement: Dict[str, ModisSupplementRecord],
) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    aerosol_counts: Counter[str] = Counter()
    bands = {str(band) for band in args.bands} if args.bands else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.modis_csv.open("r", encoding="utf-8-sig", newline="") as f_in, args.output.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        if args.max_rows is not None and args.max_rows > 0:
            reader = islice(reader, args.max_rows)
        oc_left, oc_right = args.oc_id if args.oc_id else None, None
        matched_records = 0
        for modis_row in reader:
            counts["modis_rows"] += 1
            oc_id = (modis_row.get("oc_id") or "").strip()
            if args.oc_id and not (args.oc_id[0] <= int(oc_id) <= args.oc_id[1]):
                continue
            ljn = aeronet.get(oc_id)
            if ljn is None:
                counts["missing_aeronet_oc_id"] += 1
                continue
            if not passes_ljn_quality_control(ljn):
                counts["quality_control_failed"] += 1
                continue
            if not passes_modis_match_quality_control(oc_id, modis_supplement):
                counts["quality_control_failed"] += 1
                continue
            sample_bands = None if bands is None else bands | ({"1240"} if args.swir_residual_correction else set())
            sample = build_modis_sample(modis_row, sample_bands, args.reflectance_input) # 构建 MODIS Sample对象
            if sample is None:
                counts["invalid_modis_sample"] += 1
                continue
            global_aod_550 = get_global_550_aod(ljn)
            matched_records += 1
            base = output_base_row(modis_row, sample, ljn) # 基础信息，后面输出内容会基于这个增加新的结果字段信息
            swir_residual = float("nan")
            if args.swir_residual_correction and "1240" in sample.toa_by_band:
                try:
                    swir = run_scattering(
                        args.sixs_path, sample, ljn, 1.240, args.aerosol_mode,
                        sample.toa_by_band["1240"], "1240", args.wavelength_mode, global_aod_550
                    )
                    swir_residual = float(swir["rho_water_leaving_6sv"])
                except Exception:
                    counts["swir_residual_failed"] += 1
            for band, rho_toa in sorted(sample.toa_by_band.items(), key=lambda item: float(item[0])):
                if bands is not None and band not in bands:
                    continue
                wavelength_um = float(band) / 1000.0
                oc_wavelength_um, oc_values = nearest_oc_products(ljn.oc_products_by_um, wavelength_um)
                try:
                    sca = run_scattering(
                        args.sixs_path,
                        sample,
                        ljn,
                        wavelength_um,
                        args.aerosol_mode,
                        rho_toa,
                        band,
                        args.wavelength_mode,
                        global_aod_550
                    )
                except Exception as exc:
                    counts[f"py6s_failed:{type(exc).__name__}"] += 1
                    continue
                aerosol_counts[str(sca["aerosol_source"])] += 1
                # This is only a diagnostic after removing additive path reflectance.
                rho_minus_path = rho_toa - float(sca["rho_path_total"])
                rho_surface_formula = solve_lambertian_surface_reflectance(rho_toa, sca)
                rho_surface_py6s = float(sca.pop("rho_surface_lambertian_py6s"))
                rho_surface = rho_surface_py6s if math.isfinite(rho_surface_py6s) else rho_surface_formula # 当 Py6S 未返回校正结果时，回退到考虑透过率和球形反照率的解析公式
                rho_water = float(sca["rho_water_leaving_6sv"])
                swir_applied = math.isfinite(swir_residual)
                rho_water_swir = rho_water - swir_residual if swir_applied else rho_water
                row = dict(base)
                radiance_toa = parse_float(modis_row.get(f"radiance_{band}"))
                rho_l1b = parse_float(modis_row.get(f"reflectance_{band}"))
                row.update(
                    {
                        "band": band,
                        "wavelength_um": f"{wavelength_um:.6f}",
                        "oc_wavelength_um": f"{oc_wavelength_um:.6f}" if oc_wavelength_um is not None else "",
                        "oc_rho": format_value(oc_values.get("oc_rho", "")),
                        "oc_lw": format_value(oc_values.get("oc_lw", "")),
                        "oc_lwq": format_value(oc_values.get("oc_lwq", "")),
                        "oc_lwn": format_value(oc_values.get("oc_lwn", "")),
                        "oc_lwn_fq": format_value(oc_values.get("oc_lwn_fq", "")),
                        "oc_lwn_iop": format_value(oc_values.get("oc_lwn_iop", "")),
                        "radiance_toa": format_value(radiance_toa if radiance_toa is not None else ""),
                        "rho_l1b_reflectance_factor": format_value(rho_l1b if rho_l1b is not None else ""),
                        "solar_zenith_cosine": f"{math.cos(math.radians(sample.solar_z)):.8f}",
                        "rho_toa": f"{rho_toa:.8f}",
                        "rho_surface_minus_path": f"{rho_minus_path:.8f}",
                        "rho_toa_minus_atmosphere": f"{rho_minus_path:.8f}",
                        "rho_surface_lambertian": f"{rho_surface:.8f}",
                        "rho_water_leaving_6sv_swir_corrected": f"{rho_water_swir:.8f}",
                        "rho_swir_residual_6sv": f"{swir_residual:.8f}" if swir_applied else "",
                        "swir_residual_applied": "yes" if swir_applied else "no",
                    }
                )
                row.update({key: format_value(value) for key, value in sca.items()})
                writer.writerow(row)
                counts["output_rows"] += 1
                counts["oc_product_matched"] += 1 if oc_wavelength_um is not None else 0
                counts["oc_product_missing"] += 1 if oc_wavelength_um is None else 0
            counts["matched_records"] = matched_records
            # if args.oc_id:
            #     break
            # if args.max_rows and matched_records >= args.max_rows:
            #     break

    return {
        "counts": dict(counts),
        "aerosol_source_counts": dict(aerosol_counts),
        "output": str(args.output),
        "aeronet_index_rows": len(aeronet),
        "bands": sorted(bands, key=float) if bands else "all reflectance_* columns",
    }


def process_modis_row(
    args: argparse.Namespace,
    aeronet: Dict[str, LjnAeronetRecord],
    modis_supplement: Dict[str, ModisSupplementRecord],
    bands: Optional[set[str]],
    modis_row: Dict[str, str],
) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    aerosol_counts: Counter[str] = Counter()
    output_rows: list[Dict[str, str]] = []

    counts["modis_rows"] += 1
    oc_id = (modis_row.get("oc_id") or "").strip()
    if args.oc_id:
        try:
            oc_id_int = int(oc_id)
        except ValueError:
            counts["invalid_oc_id"] += 1
            return {"counts": dict(counts), "aerosol_source_counts": {}, "rows": []}
        if not (args.oc_id[0] <= oc_id_int <= args.oc_id[1]):
            return {"counts": dict(counts), "aerosol_source_counts": {}, "rows": []}

    ljn = aeronet.get(oc_id)
    if ljn is None:
        counts["missing_aeronet_oc_id"] += 1
        return {"counts": dict(counts), "aerosol_source_counts": {}, "rows": []}
    if not passes_ljn_quality_control(ljn):
        counts["quality_control_failed"] += 1
        return {"counts": dict(counts), "aerosol_source_counts": {}, "rows": []}
    if not passes_modis_match_quality_control(oc_id, modis_supplement):
        counts["quality_control_failed"] += 1
        return {"counts": dict(counts), "aerosol_source_counts": {}, "rows": []}

    sample_bands = None if bands is None else bands | ({"1240"} if args.swir_residual_correction else set())
    sample = build_modis_sample(modis_row, sample_bands, args.reflectance_input)
    if sample is None:
        counts["invalid_modis_sample"] += 1
        return {"counts": dict(counts), "aerosol_source_counts": {}, "rows": []}

    global_aod_550 = get_global_550_aod(ljn)
    counts["matched_records"] += 1
    base = output_base_row(modis_row, sample, ljn)
    swir_residual = float("nan")
    if args.swir_residual_correction and "1240" in sample.toa_by_band:
        try:
            swir = run_scattering(
                args.sixs_path, sample, ljn, 1.240, args.aerosol_mode,
                sample.toa_by_band["1240"], "1240", args.wavelength_mode, global_aod_550
            )
            swir_residual = float(swir["rho_water_leaving_6sv"])
        except Exception:
            counts["swir_residual_failed"] += 1

    for band, rho_toa in sorted(sample.toa_by_band.items(), key=lambda item: float(item[0])):
        if bands is not None and band not in bands:
            continue
        wavelength_um = float(band) / 1000.0
        oc_wavelength_um, oc_values = nearest_oc_products(ljn.oc_products_by_um, wavelength_um)
        try:
            sca = run_scattering(
                args.sixs_path,
                sample,
                ljn,
                wavelength_um,
                args.aerosol_mode,
                rho_toa,
                band,
                args.wavelength_mode,
                global_aod_550,
            )
        except Exception as exc:
            counts[f"py6s_failed:{type(exc).__name__}"] += 1
            continue
        aerosol_counts[str(sca["aerosol_source"])] += 1
        # This is only a diagnostic after removing additive path reflectance.
        rho_minus_path = rho_toa - float(sca["rho_path_total"])
        rho_surface_formula = solve_lambertian_surface_reflectance(rho_toa, sca)
        rho_surface_py6s = float(sca.pop("rho_surface_lambertian_py6s"))
        rho_surface = rho_surface_py6s if math.isfinite(rho_surface_py6s) else rho_surface_formula
        rho_water = float(sca["rho_water_leaving_6sv"])
        swir_applied = math.isfinite(swir_residual)
        rho_water_swir = rho_water - swir_residual if swir_applied else rho_water
        row = dict(base)
        radiance_toa = parse_float(modis_row.get(f"radiance_{band}"))
        rho_l1b = parse_float(modis_row.get(f"reflectance_{band}"))
        row.update(
            {
                "band": band,
                "wavelength_um": f"{wavelength_um:.6f}",
                "oc_wavelength_um": f"{oc_wavelength_um:.6f}" if oc_wavelength_um is not None else "",
                "oc_rho": format_value(oc_values.get("oc_rho", "")),
                "oc_lw": format_value(oc_values.get("oc_lw", "")),
                "oc_lwq": format_value(oc_values.get("oc_lwq", "")),
                "oc_lwn": format_value(oc_values.get("oc_lwn", "")),
                "oc_lwn_fq": format_value(oc_values.get("oc_lwn_fq", "")),
                "oc_lwn_iop": format_value(oc_values.get("oc_lwn_iop", "")),
                "radiance_toa": format_value(radiance_toa if radiance_toa is not None else ""),
                "rho_l1b_reflectance_factor": format_value(rho_l1b if rho_l1b is not None else ""),
                "solar_zenith_cosine": f"{math.cos(math.radians(sample.solar_z)):.8f}",
                "rho_toa": f"{rho_toa:.8f}",
                "rho_surface_minus_path": f"{rho_minus_path:.8f}",
                "rho_toa_minus_atmosphere": f"{rho_minus_path:.8f}",
                "rho_surface_lambertian": f"{rho_surface:.8f}",
                "rho_water_leaving_6sv_swir_corrected": f"{rho_water_swir:.8f}",
                "rho_swir_residual_6sv": f"{swir_residual:.8f}" if swir_applied else "",
                "swir_residual_applied": "yes" if swir_applied else "no",
            }
        )
        row.update({key: format_value(value) for key, value in sca.items()})
        output_rows.append(row)
        counts["output_rows"] += 1
        counts["oc_product_matched"] += 1 if oc_wavelength_um is not None else 0
        counts["oc_product_missing"] += 1 if oc_wavelength_um is None else 0

    return {
        "counts": dict(counts),
        "aerosol_source_counts": dict(aerosol_counts),
        "rows": output_rows,
    }


def process_rows(
    args: argparse.Namespace,
    aeronet: Dict[str, LjnAeronetRecord],
    modis_supplement: Dict[str, ModisSupplementRecord],
) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    aerosol_counts: Counter[str] = Counter()
    bands = {str(band) for band in args.bands} if args.bands else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.modis_csv.open("r", encoding="utf-8-sig", newline="") as f_in, args.output.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        row_iter = reader
        if args.max_rows is not None and args.max_rows > 0:
            row_iter = islice(reader, args.max_rows)

        if args.workers == 1:
            results = (
                process_modis_row(args, aeronet, modis_supplement, bands, modis_row)
                for modis_row in row_iter
            )
            for result in results:
                counts.update(result["counts"])
                aerosol_counts.update(result["aerosol_source_counts"])
                writer.writerows(result["rows"])
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(args, aeronet, modis_supplement, bands),
            ) as executor:
                for result in executor.map(process_row_worker, row_iter, chunksize=args.chunksize):
                    counts.update(result["counts"])
                    aerosol_counts.update(result["aerosol_source_counts"])
                    writer.writerows(result["rows"])

    return {
        "counts": dict(counts),
        "aerosol_source_counts": dict(aerosol_counts),
        "output": str(args.output),
        "aeronet_index_rows": len(aeronet),
        "bands": sorted(bands, key=float) if bands else "all reflectance_* columns",
        "workers": args.workers,
        "chunksize": args.chunksize,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_workers = max(1, min(4, os.cpu_count() or 1))
    parser.add_argument("--modis-csv", type=Path, default=Path("Data/ljn/modis_l1b_result.csv"))
    parser.add_argument(
        "--modis-supplement-csv",
        type=Path,
        help="MODIS supplement CSV with oc_id and abs_time_diff_minutes; defaults to modis_l1b_result_v2_rayleigh_rhos_1.csv next to --modis-csv.",
    )
    parser.add_argument("--aeronet-csv", type=Path, default=Path("Data/ljn/lwn_with_aod_inv15_ocid.csv"))
    parser.add_argument("--output", type=Path, default=Path("Code/py6s_only/outputs/ljn_ocid_surface_reflectance.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("Code/py6s_only/outputs/ljn_ocid_surface_reflectance_summary.json"))
    parser.add_argument("--sixs-path", type=Path, default=Path("Code/Py6SV/envs/py6s/Library/bin/sixs.exe"))
    parser.add_argument("--max-rows", type=int, help="Process only this many matched MODIS records.") #一般用站点、特定波段限制结果数量
    parser.add_argument("--oc-id", nargs=2, type=int, help="Process oc_id range, format: left right, e.g. 100 200")
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of worker processes. Use 1 for serial execution.")
    parser.add_argument("--chunksize", type=int, default=1, help="Rows submitted to each worker task batch.")
    parser.add_argument("--bands", nargs="+", help="Limit MODIS reflectance bands, e.g. 412 443 488 555.")
    parser.add_argument("--aerosol-mode", choices=["auto", "inv", "fallback"], default="auto")
    parser.add_argument("--wavelength-mode", choices=["modis-rsr", "point"], default="modis-rsr")
    parser.add_argument(
        "--reflectance-input",
        choices=["l1b-rho-cos", "toa"],
        default="l1b-rho-cos",
        help="MODIS L1B reflectance is rho*cos(solar_zenith); use toa only for pre-corrected input.",
    )
    parser.add_argument("--no-swir-residual-correction", dest="swir_residual_correction", action="store_false")
    parser.set_defaults(swir_residual_correction=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    if not args.sixs_path.exists():
        raise SystemExit(f"Cannot find sixs executable: {args.sixs_path}")
    modis_supplement_csv = modis_supplement_csv_path(args)
    if not modis_supplement_csv.exists():
        raise SystemExit(f"Cannot find MODIS supplement CSV: {modis_supplement_csv}")
    print(f"Loading AERONET index from {args.aeronet_csv}")
    aeronet = read_aeronet_index(args.aeronet_csv, tuple(args.oc_id) if args.oc_id else None)
    print(f"Loaded {len(aeronet)} AERONET oc_id records")
    print(f"Loading MODIS supplement index from {modis_supplement_csv}")
    modis_supplement = read_modis_supplement_index(
        modis_supplement_csv,
        tuple(args.oc_id) if args.oc_id else None,
    )
    print(f"Loaded {len(modis_supplement)} MODIS supplement oc_id records")
    summary = process_rows(args, aeronet, modis_supplement)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote correction CSV to {args.output}")
    print(f"Wrote summary JSON to {args.summary_json}")
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
