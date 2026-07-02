#!/usr/bin/env python
"""Run Py6S correction for ljn MODIS/AERONET rows matched by oc_id."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

from Py6S import AeroProfile

from modules.aeronet import LjnAeronetRecord, read_aeronet_index
from modules.common import interpolate_linear, nearest_oc_products, nearest_wavelength_value, parse_float
from modules.modis import ModisSample, build_modis_sample
from modules.sixs_utils import (
    configure_sixs,
    format_value,
    ozone_cm_atm,
    solve_lambertian_surface_reflectance,
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
    "band",
    "wavelength_um",
    "oc_wavelength_um",
    "oc_rho",
    "oc_lw",
    "oc_lwq",
    "oc_lwn",
    "oc_lwn_fq",
    "oc_lwn_iop",
    "radiance_toa",
    "rho_toa",
    "rho_path_total",
    "rho_rayleigh",
    "rho_aerosol",
    "rho_surface_minus_path",
    "rho_toa_minus_atmosphere",
    "rho_surface_lambertian",
    "aod_band",
    "aod550",
    "angstrom",
    "precipitable_water_cm",
    "ozone_dobson",
    "ozone_cm_atm",
    "no2_dobson",
    "atmos_profile",
    "aerosol_model",
    "aerosol_source",
    "trans_rayleigh",
    "trans_aerosol",
    "trans_total_scattering",
    "spherical_albedo",
]


def interpolate_aod(record: LjnAeronetRecord, wavelength_um: float) -> Tuple[float, float]:
    nearest = nearest_wavelength_value(record.aod_by_um, wavelength_um, max_delta=0.0005)
    alpha = record.angstrom_or_default()
    if nearest:
        return nearest[1], alpha
    if record.aod_by_um:
        nearest_550 = nearest_wavelength_value(record.aod_by_um, 0.55, max_delta=0.03)
        if nearest_550:
            return nearest_550[1] * (wavelength_um / nearest_550[0]) ** (-alpha), alpha
    nearest_any = nearest_wavelength_value(record.aod_by_um, wavelength_um)
    if nearest_any is None:
        raise ValueError("AOD spectrum is empty")
    return nearest_any[1] * (wavelength_um / nearest_any[0]) ** (-alpha), alpha


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
) -> Dict[str, float | str]:
    aod_band, alpha = interpolate_aod(ljn, wavelength_um)
    aod550 = aod_band * (0.55 / wavelength_um) ** (-alpha)

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

    total = configure_sixs(sixs_path, sample, ljn, wavelength_um)
    if profile is not None:
        total.aero_profile = profile
    else:
        assert user_components is not None
        total.aero_profile = AeroProfile.User(**user_components)
    total.aot550 = max(0.0, aod550)
    total.run()

    rayleigh = configure_sixs(sixs_path, sample, ljn, wavelength_um)
    rayleigh.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    rayleigh.aot550 = 0.0
    rayleigh.run()

    rho_total = total.outputs.atmospheric_intrinsic_reflectance
    rho_rayleigh = rayleigh.outputs.atmospheric_intrinsic_reflectance
    ozone = ozone_cm_atm(ljn)
    atmos_profile = (
        f"LJN_UserWaterAndOzone(water_gcm2={ljn.water_cm:.3f},ozone_cmatm={ozone:.4f})"
        if ljn.water_cm is not None and ozone is not None
        else "MidlatitudeSummer(fallback)"
    )
    return {
        "rho_path_total": rho_total,
        "rho_rayleigh": rho_rayleigh,
        "rho_aerosol": rho_total - rho_rayleigh,
        "aod_band": aod_band,
        "aod550": aod550,
        "angstrom": alpha,
        "precipitable_water_cm": ljn.water_cm if ljn.water_cm is not None else "",
        "ozone_dobson": ljn.ozone_dobson if ljn.ozone_dobson is not None else "",
        "ozone_cm_atm": ozone if ozone is not None else "",
        "no2_dobson": ljn.no2_dobson if ljn.no2_dobson is not None else "",
        "atmos_profile": atmos_profile,
        "aerosol_model": model_name,
        "aerosol_source": source,
        "trans_rayleigh": total.outputs.transmittance_rayleigh_scattering.total,
        "trans_aerosol": total.outputs.transmittance_aerosol_scattering.total,
        "trans_total_scattering": total.outputs.transmittance_total_scattering.total,
        "spherical_albedo": total.outputs.spherical_albedo.total,
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
    }


def process_rows(args: argparse.Namespace, aeronet: Dict[str, LjnAeronetRecord]) -> Dict[str, object]:
    counts: Counter[str] = Counter()
    aerosol_counts: Counter[str] = Counter()
    bands = {str(band) for band in args.bands} if args.bands else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.modis_csv.open("r", encoding="utf-8-sig", newline="") as f_in, args.output.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        matched_records = 0
        for modis_row in reader:
            counts["modis_rows"] += 1
            oc_id = (modis_row.get("oc_id") or "").strip()
            if args.oc_id and oc_id != args.oc_id:
                continue
            ljn = aeronet.get(oc_id)
            if ljn is None:
                counts["missing_aeronet_oc_id"] += 1
                continue
            sample = build_modis_sample(modis_row, bands)
            if sample is None:
                counts["invalid_modis_sample"] += 1
                continue
            matched_records += 1
            base = output_base_row(modis_row, sample, ljn)
            for band, rho_toa in sorted(sample.toa_by_band.items(), key=lambda item: float(item[0])):
                wavelength_um = float(band) / 1000.0
                oc_wavelength_um, oc_values = nearest_oc_products(ljn.oc_products_by_um, wavelength_um)
                try:
                    sca = run_scattering(args.sixs_path, sample, ljn, wavelength_um, args.aerosol_mode)
                except Exception as exc:
                    counts[f"py6s_failed:{type(exc).__name__}"] += 1
                    continue
                aerosol_counts[str(sca["aerosol_source"])] += 1
                rho_minus_path = rho_toa - float(sca["rho_path_total"]) #未考虑大气透过率、球面反照率等影响因素
                row = dict(base)
                radiance_toa = parse_float(modis_row.get(f"radiance_{band}"))
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
                        "rho_toa": f"{rho_toa:.8f}",
                        "rho_surface_minus_path": f"{rho_minus_path:.8f}",
                        "rho_toa_minus_atmosphere": f"{rho_minus_path:.8f}",
                        "rho_surface_lambertian": f"{solve_lambertian_surface_reflectance(rho_toa, sca):.8f}",
                    }
                )
                row.update({key: format_value(value) for key, value in sca.items()})
                writer.writerow(row)
                counts["output_rows"] += 1
                counts["oc_product_matched"] += 1 if oc_wavelength_um is not None else 0
                counts["oc_product_missing"] += 1 if oc_wavelength_um is None else 0
            counts["matched_records"] = matched_records
            if args.oc_id:
                break
            if args.max_rows and matched_records >= args.max_rows:
                break

    return {
        "counts": dict(counts),
        "aerosol_source_counts": dict(aerosol_counts),
        "output": str(args.output),
        "aeronet_index_rows": len(aeronet),
        "bands": sorted(bands, key=float) if bands else "all reflectance_* columns",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modis-csv", type=Path, default=Path("Data/ljn/modis_l1b_result.csv"))
    parser.add_argument("--aeronet-csv", type=Path, default=Path("Data/ljn/lwn_with_aod_inv15_ocid.csv"))
    parser.add_argument("--output", type=Path, default=Path("Code/py6s_only/outputs/ljn_ocid_surface_reflectance.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("Code/py6s_only/outputs/ljn_ocid_surface_reflectance_summary.json"))
    parser.add_argument("--sixs-path", type=Path, default=Path("Code/Py6SV/envs/py6s/Library/bin/sixs.exe"))
    parser.add_argument("--max-rows", type=int, help="Process only this many matched MODIS records.")
    parser.add_argument("--oc-id", help="Process one oc_id only.")
    parser.add_argument("--bands", nargs="+", help="Limit MODIS reflectance bands, e.g. 412 443 488 555.")
    parser.add_argument("--aerosol-mode", choices=["auto", "inv", "fallback"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sixs_path.exists():
        raise SystemExit(f"Cannot find sixs executable: {args.sixs_path}")
    print(f"Loading AERONET index from {args.aeronet_csv}")
    aeronet = read_aeronet_index(args.aeronet_csv)
    print(f"Loaded {len(aeronet)} AERONET oc_id records")
    summary = process_rows(args, aeronet)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote correction CSV to {args.output}")
    print(f"Wrote summary JSON to {args.summary_json}")
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
