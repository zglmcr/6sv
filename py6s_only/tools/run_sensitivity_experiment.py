#!/usr/bin/env python
"""Compare physically plausible Py6S configurations against AERONET-OC."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ljn_ocid_py6s_correction import run_scattering, get_global_550_aod
from modules.aeronet import read_aeronet_index
from modules.common import nearest_oc_products, parse_float
from modules.modis import build_modis_sample, glint_angle_deg
from tools.plot_oc_vs_py6s import calculate_statistics, interpolate_solar_irradiance


VARIANTS = {
    "baseline_inv_rsr": {"aerosol_mode": "auto", "wavelength_mode": "modis-rsr", "azimuth": "as-is"},
    "point_wavelength": {"aerosol_mode": "auto", "wavelength_mode": "point", "azimuth": "as-is"},
    "fallback_aerosol": {"aerosol_mode": "fallback", "wavelength_mode": "modis-rsr", "azimuth": "as-is"},
    "supplement_azimuth": {"aerosol_mode": "auto", "wavelength_mode": "modis-rsr", "azimuth": "supplement"},
}


def load_modis_candidates(path: Path, bands: set[str], max_nir_toa: float, max_distance_km: float) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected: list[dict[str, str]] = []
    for row in rows:
        sample = build_modis_sample(row, bands, "l1b-rho-cos")
        if sample is None or sample.sensor_z > 60.0 or glint_angle_deg(sample) < 40.0:
            continue
        distance = parse_float(row.get("distance_to_request_km"))
        if distance is None or distance > max_distance_km:
            continue
        nir = parse_float(row.get("reflectance_869")) or parse_float(row.get("reflectance_859"))
        if nir is not None:
            nir /= math.cos(math.radians(sample.solar_z))
            if nir > max_nir_toa:
                continue
        selected.append(row)
    return selected


def oc_reflectance(record, wavelength_um: float, product: str) -> tuple[float, float] | None:
    oc_wavelength_um, values = nearest_oc_products(record.oc_products_by_um, wavelength_um)
    if oc_wavelength_um is None or abs(oc_wavelength_um - wavelength_um) > 0.015:
        return None
    column = "oc_lwn_fq" if product == "fq" else "oc_lwn_iop"
    lwn = values.get(column)
    if lwn is None or not math.isfinite(lwn):
        return None
    return math.pi * lwn / interpolate_solar_irradiance(oc_wavelength_um), oc_wavelength_um


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    bands = set(args.bands)
    candidates = load_modis_candidates(args.modis_csv, bands, args.max_nir_toa, args.max_distance_km)
    if args.oc_ids:
        wanted = set(args.oc_ids)
        candidates = [row for row in candidates if row.get("oc_id") in wanted]
    if len(candidates) > args.max_samples:
        if args.max_samples == 1:
            candidates = [candidates[len(candidates) // 2]]
        else:
            indexes = [round(i * (len(candidates) - 1) / (args.max_samples - 1)) for i in range(args.max_samples)]
            candidates = [candidates[index] for index in indexes]
    wanted_oc_ids = {(row.get("oc_id") or "").strip() for row in candidates}
    aeronet = read_aeronet_index(args.aeronet_csv, wanted_oc_ids)
    results: list[dict[str, object]] = []

    for modis_row in candidates:
        oc_id = (modis_row.get("oc_id") or "").strip()
        record = aeronet.get(oc_id)
        sample = build_modis_sample(modis_row, bands, "l1b-rho-cos")
        global_aod_550 = get_global_550_aod(record)
        if record is None or sample is None:
            continue
        for band, rho_toa in sorted(sample.toa_by_band.items(), key=lambda item: float(item[0])):
            wavelength_um = float(band) / 1000.0
            reference = oc_reflectance(record, wavelength_um, args.oc_product)
            if reference is None:
                continue
            rho_oc, oc_wavelength_um = reference
            variants = VARIANTS if not args.variants else {name: VARIANTS[name] for name in args.variants}
            for variant_name, settings in variants.items():
                configured_sample = copy.copy(sample)
                if settings["azimuth"] == "supplement":
                    configured_sample.relative_azimuth = 180.0 - configured_sample.relative_azimuth
                scattering = run_scattering(
                    args.sixs_path,
                    configured_sample,
                    record,
                    wavelength_um,
                    str(settings["aerosol_mode"]),
                    rho_toa,
                    band,
                    str(settings["wavelength_mode"]),
                    global_aod_550
                )
                results.append(
                    {
                        "variant": variant_name,
                        "oc_id": oc_id,
                        "site": record.site,
                        "oc_quality_level": record.oc_quality_level,
                        "inversion_quality_level": record.inversion_quality_level,
                        "band": band,
                        "oc_wavelength_um": oc_wavelength_um,
                        "rho_oc": rho_oc,
                        "rho_toa": rho_toa,
                        "rho_water_leaving_6sv": scattering["rho_water_leaving_6sv"],
                        "rho_surface_lambertian": scattering["rho_surface_lambertian_py6s"],
                        "rho_path_total": scattering["rho_path_total"],
                        "aod_band_target": scattering["aod_band"],
                        "aod_band_6sv": scattering["aod_band_6sv"],
                        "ssa_aeronet": scattering["ssa_aeronet_band"],
                        "ssa_6sv": scattering["single_scattering_albedo_6sv"],
                        "sensor_zenith_deg": sample.sensor_z,
                        "glint_angle_deg": glint_angle_deg(sample),
                        "relative_azimuth_used_deg": configured_sample.relative_azimuth,
                        "aerosol_source": scattering["aerosol_source"],
                        "wavelength_configuration": scattering["wavelength_configuration"],
                    }
                )
    return results


def summarize(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[str(row["variant"])].append(row)
    summary: dict[str, dict[str, float]] = {}
    for variant, rows in grouped.items():
        valid = [
            row
            for row in rows
            if math.isfinite(float(row["rho_oc"])) and math.isfinite(float(row["rho_water_leaving_6sv"]))
        ]
        x = [float(row["rho_oc"]) for row in valid]
        y = [float(row["rho_water_leaving_6sv"]) for row in valid]
        stats = calculate_statistics(x, y) if x else {}
        summary[variant] = {"n": float(len(valid)), **stats}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modis-csv", type=Path, default=Path("Data/ljn/modis_l1b_result.csv"))
    parser.add_argument("--aeronet-csv", type=Path, default=Path("Data/ljn/lwn_with_aod_inv15_ocid.csv"))
    parser.add_argument("--sixs-path", type=Path, default=Path("Code/Py6SV/envs/py6s/Library/bin/sixs.exe"))
    parser.add_argument("--output", type=Path, default=Path("Code/py6s_only/outputs/py6s_sensitivity.csv"))
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--oc-ids", nargs="+")
    parser.add_argument("--bands", nargs="+", default=["412", "443", "555"])
    parser.add_argument("--oc-product", choices=["fq", "iop"], default="fq")
    parser.add_argument("--max-nir-toa", type=float, default=0.05)
    parser.add_argument("--max-distance-km", type=float, default=0.5)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run(args)
    if not results:
        raise SystemExit("No experiment pairs were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summarize(results), indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
