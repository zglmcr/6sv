#!/usr/bin/env python
"""Compare CSV Rayleigh LUT radiance columns against Py6S Rayleigh scattering."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from Py6S import AeroProfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.aeronet import LjnAeronetRecord, passes_ljn_quality_control, read_aeronet_index
from modules.common import parse_datetime, parse_float
from modules.modis import ModisSample
from modules.sixs_utils import configure_sixs, wavelength_configuration_name
from tools.plot_oc_vs_py6s import interpolate_solar_irradiance


DEFAULT_RAYLEIGH_CSV = Path("Data/ljn/modis_l1b_result_v2_rayleigh_rhos_1.csv")
DEFAULT_MODIS_CSV = Path("Data/ljn/modis_l1b_result.csv")
DEFAULT_AERONET_CSV = Path("Data/ljn/lwn_with_aod_inv15_ocid.csv")
DEFAULT_SIXS_PATH = Path("Code/Py6SV/envs/py6s/Library/bin/sixs.exe")
DEFAULT_OUTPUT = Path("Code/py6s_only/outputs/rayleigh_lut_compare.csv")
DEFAULT_SUMMARY = Path("Code/py6s_only/outputs/rayleigh_lut_compare_summary.json")


def discover_lr_bands(fieldnames: list[str]) -> list[str]:
    bands = []
    for name in fieldnames:
        if name.startswith("Lr_"):
            band = name[3:]
            if band.isdigit():
                bands.append(band)
    return sorted(set(bands), key=float)


def row_in_oc_id_range(row: dict[str, str], oc_id_range: Optional[tuple[int, int]]) -> bool:
    if oc_id_range is None:
        return True
    oc_id = (row.get("oc_id") or "").strip()
    try:
        value = int(oc_id)
    except ValueError:
        return False
    return oc_id_range[0] <= value <= oc_id_range[1]


def load_rayleigh_rows(
    path: Path,
    bands: Optional[list[str]],
    max_rows: Optional[int],
    oc_id_range: Optional[tuple[int, int]],
) -> tuple[list[dict[str, str]], list[str]]:
    selected: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {path}")
        available_bands = discover_lr_bands(reader.fieldnames)
        selected_bands = bands or available_bands
        missing = [band for band in selected_bands if f"Lr_{band}" not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV is missing Lr columns for bands: {missing}")
        for row in reader:
            if not row_in_oc_id_range(row, oc_id_range):
                continue
            selected.append(row)
            if max_rows is not None and max_rows > 0 and len(selected) >= max_rows:
                break
    return selected, selected_bands


def load_modis_index(path: Path, wanted_oc_ids: set[str]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            oc_id = (row.get("oc_id") or "").strip()
            if oc_id in wanted_oc_ids and oc_id not in index:
                index[oc_id] = row
                if len(index) == len(wanted_oc_ids):
                    break
    return index


def build_sample_from_modis_row(row: dict[str, str]) -> Optional[ModisSample]:
    lat = parse_float(row.get("pixel_latitude"))
    lon = parse_float(row.get("pixel_longitude"))
    solar_z = parse_float(row.get("solar_zenith_deg"))
    sensor_z = parse_float(row.get("sensor_zenith_deg"))
    relaz = parse_float(row.get("relative_azimuth_abs_deg"))
    if None in (lat, lon, solar_z, sensor_z, relaz):
        return None
    return ModisSample(
        sample_id=(row.get("oc_id") or "").strip(),
        dt_utc=parse_datetime(row.get("date", ""), row.get("Time_UTC", "")),
        lat=lat,
        lon=lon,
        solar_z=solar_z,
        sensor_z=sensor_z,
        relative_azimuth=relaz,
        toa_by_band={},
        raw=row,
    )


def compute_py6s_rayleigh(
    sixs_path: Path,
    sample: ModisSample,
    record: LjnAeronetRecord,
    band: str,
    wavelength_mode: str,
) -> float:
    wavelength_um = float(band) / 1000.0
    rayleigh = configure_sixs(
        sixs_path,
        sample,
        record,
        wavelength_um,
        band=band,
        wavelength_mode=wavelength_mode,
    )
    rayleigh.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    rayleigh.aot550 = 0.0
    rayleigh.run()
    return float(rayleigh.outputs.atmospheric_intrinsic_reflectance)


def rayleigh_radiance_from_reflectance(
    rho_rayleigh: float,
    wavelength_um: float,
    solar_zenith_deg: float,
    solar_scale: float,
) -> float:
    solar_irradiance = interpolate_solar_irradiance(wavelength_um) * solar_scale
    mu0 = math.cos(math.radians(solar_zenith_deg))
    return rho_rayleigh * solar_irradiance * mu0 / math.pi


def rayleigh_reflectance_from_radiance(
    radiance: float,
    wavelength_um: float,
    solar_zenith_deg: float,
    solar_scale: float,
) -> float:
    solar_irradiance = interpolate_solar_irradiance(wavelength_um) * solar_scale
    mu0 = math.cos(math.radians(solar_zenith_deg))
    return math.pi * radiance / (solar_irradiance * mu0)


def empty_value() -> str:
    return ""


def format_float(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return empty_value()
    return f"{value:.10g}"


def summarize(rows: list[dict[str, str]], counts: Counter[str]) -> dict[str, object]:
    by_band: dict[str, list[float]] = defaultdict(list)
    by_band_abs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        band = row["band"]
        diff = parse_float(row.get("lr_diff_py6s_minus_table"))
        abs_diff = parse_float(row.get("lr_abs_diff"))
        if diff is not None:
            by_band[band].append(diff)
        if abs_diff is not None:
            by_band_abs[band].append(abs_diff)

    per_band = {}
    for band in sorted(set(by_band) | set(by_band_abs), key=float):
        diffs = by_band.get(band, [])
        abs_diffs = by_band_abs.get(band, [])
        per_band[band] = {
            "n": len(diffs),
            "bias_py6s_minus_table": sum(diffs) / len(diffs) if diffs else None,
            "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else None,
            "rmse": math.sqrt(sum(value * value for value in diffs) / len(diffs)) if diffs else None,
        }
    return {"counts": dict(counts), "per_band": per_band}


def compare(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, object]]:
    oc_id_range = tuple(args.oc_id) if args.oc_id else None
    rows, bands = load_rayleigh_rows(args.rayleigh_csv, args.bands, args.max_rows, oc_id_range)
    wanted_oc_ids = {(row.get("oc_id") or "").strip() for row in rows if (row.get("oc_id") or "").strip()}
    aeronet = read_aeronet_index(args.aeronet_csv, wanted_oc_ids)
    modis = load_modis_index(args.modis_csv, wanted_oc_ids)
    counts: Counter[str] = Counter()
    output_rows: list[dict[str, str]] = []

    for row in rows:
        counts["input_rows"] += 1
        oc_id = (row.get("oc_id") or "").strip()
        record = aeronet.get(oc_id)
        if record is None:
            counts["missing_aeronet_oc_id"] += 1
            continue
        if args.quality_control and not passes_ljn_quality_control(record):
            counts["quality_control_failed"] += 1
            continue
        modis_row = modis.get(oc_id)
        if modis_row is None:
            counts["missing_modis_oc_id"] += 1
            continue
        sample = build_sample_from_modis_row(modis_row)
        if sample is None:
            counts["invalid_modis_sample"] += 1
            continue
        for band in bands:
            table_lr = parse_float(row.get(f"Lr_{band}"))
            if table_lr is None:
                counts["missing_table_lr"] += 1
                continue
            wavelength_um = float(band) / 1000.0
            try:
                py6s_rho = compute_py6s_rayleigh(args.sixs_path, sample, record, band, args.wavelength_mode)
            except Exception as exc:
                counts[f"py6s_failed:{type(exc).__name__}"] += 1
                continue

            py6s_lr = None
            table_rho = None
            lr_diff = None
            rho_diff = None
            try:
                py6s_lr = rayleigh_radiance_from_reflectance(
                    py6s_rho, wavelength_um, sample.solar_z, args.solar_scale
                )
                table_rho = rayleigh_reflectance_from_radiance(
                    table_lr, wavelength_um, sample.solar_z, args.solar_scale
                )
                lr_diff = py6s_lr - table_lr
                rho_diff = py6s_rho - table_rho
            except ValueError:
                counts["solar_irradiance_unavailable"] += 1

            output_rows.append(
                {
                    "oc_id": oc_id,
                    "date": row.get("date", ""),
                    "Time_UTC": row.get("Time_UTC", ""),
                    "band": band,
                    "wavelength_um": f"{wavelength_um:.6f}",
                    "wavelength_configuration": wavelength_configuration_name(sample, band, args.wavelength_mode),
                    "solar_zenith_deg": format_float(sample.solar_z),
                    "sensor_zenith_deg": format_float(sample.sensor_z),
                    "relative_azimuth_abs_deg": format_float(sample.relative_azimuth),
                    "table_Lr": format_float(table_lr),
                    "py6s_Lr": format_float(py6s_lr),
                    "lr_diff_py6s_minus_table": format_float(lr_diff),
                    "lr_abs_diff": format_float(abs(lr_diff) if lr_diff is not None else None),
                    "table_rho_rayleigh": format_float(table_rho),
                    "py6s_rho_rayleigh": format_float(py6s_rho),
                    "rho_diff_py6s_minus_table": format_float(rho_diff),
                    "radiance_toa": format_float(parse_float(row.get(f"radiance_{band}"))),
                    "modis_radiance_toa": format_float(parse_float(modis_row.get(f"radiance_{band}"))),
                    "modis_reflectance": format_float(parse_float(modis_row.get(f"reflectance_{band}"))),
                    "aeronet_site": record.site,
                    "oc_quality_level": record.oc_quality_level,
                    "inversion_quality_level": record.inversion_quality_level,
                    "inv_time_diff_minutes": format_float(record.inv_time_diff_minutes),
                }
            )
            counts["output_rows"] += 1
    return output_rows, summarize(output_rows, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rayleigh-csv", type=Path, default=DEFAULT_RAYLEIGH_CSV)
    parser.add_argument("--modis-csv", type=Path, default=DEFAULT_MODIS_CSV)
    parser.add_argument("--aeronet-csv", type=Path, default=DEFAULT_AERONET_CSV)
    parser.add_argument("--sixs-path", type=Path, default=DEFAULT_SIXS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-rows", type=int, help="Read at most this many Rayleigh CSV rows after oc_id filtering.")
    parser.add_argument("--oc-id", nargs=2, type=int, help="Process oc_id range, e.g. 122000 122100.")
    parser.add_argument("--bands", nargs="+", help="Bands to compare, e.g. 412 443 488 555. Defaults to all Lr_* columns.")
    parser.add_argument("--wavelength-mode", choices=["modis-rsr", "point"], default="modis-rsr")
    parser.add_argument(
        "--solar-scale",
        type=float,
        default=1.0,
        help="Scale applied to the existing solar irradiance table when converting reflectance to Lr.",
    )
    parser.add_argument("--no-quality-control", dest="quality_control", action="store_false")
    parser.set_defaults(quality_control=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sixs_path.exists():
        raise SystemExit(f"Cannot find sixs executable: {args.sixs_path}")
    rows, summary = compare(args)
    if not rows:
        raise SystemExit(f"No comparison rows were produced: {json.dumps(summary['counts'], indent=2)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    print(f"Wrote comparison CSV to {args.output}")
    print(f"Wrote summary JSON to {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
