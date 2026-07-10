#!/usr/bin/env python
"""Plot AERONET-OC reflectance against Py6S-corrected BOA reflectance."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "py6s_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Thuillier et al. (2003) mean extraterrestrial spectral irradiance,
# expressed in the same units as AERONET-OC Lwn: mW cm-2 um-1.
SOLAR_IRRADIANCE_MW_CM2_UM = {
    0.400: 160.00,
    0.412: 172.81,
    0.440: 188.75,
    0.443: 190.20,
    0.490: 196.26,
    0.500: 194.90,
    0.510: 192.80,
    0.532: 187.20,
    0.551: 184.12,
    0.555: 183.20,
    0.560: 181.90,
    0.620: 165.50,
    0.667: 151.60,
    0.675: 149.00,
    0.681: 147.00,
    0.709: 138.80,
    0.779: 119.30,
    0.865: 95.70,
    0.870: 94.90,
    1.020: 68.50,
}


def parse_float(value: str | None) -> float | None:
    try:
        number = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def interpolate_solar_irradiance(wavelength_um: float) -> float:
    wavelengths = sorted(SOLAR_IRRADIANCE_MW_CM2_UM)
    if wavelength_um < wavelengths[0] or wavelength_um > wavelengths[-1]:
        raise ValueError(f"No solar irradiance available at {wavelength_um:.6f} um")
    if wavelength_um in SOLAR_IRRADIANCE_MW_CM2_UM:
        return SOLAR_IRRADIANCE_MW_CM2_UM[wavelength_um]
    for left, right in zip(wavelengths, wavelengths[1:]):
        if left <= wavelength_um <= right:
            fraction = (wavelength_um - left) / (right - left)
            return SOLAR_IRRADIANCE_MW_CM2_UM[left] + fraction * (
                SOLAR_IRRADIANCE_MW_CM2_UM[right] - SOLAR_IRRADIANCE_MW_CM2_UM[left]
            )
    raise ValueError(f"Cannot interpolate solar irradiance at {wavelength_um:.6f} um")


def load_pairs(
    path: Path,
    max_wavelength_delta_nm: float,
    oc_product: str,
    quality_filter: bool,
    max_sensor_zenith_deg: float,
    min_glint_angle_deg: float,
    max_nir_toa: float,
    max_aod: float | None,
    py6s_field: str,
    max_distance_km: float,
) -> tuple[list[dict[str, float | str]], Counter[str]]:
    pairs: list[dict[str, float | str]] = []
    counts: Counter[str] = Counter()
    lwn_column = "oc_lwn_fq" if oc_product == "fq" else "oc_lwn_iop"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            lwn_column,
            py6s_field,
            "wavelength_um",
            "oc_wavelength_um",
            "sensor_zenith_deg",
            "glint_angle_deg",
            "rho_toa",
            "band",
            "oc_id",
            "distance_to_request_km",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing {sorted(missing)}; rerun ljn_ocid_py6s_correction.py with the current code"
            )
        rows = list(reader)
        nir_toa_by_sample: dict[str, float] = {}
        for row in rows:
            if row.get("band") not in {"859", "869"}:
                continue
            value = parse_float(row.get("rho_toa"))
            if value is not None:
                sample_key = f"{row.get('oc_id', '')}|{row.get('modis_datetime_utc', '')}"
                nir_toa_by_sample[sample_key] = max(value, nir_toa_by_sample.get(sample_key, value))

        for row in rows:
            counts["input_rows"] += 1
            if quality_filter:
                sensor_zenith = parse_float(row.get("sensor_zenith_deg"))
                glint_angle = parse_float(row.get("glint_angle_deg"))
                sample_key = f"{row.get('oc_id', '')}|{row.get('modis_datetime_utc', '')}"
                nir_toa = nir_toa_by_sample.get(sample_key)
                if sensor_zenith is None or sensor_zenith > max_sensor_zenith_deg:
                    counts["qc_sensor_zenith"] += 1
                    continue
                if glint_angle is None or glint_angle < min_glint_angle_deg:
                    counts["qc_glint_angle"] += 1
                    continue
                if nir_toa is not None and nir_toa > max_nir_toa:
                    counts["qc_bright_nir"] += 1
                    continue
                aod = parse_float(row.get("aod_band_6sv"))
                if max_aod is not None and aod is not None and aod > max_aod:
                    counts["qc_aod_limit"] += 1
                    continue
                distance = parse_float(row.get("distance_to_request_km"))
                if distance is None or distance > max_distance_km:
                    counts["qc_distance"] += 1
                    continue
            oc_lwn = parse_float(row.get(lwn_column))
            py6s_rho = parse_float(row.get(py6s_field))
            modis_um = parse_float(row.get("wavelength_um"))
            oc_um = parse_float(row.get("oc_wavelength_um"))
            if None in (oc_lwn, py6s_rho, modis_um, oc_um):
                counts["missing_or_invalid"] += 1
                continue
            delta_nm = abs(float(modis_um) - float(oc_um)) * 1000.0
            if delta_nm > max_wavelength_delta_nm:
                counts["wavelength_mismatch"] += 1
                continue
            try:
                solar_irradiance = interpolate_solar_irradiance(float(oc_um))
            except ValueError:
                counts["solar_irradiance_unavailable"] += 1
                continue
            oc_water_reflectance = math.pi * float(oc_lwn) / solar_irradiance
            pairs.append(
                {
                    "oc_water_reflectance": oc_water_reflectance,
                    "py6s_rho": float(py6s_rho),
                    "band": row.get("band", "unknown"),
                }
            )
            counts["valid_pairs"] += 1
    return pairs, counts


def calculate_statistics(x: list[float], y: list[float]) -> dict[str, float]:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    ss_x = sum((value - mean_x) ** 2 for value in x)
    ss_y = sum((value - mean_y) ** 2 for value in y)
    covariance = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y))
    slope = covariance / ss_x if ss_x > 0 else float("nan")
    intercept = mean_y - slope * mean_x
    r_squared = covariance**2 / (ss_x * ss_y) if ss_x > 0 and ss_y > 0 else float("nan")
    differences = [yv - xv for xv, yv in zip(x, y)]
    rmse = math.sqrt(sum(value**2 for value in differences) / n)
    bias = sum(differences) / n
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "rmse": rmse,
        "bias": bias,
    }


def plot_validation(pairs: list[dict[str, float | str]], output: Path, dpi: int, oc_product: str) -> None:
    if not pairs:
        raise ValueError("No valid reflectance pairs remain after filtering")

    x = [float(item["oc_water_reflectance"]) for item in pairs] # 结果集中用于比较的两个字段其一
    y = [float(item["py6s_rho"]) for item in pairs] # 结果集中用于比较的两个字段其二
    stats = calculate_statistics(x, y)
    lower = min(x + y)
    upper = max(x + y)
    padding = max((upper - lower) * 0.06, 0.005)
    limits = (lower - padding, upper + padding)

    figure, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    bands = sorted({str(item["band"]) for item in pairs}, key=lambda value: float(value))
    colors = plt.get_cmap("tab10")
    for index, band in enumerate(bands):
        subset = [item for item in pairs if str(item["band"]) == band]
        axis.scatter(
            [float(item["oc_water_reflectance"]) for item in subset],
            [float(item["py6s_rho"]) for item in subset],
            s=28,
            alpha=0.72,
            color=colors(index % 10),
            edgecolors="none",
            label=f"{band} nm (n={len(subset)})",
        )

    axis.plot(limits, limits, color="black", linewidth=1.2, linestyle="--", label="1:1")
    if math.isfinite(stats["slope"]):
        fit_y = [stats["slope"] * value + stats["intercept"] for value in limits]
        axis.plot(limits, fit_y, color="#c43c35", linewidth=1.5, label="Linear fit")

    annotation = (
        f"n = {len(pairs)}\n"
        f"y = {stats['slope']:.3f}x {stats['intercept']:+.4f}\n"
        f"$R^2$ = {stats['r_squared']:.3f}\n"
        f"RMSE = {stats['rmse']:.4f}\n"
        f"Bias = {stats['bias']:+.4f}"
    )
    axis.text(
        0.03,
        0.97,
        annotation,
        transform=axis.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.9, "pad": 6},
    )
    axis.set(
        title=f"AERONET-OC ({'Lwn_f/Q' if oc_product == 'fq' else 'Lwn_IOP'}) vs Py6S BOA",
        xlabel=r"AERONET-OC water-leaving reflectance  $\rho_w=\pi L_{wn}/F_0$",
        ylabel="Py6S ocean-model water-leaving reflectance",
        xlim=limits,
        ylim=limits,
        aspect="equal",
    )
    axis.grid(True, linewidth=0.6, alpha=0.25)
    axis.legend(loc="best", fontsize=8, frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Code/py6s_only/outputs/ljn_ocid_surface_reflectance.csv"),
    )
    parser.add_argument("--max-distance-km", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Code/py6s_only/outputs/aeronet_oc_vs_py6s_boa.png"),
    )
    parser.add_argument("--max-wavelength-delta-nm", type=float, default=15.0)
    parser.add_argument(
        "--oc-product",
        choices=["fq", "iop"],
        default="fq",
        help="Use oc_lwn_fq (fq) or oc_lwn_iop (iop); products are never mixed.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-quality-filter", action="store_true")
    parser.add_argument("--max-sensor-zenith-deg", type=float, default=60.0)
    parser.add_argument("--min-glint-angle-deg", type=float, default=40.0)
    parser.add_argument("--max-nir-toa", type=float, default=0.05)
    parser.add_argument("--max-aod", type=float, help="Optional AOD stratification limit; disabled by default.")
    parser.add_argument(
        "--py6s-field",
        choices=["rho_water_leaving_6sv_swir_corrected", "rho_water_leaving_6sv"],
        default="rho_water_leaving_6sv_swir_corrected",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs, counts = load_pairs(
        args.input,
        args.max_wavelength_delta_nm,
        args.oc_product,
        not args.no_quality_filter,
        args.max_sensor_zenith_deg,
        args.min_glint_angle_deg,
        args.max_nir_toa,
        args.max_aod,
        args.py6s_field,
        args.max_distance_km,
    )
    plot_validation(pairs, args.output, args.dpi, args.oc_product)
    print(f"Wrote {args.output}")
    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
