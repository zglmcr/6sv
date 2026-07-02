#!/usr/bin/env python
"""Audit surface-reflectance outputs from the Py6S-only correction workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REQUIRED_COLUMNS = {
    "sample_id",
    "band",
    "rho_toa",
    "rho_surface_minus_path",
    "rho_surface_lambertian",
    "rho_path_total",
    "rho_rayleigh",
    "rho_aerosol",
    "aod550",
    "aerosol_model",
    "time_delta_min",
    "distance_km",
}


def parse_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def read_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def add_issue(issues: List[Dict[str, str]], severity: str, sample_id: str, band: str, code: str, detail: str) -> None:
    issues.append(
        {
            "severity": severity,
            "sample_id": sample_id,
            "band": band,
            "code": code,
            "detail": detail,
        }
    )


def audit_rows(
    header: Sequence[str],
    rows: Sequence[Dict[str, str]],
    max_minutes: float,
    max_km: float,
    surface_min: float,
    surface_max: float,
    tolerance: float,
) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    issues: List[Dict[str, str]] = []
    missing = sorted(REQUIRED_COLUMNS - set(header))
    for col in missing:
        add_issue(issues, "FAIL", "", "", "missing_column", col)

    band_counter: Counter[str] = Counter()
    sample_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    numeric_ranges: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        sample_id = row.get("sample_id", "")
        band = row.get("band", "")
        band_counter[band] += 1
        sample_counter[sample_id] += 1
        model_counter[row.get("aerosol_model", "")] += 1

        rho_toa = parse_float(row.get("rho_toa", ""))
        rho_path = parse_float(row.get("rho_path_total", ""))
        rho_rayleigh = parse_float(row.get("rho_rayleigh", ""))
        rho_aerosol = parse_float(row.get("rho_aerosol", ""))
        rho_minus = parse_float(row.get("rho_surface_minus_path", ""))
        rho_lamb = parse_float(row.get("rho_surface_lambertian", ""))
        time_delta = parse_float(row.get("time_delta_min", ""))
        distance = parse_float(row.get("distance_km", ""))
        aod550 = parse_float(row.get("aod550", ""))

        values = {
            "rho_toa": rho_toa,
            "rho_path_total": rho_path,
            "rho_rayleigh": rho_rayleigh,
            "rho_aerosol": rho_aerosol,
            "rho_surface_minus_path": rho_minus,
            "rho_surface_lambertian": rho_lamb,
            "time_delta_min": time_delta,
            "distance_km": distance,
            "aod550": aod550,
        }
        for key, value in values.items():
            if value is None:
                add_issue(issues, "FAIL", sample_id, band, "invalid_numeric", key)
            else:
                numeric_ranges[key].append(value)

        if None not in (rho_path, rho_rayleigh, rho_aerosol) and abs((rho_rayleigh + rho_aerosol) - rho_path) > tolerance:
            add_issue(
                issues,
                "FAIL",
                sample_id,
                band,
                "scattering_sum_mismatch",
                f"rho_rayleigh + rho_aerosol != rho_path_total within {tolerance}",
            )
        if None not in (rho_toa, rho_path, rho_minus) and abs((rho_toa - rho_path) - rho_minus) > tolerance:
            add_issue(
                issues,
                "FAIL",
                sample_id,
                band,
                "path_subtraction_mismatch",
                f"rho_toa - rho_path_total != rho_surface_minus_path within {tolerance}",
            )
        if rho_lamb is not None and not (surface_min <= rho_lamb <= surface_max):
            add_issue(
                issues,
                "WARN",
                sample_id,
                band,
                "surface_reflectance_range",
                f"rho_surface_lambertian={rho_lamb:.6g} outside [{surface_min}, {surface_max}]",
            )
        if rho_minus is not None and not (surface_min <= rho_minus <= surface_max):
            add_issue(
                issues,
                "WARN",
                sample_id,
                band,
                "direct_subtraction_range",
                f"rho_surface_minus_path={rho_minus:.6g} outside [{surface_min}, {surface_max}]",
            )
        if time_delta is not None and time_delta > max_minutes:
            add_issue(issues, "FAIL", sample_id, band, "time_window_exceeded", f"{time_delta:.3f} > {max_minutes}")
        if distance is not None and distance > max_km:
            add_issue(issues, "FAIL", sample_id, band, "distance_window_exceeded", f"{distance:.3f} > {max_km}")
        if aod550 is not None and aod550 < 0:
            add_issue(issues, "FAIL", sample_id, band, "negative_aod550", f"{aod550:.6g}")

    ranges = {
        key: {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}
        for key, vals in numeric_ranges.items()
        if vals
    }
    summary: Dict[str, object] = {
        "rows": len(rows),
        "samples": len(sample_counter),
        "bands": dict(sorted(band_counter.items())),
        "aerosol_models": dict(model_counter),
        "issue_counts": dict(Counter(issue["severity"] for issue in issues)),
        "numeric_ranges": ranges,
    }
    return summary, issues


def write_issues(path: Path, issues: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["severity", "sample_id", "band", "code", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Surface-reflectance CSV to audit.")
    parser.add_argument("--issues-output", type=Path, help="Optional CSV of audit issues.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--max-km", type=float, default=10.0)
    parser.add_argument("--surface-min", type=float, default=-0.05)
    parser.add_argument("--surface-max", type=float, default=1.2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on WARN as well as FAIL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    header, rows = read_rows(args.input)
    summary, issues = audit_rows(
        header,
        rows,
        args.max_minutes,
        args.max_km,
        args.surface_min,
        args.surface_max,
        args.tolerance,
    )
    if args.issues_output:
        write_issues(args.issues_output, issues)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"summary": summary, "issues": issues}, indent=2), encoding="utf-8")

    print(f"Rows={summary['rows']} Samples={summary['samples']} Bands={summary['bands']}")
    print(f"Aerosol models={summary['aerosol_models']}")
    print(f"Issues={summary['issue_counts']}")
    if args.issues_output:
        print(f"Issue CSV={args.issues_output}")
    if args.json_output:
        print(f"Summary JSON={args.json_output}")

    issue_counts = summary["issue_counts"]
    fail_count = issue_counts.get("FAIL", 0) if isinstance(issue_counts, dict) else 0
    warn_count = issue_counts.get("WARN", 0) if isinstance(issue_counts, dict) else 0
    if fail_count or (args.strict and warn_count):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
