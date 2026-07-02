#!/usr/bin/env python
"""Download AERONET AOD data from the public Web Services endpoint.

This is a small standalone helper for the Py6S-only workflow.  It downloads
AERONET AOD records, normalizes the date/time fields, converts AERONET missing
values to blanks, and writes a CSV that can be passed to
`ljn_ocid_py6s_correction.py --aeronet-csv`.

It does not download SDA fine/coarse products.  Keep using an existing SDA CSV
for `--sda-csv`, or add a separate SDA downloader later.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


AERONET_ENDPOINT = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
MISSING_VALUES = {"-999", "-999.0", "-999.000000", "N/A", "NaN", "nan"}


def level_code(level: str) -> str:
    """Convert user level input to the AERONET URL code."""

    normalized = str(level).strip().lower().replace("level", "").replace(" ", "")
    table = {
        "1": "10",
        "1.0": "10",
        "10": "10",
        "1.5": "15",
        "15": "15",
        "2": "20",
        "2.0": "20",
        "20": "20",
    }
    if normalized not in table:
        raise ValueError(f"Unsupported AERONET level: {level}. Use 1.0, 1.5, or 2.0.")
    return table[normalized]


def parse_yyyymmdd(value: str) -> datetime:
    """Parse YYYYMMDD from command line."""

    return datetime.strptime(value, "%Y%m%d")


def build_url(site: Optional[str], start: datetime, end: datetime, level: str, avg: int) -> str:
    """Build a public AERONET Web Services URL for AOD records."""

    code = level_code(level)
    query: Dict[str, str] = {
        "year": f"{start.year:04d}",
        "month": f"{start.month:02d}",
        "day": f"{start.day:02d}",
        "year2": f"{end.year:04d}",
        "month2": f"{end.month:02d}",
        "day2": f"{end.day:02d}",
        f"AOD{code}": "1",
        "AVG": str(avg),
    }
    if site:
        query["site"] = site
    return f"{AERONET_ENDPOINT}?{urllib.parse.urlencode(query)}"


def download_text(url: str, timeout: float) -> str:
    """Download text from AERONET using urllib from the Python standard library."""

    request = urllib.request.Request(url, headers={"User-Agent": "py6s-only-aeronet-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="replace")


def strip_aeronet_preamble(text: str) -> str:
    """Remove AERONET text before the CSV header row."""

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("AERONET_Site") or line.startswith("Date(dd:mm:yyyy)"):
            return "\n".join(lines[idx:])
    raise ValueError("Could not find an AERONET CSV header in the downloaded response.")


def normalize_missing(value: str) -> str:
    """Convert AERONET missing-value markers to empty CSV fields."""

    text = (value or "").strip()
    if text in MISSING_VALUES:
        return ""
    return text


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    """Normalize one downloaded AERONET row for local Py6S processing."""

    out = {key: normalize_missing(value) for key, value in row.items() if key is not None}
    date_text = out.get("Date(dd:mm:yyyy)", "")
    time_text = out.get("Time(hh:mm:ss)", "")
    if date_text and time_text:
        day, month, year = date_text.split(":")
        out["DateTime_UTC"] = f"{year}-{month}-{day}T{time_text}Z"
    elif date_text:
        day, month, year = date_text.split(":")
        out["DateTime_UTC"] = f"{year}-{month}-{day}T00:00:00Z"
    if "AERONET_Site" in out and "AERONET_Site_Name" not in out:
        out["AERONET_Site_Name"] = out["AERONET_Site"]
    if "Site_Latitude(Degrees)" not in out and "Site_Latitude" in out:
        out["Site_Latitude(Degrees)"] = out["Site_Latitude"]
    if "Site_Longitude(Degrees)" not in out and "Site_Longitude" in out:
        out["Site_Longitude(Degrees)"] = out["Site_Longitude"]
    return out


def parse_rows(text: str) -> List[Dict[str, str]]:
    """Parse AERONET CSV text into normalized row dictionaries."""

    csv_text = strip_aeronet_preamble(text)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = [normalize_row(row) for row in reader if any((value or "").strip() for value in row.values())]
    if not rows:
        raise ValueError("Downloaded AERONET response did not contain data rows.")
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Write rows to CSV using the union of all observed fields."""

    preferred = [
        "AERONET_Site_Name",
        "DateTime_UTC",
        "Date(dd:mm:yyyy)",
        "Time(hh:mm:ss)",
        "Site_Latitude(Degrees)",
        "Site_Longitude(Degrees)",
        "Site_Elevation(m)",
        "Precipitable_Water(cm)",
        "Ozone(Dobson)",
        "NO2(Dobson)",
        "440-870_Angstrom_Exponent",
    ]
    fields: List[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            fields.append(name)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", help="Optional AERONET site name, e.g. AAOT. Omit to request all sites.")
    parser.add_argument("--start", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument("--level", default="1.5", help="AERONET AOD level: 1.0, 1.5, or 2.0. Default: 1.5.")
    parser.add_argument("--avg", type=int, default=20, help="AERONET AVG code. Use 20 for site time series, 10 for broad map-style queries.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--print-url", action="store_true", help="Print the generated URL before downloading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = parse_yyyymmdd(args.start)
    end = parse_yyyymmdd(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")
    url = build_url(args.site, start, end, args.level, args.avg)
    if args.print_url:
        print(url)
    text = download_text(url, args.timeout)
    rows = parse_rows(text)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} AERONET rows to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

