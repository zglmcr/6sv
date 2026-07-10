#!/usr/bin/env python
"""Plot MODIS match-up spectra by site: mean and +/- one standard deviation.

Accepted CSV layouts:
  wide: site, reflectance_412, reflectance_443, ...
  long: site, wavelength_nm, reflectance
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "modis_spectra_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 将数据文件中的站点全称或缩写统一为绘图所使用的站点简称。
SITE_ALIASES = {
    "AAOT": {"AAOT", "ACQUA ALTA OCEANOGRAPHIC TOWER"},
    "GDLT": {"GDLT", "GUSTAV DALEN TOWER"},
    "HLT": {"HLT", "HELSINKI LIGHTHOUSE"},
}


def normalized(value: object) -> str:
    """统一大小写以及空格、下划线和连字符，便于比较字段名和站点名。"""
    return re.sub(r"[\s_-]+", " ", str(value).strip()).upper()


def canonical_site(value: object) -> str:
    """把站点全称转换为 AAOT、GDLT、HLT 等标准名称。"""
    name = normalized(value)
    for site, aliases in SITE_ALIASES.items():
        if name in {normalized(alias) for alias in aliases}:
            return site
    return name


def finite_float(value: object) -> float | None:
    """安全读取有限浮点数；空值、非法值和无穷值均返回 None。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_oc_id(value: object) -> str:
    """标准化 oc_id，同时保留可能存在的非数字标识符。"""
    text = str(value).strip()
    # 某些 CSV 工具会把整数 ID 写成“123.0”，此处恢复为“123”。
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def detect_column(fieldnames: list[str], requested: str | None, candidates: list[str]) -> str | None:
    """优先使用用户指定列，否则从候选名称中自动识别字段。"""
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"Column {requested!r} was not found")
        return requested
    lookup = {normalized(name): name for name in fieldnames}
    for candidate in candidates:
        if normalized(candidate) in lookup:
            return lookup[normalized(candidate)]
    return None


def load_site_lookup(path: Path) -> dict[str, str]:
    """从 LWN 数据表读取唯一的 oc_id -> site 对照关系。"""
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        oc_id_col = detect_column(fields, None, ["oc_id"])
        site_col = detect_column(fields, None, ["site"])
        if oc_id_col is None or site_col is None:
            raise ValueError(f"Site lookup {path} must contain 'oc_id' and 'site' columns")

        for line_number, row in enumerate(reader, start=2):
            oc_id = normalize_oc_id(row.get(oc_id_col, ""))
            site = canonical_site(row.get(site_col, ""))
            if not oc_id or not site:
                continue
            previous = mapping.get(oc_id)
            # 同一个 oc_id 不应属于不同站点，否则无法可靠地关联 MODIS 数据。
            if previous is not None and previous != site:
                raise ValueError(
                    f"Conflicting sites for oc_id={oc_id!r} in {path}: "
                    f"{previous!r} and {site!r} (line {line_number})"
                )
            mapping[oc_id] = site
    if not mapping:
        raise ValueError(f"No valid oc_id-to-site mappings found in {path}")
    return mapping


def load_spectra(args: argparse.Namespace) -> dict[str, dict[float, list[float]]]:
    """关联站点并整理为 site -> wavelength -> reflectance values 结构。"""
    result: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    # MODIS 文件本身没有 site，因此先从 LWN 文件建立 oc_id 到 site 的索引。
    site_lookup = load_site_lookup(args.site_lookup)
    unmatched_oc_ids: set[str] = set()
    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        oc_id_col = detect_column(fields, None, ["oc_id"])
        if oc_id_col is None:
            raise ValueError(f"Input {args.input} must contain an 'oc_id' column")

        wave_col = detect_column(
            fields, args.wavelength_column, ["wavelength", "wavelength_nm", "lambda_nm", "band"]
        )
        value_col = detect_column(
            fields, args.value_column, ["reflectance", "rho", "rhow", "rrs", "modis_reflectance"]
        )
        # 宽表字段示例：reflectance_412、reflectance_443。
        wide_pattern = re.compile(rf"^{re.escape(args.wide_prefix)}(?P<wave>\d+(?:\.\d+)?)$", re.I)
        wide_cols = [(name, float(match.group("wave"))) for name in fields if (match := wide_pattern.match(name))]
        if not ((wave_col and value_col) or wide_cols):
            raise ValueError(f"No long-form spectrum or {args.wide_prefix}<wavelength> columns found")

        for row in reader:
            oc_id = normalize_oc_id(row.get(oc_id_col, ""))
            # 以 oc_id 为连接键获取该条 MODIS 匹配记录所属的站点。
            site = site_lookup.get(oc_id)
            if site is None:
                unmatched_oc_ids.add(oc_id or "<blank>")
                continue
            if wave_col and value_col:
                # 长表：每行只表示一个波长的反射率。
                wave = finite_float(row.get(wave_col))
                value = finite_float(row.get(value_col))
                if wave is not None and value is not None:
                    result[site][wave].append(value)
            else:
                # 宽表：一行中包含多个 reflectance_<波长> 字段。
                for column, wave in wide_cols:
                    value = finite_float(row.get(column))
                    if value is not None:
                        result[site][wave].append(value)
    if unmatched_oc_ids:
        # 不静默丢弃无法关联的记录，避免统计结果在不知情时发生偏差。
        examples = ", ".join(sorted(unmatched_oc_ids)[:10])
        raise ValueError(
            f"{len(unmatched_oc_ids)} oc_id value(s) from {args.input} have no site match "
            f"in {args.site_lookup}. Examples: {examples}"
        )
    return result


def plot_spectra(
    data: dict[str, dict[float, list[float]]], sites: list[str], output: Path, dpi: int
) -> None:
    """按站点计算逐波段均值和样本标准差，并绘制并排光谱图。"""
    missing = [site for site in sites if site not in data]
    if missing:
        raise ValueError(
            f"No records for site(s): {', '.join(missing)}. Available: {', '.join(sorted(data)) or 'none'}"
        )

    figure, axes = plt.subplots(1, len(sites), figsize=(4.5 * len(sites), 4.2), sharey=True)
    if len(sites) == 1:
        axes = [axes]
    color = "#1769aa"

    for axis, site in zip(axes, sites):
        waves = sorted(data[site])
        # 每个波长独立统计所有匹配样本；标准差采用样本标准差（分母 n-1）。
        means = [statistics.fmean(data[site][wave]) for wave in waves]
        # 单个样本无法计算样本标准差，绘图时将其离散程度显示为 0。
        stds = [statistics.stdev(data[site][wave]) if len(data[site][wave]) > 1 else 0.0 for wave in waves]
        upper = [mean + std for mean, std in zip(means, stds)]
        lower = [mean - std for mean, std in zip(means, stds)]
        n_max = max(len(data[site][wave]) for wave in waves)

        # 粗实线表示均值，两条粗虚线分别表示均值加减一个标准差。
        axis.plot(waves, means, color=color, linewidth=3.0, label="Mean")
        axis.plot(waves, upper, color=color, linewidth=2.2, linestyle="--", label=r"Mean $\pm$ 1 SD")
        axis.plot(waves, lower, color=color, linewidth=2.2, linestyle="--")
        axis.set_title(f"{site} (max n={n_max})")
        axis.set_xlabel("Wavelength (nm)")
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.tick_params(direction="in", top=True, right=True)

    axes[0].set_ylabel("MODIS reflectance")
    axes[-1].legend(frameon=False)
    figure.suptitle("MODIS match-up spectra")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """解析命令行参数，读取数据并生成图片。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default = Path("Data/ljn/modis_l1b_result.csv"))
    parser.add_argument(
        "--site-lookup",
        type=Path,
        default=Path("Data/ljn/lwn_with_aod_inv15_ocid.csv"),
        help="CSV containing the oc_id and site columns",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("Code/py6s_only/outputs/modis_matchup_spectra.png"))
    parser.add_argument("--sites", nargs="+", default=["AAOT", "GDLT", "HLT"])
    parser.add_argument("--wavelength-column")
    parser.add_argument("--value-column")
    parser.add_argument("--wide-prefix", default="reflectance_")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data = load_spectra(args)
    sites = [canonical_site(site) for site in args.sites] # 默认仅绘制三个站点
    plot_spectra(data, sites, args.output, args.dpi)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
