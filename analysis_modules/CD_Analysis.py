# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 14:25:34 2025

@author: JoeCh
"""

# CD_Analysis.py
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys
from itertools import combinations
from scipy.stats import ks_2samp
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import re


# ──────────────────────────────────────────────────────────────────────────────
# CD_DATA INDEX
# ──────────────────────────────────────────────────────────────────────────────

def _clean_fname(raw_fname):
    """Mirror of Raw_Trace_Module.clean_fname — extract a canonical cell ID."""
    raw_fname = str(raw_fname)
    match = re.match(r"(JC20\d{8}[a-z])", raw_fname, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return raw_fname.split(" ")[0].lower()


def _build_cd_index(cd_folder):
    index = {}

    try:
        xlsx_files = [f for f in os.listdir(cd_folder) if f.lower().endswith(".xlsx")]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CD_Data folder not found:\n{cd_folder}\n\n"
            "Create it and place your CD Excel files inside."
        )

    if not xlsx_files:
        print(f"⚠️  No .xlsx files found in {cd_folder}")
        return index

    for fname in xlsx_files:
        fpath = os.path.join(cd_folder, fname)
        try:
            df_cd = pd.read_excel(fpath, header=None)
        except Exception as e:
            print(f"⚠️  Could not read {fname}: {e}")
            continue

        if df_cd.empty:
            continue

        for col_idx in range(df_cd.shape[1]):
            raw_header = str(df_cd.iloc[0, col_idx])
            cell_key   = _clean_fname(raw_header)
            if not cell_key:
                continue

            col_series = pd.to_numeric(
                df_cd[col_idx].iloc[5:], errors="coerce"
            ).dropna()

            if col_series.empty:
                print(f"  Skipping column '{raw_header}' in {fname} "
                      "— no numeric data after row 4")
                continue

            col_data = col_series.tolist()

            if cell_key in index:
                print(f"⚠️  Duplicate cell ID '{cell_key}' found in {fname} "
                      f"(previously seen in {os.path.basename(index[cell_key][0])}). "
                      "Using the later occurrence.")

            index[cell_key] = (fpath, col_idx, col_data)

    print(f"  CD index built: {len(index)} cell(s) across {len(xlsx_files)} file(s)")
    return index


def _lookup_cell(cd_index, raw_fname, sample_size, missing_log=None,
                 context=None):
    query = _clean_fname(raw_fname)

    def _log_miss(reason):
        if missing_log is not None:
            entry = {"cell_id": query, "reason": reason}
            if context:
                entry.update(context)
            missing_log.append(entry)

    if query in cd_index:
        _, _, col_data = cd_index[query]
        if len(col_data) >= sample_size:
            return col_data
        msg = f"found but only {len(col_data)} values (need {sample_size})"
        print(f"    '{query}' {msg} — skipped")
        _log_miss(msg)
        return None

    candidates = [k for k in cd_index if k.startswith(query) or query.startswith(k)]
    if candidates:
        key = candidates[0]
        _, _, col_data = cd_index[key]
        if len(col_data) >= sample_size:
            return col_data
        msg = f"prefix-matched '{key}' but only {len(col_data)} values"
        print(f"    '{query}' {msg} — skipped")
        _log_miss(msg)
        return None

    print(f"    '{query}' NOT found in CD index")
    _log_miss("not found in CD index")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# COLOR VARIATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def _color_to_rgb(color):
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return tuple(float(c) for c in color[:3])
    if isinstance(color, str):
        if color.startswith("#"):
            return _hex_to_rgb(color)
        import matplotlib.colors as mcolors
        return mcolors.to_rgb(color)
    return (0.3, 0.3, 0.3)


def _vary_lightness(base_color, n_groups):
    if n_groups == 1:
        return [base_color]

    r, g, b = _color_to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    l_min = max(0.20, l - 0.35)
    l_max = min(0.85, l + 0.35)
    lightness_values = np.linspace(l_min, l_max, n_groups)

    colours = []
    for lv in lightness_values:
        rr, gg, bb = colorsys.hls_to_rgb(h, lv, s)
        colours.append((rr, gg, bb))
    return colours


def _assign_group_colors(groups, base_color_map, use_lesion_color=False):
    if not groups:
        return {}

    if use_lesion_color:
        base = list(base_color_map.values())[0] if base_color_map else "steelblue"
        variants = _vary_lightness(base, len(groups))
        return {g: c for g, c in zip(groups, variants)}
    else:
        result = {}
        for g in groups:
            bare = g.split("-")[-1]
            raw  = (base_color_map or {}).get(bare, "gray")
            result[g] = _color_to_rgb(raw)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# MISSING CELLS EXCEL
# ──────────────────────────────────────────────────────────────────────────────

def _write_missing_cells_excel(missing_log, output_path):
    if not missing_log:
        print("  No missing cells — skipping missing-cells Excel.")
        return

    wb = Workbook()

    hdr_fill = PatternFill("solid", start_color="2F5496", end_color="2F5496")
    hdr_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    body_font = Font(name="Arial", size=10)
    center = Alignment(horizontal="center")
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    warn_fill = PatternFill("solid", start_color="FFC7CE", end_color="FFC7CE")
    alt_fill  = PatternFill("solid", start_color="F2F2F2", end_color="F2F2F2")

    def _style_header(ws, headers):
        for col_i, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col_i, value=h)
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.alignment = center
            cell.border = border
            ws.column_dimensions[get_column_letter(col_i)].width = max(len(h) + 4, 18)

    def _style_row(ws, row_i, values, use_alt):
        fill = alt_fill if use_alt else None
        for col_i, val in enumerate(values, start=1):
            cell = ws.cell(row=row_i, column=col_i, value=val)
            cell.font = body_font
            cell.alignment = center
            cell.border = border
            if fill:
                cell.fill = fill

    ws1 = wb.active
    ws1.title = "All Missing"

    all_keys = ["cell_id", "reason", "training_group", "lesion", "depth", "section"]
    extra_keys = sorted({k for e in missing_log for k in e if k not in all_keys})
    headers1 = all_keys + extra_keys

    _style_header(ws1, headers1)

    for row_i, entry in enumerate(missing_log, start=2):
        values = [entry.get(k, "") for k in headers1]
        _style_row(ws1, row_i, values, use_alt=(row_i % 2 == 0))
        if "not found" in str(entry.get("reason", "")):
            for col_i in range(1, len(headers1) + 1):
                ws1.cell(row=row_i, column=col_i).fill = warn_fill

    ws1.freeze_panes = "A2"
    ws1.auto_filter.ref = f"A1:{get_column_letter(len(headers1))}1"

    ws2 = wb.create_sheet(title="Unique Cell IDs")

    from collections import defaultdict
    by_id = defaultdict(lambda: {"reasons": set(), "training_groups": set(),
                                  "lesions": set(), "depths": set()})
    for entry in missing_log:
        cid = entry.get("cell_id", "")
        by_id[cid]["reasons"].add(entry.get("reason", ""))
        by_id[cid]["training_groups"].add(str(entry.get("training_group", "")))
        by_id[cid]["lesions"].add(str(entry.get("lesion", "")))
        by_id[cid]["depths"].add(str(entry.get("depth", "")))

    headers2 = ["cell_id", "reason(s)", "training_group(s)",
                 "lesion(s)", "depth(s)", "n_lookups"]
    _style_header(ws2, headers2)

    lookup_counts = {}
    for entry in missing_log:
        cid = entry.get("cell_id", "")
        lookup_counts[cid] = lookup_counts.get(cid, 0) + 1

    for row_i, (cid, info) in enumerate(
            sorted(by_id.items()), start=2):
        values = [
            cid,
            "; ".join(sorted(r for r in info["reasons"] if r)),
            "; ".join(sorted(g for g in info["training_groups"] if g)),
            "; ".join(sorted(l for l in info["lesions"] if l)),
            "; ".join(sorted(d for d in info["depths"] if d)),
            lookup_counts.get(cid, 1),
        ]
        _style_row(ws2, row_i, values, use_alt=(row_i % 2 == 0))
        if "not found" in values[1]:
            for col_i in range(1, len(headers2) + 1):
                ws2.cell(row=row_i, column=col_i).fill = warn_fill

    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = f"A1:{get_column_letter(len(headers2))}1"

    for ws in (ws1, ws2):
        ws.column_dimensions["A"].width = 22

    wb.save(output_path)
    n_unique = len(by_id)
    print(f"  Missing cells Excel saved: {output_path} "
          f"({len(missing_log)} lookups, {n_unique} unique cell IDs)")


# ──────────────────────────────────────────────────────────────────────────────
# LESION COLORS
# ──────────────────────────────────────────────────────────────────────────────

def load_lesion_colors(filepath):
    lesion_colors = {}
    if not os.path.exists(filepath):
        print(f"WARNING: lesion color file not found: {filepath}")
        return lesion_colors
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            lesion_colors[key.strip()] = val.strip()
    print(f"Loaded lesion colors: {lesion_colors}")
    return lesion_colors


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY PNG
# ──────────────────────────────────────────────────────────────────────────────

def _write_summary_png(summary_records, output_path):
    if not summary_records:
        print("  No summary records to write.")
        return

    headers = [
        "Figure Name", "Comparison", "Lesion", "Depth",
        "Control Group", "Test Group",
        "N Ctrl", "N Test",
        "Med. KS", "Med. p",
        "% p<0.05", "% p<0.01",
        "Rep. p", "Sig?"
    ]

    rows = []
    for rec in summary_records:
        rep_p = rec.get("representative_p", 1)
        rows.append([
            str(rec.get("fig_name", "")),
            str(rec.get("comparison", "")),
            str(rec.get("lesion", "")),
            str(rec.get("depth", "")),
            str(rec.get("control_group", "")),
            str(rec.get("test_group", "")),
            str(rec.get("n_control", "")),
            str(rec.get("n_test", "")),
            f"{rec.get('median_ks_stat', 0):.4f}",
            f"{rec.get('median_p', 0):.4f}",
            f"{rec.get('pct_sig_05', 0):.1f}",
            f"{rec.get('pct_sig_01', 0):.1f}",
            f"{rep_p:.4f}",
            "Yes" if rep_p < 0.05 else "No",
        ])

    n_rows  = len(rows)
    n_cols  = len(headers)
    row_h   = 0.35
    col_w   = 1.45
    fig_w   = n_cols * col_w
    fig_h   = (n_rows + 1) * row_h + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)

    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2F5496")
        cell.set_text_props(color="white", fontweight="bold")

    sig_col = n_cols - 1
    for row_idx, rec_row in enumerate(rows, start=1):
        is_sig = rec_row[sig_col] == "Yes"
        for col_idx in range(n_cols):
            cell = tbl[row_idx, col_idx]
            if col_idx == sig_col:
                cell.set_facecolor("#C6EFCE" if is_sig else "#FFC7CE")
            else:
                cell.set_facecolor("#F2F2F2" if row_idx % 2 == 0 else "white")

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Summary PNG saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CDH DATA EXCEL  ← UPDATED: writes raw values from representative iteration
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_sheet_name(name):
    """Strip characters Excel forbids in sheet names."""
    invalid = ['\\', '/', '*', '?', ':', '[', ']']
    cleaned = str(name)
    for ch in invalid:
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip() or "Sheet"
    return cleaned


def _make_unique_sheet_names(base_names, max_len=31):
    """
    Given an ordered list of desired (unsanitized) base names, return a list
    of unique, Excel-safe sheet names of the same length/order.

    Duplicates are disambiguated as Base_1, Base_2, ... in order of
    appearance, truncated to fit Excel's 31-character sheet-name limit.
    """
    sanitized = [_sanitize_sheet_name(n) for n in base_names]

    counts = {}
    for n in sanitized:
        counts[n] = counts.get(n, 0) + 1

    seen = {}
    result = []
    for n in sanitized:
        if counts[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            suffix = f"_{seen[n]}"
            truncated = n[: max_len - len(suffix)]
            final = f"{truncated}{suffix}"
        else:
            final = n[:max_len]
        result.append(final)

    return result


def _write_cdh_data_excel(cdh_records, output_path):
    """
    Write raw resampled values from the single representative bootstrap
    iteration to Excel.

    cdh_records structure (updated from original):
        {
          fig_name: {
              "rep_info": {
                  pair_tuple: {
                      "rep_idx":    int,          # which bootstrap iteration
                      "rep_p":      float,         # p-value of that iteration
                      "median_p":   float,         # median p across all iters
                      "samp_A":     np.ndarray,    # raw values group A
                      "samp_B":     np.ndarray,    # raw values group B
                      "group_A":    str,
                      "group_B":    str,
                  }
              },
              "all_group_samples": {
                  group_name: np.ndarray   # raw values for that group
                                           # from the representative iteration
              },
              "lesion_label": str,   # e.g. lesion tag, or training group for
                                     # lesion-vs-lesion sheets — used to build
                                     # the short sheet-tab name
              "layer_label":  str,   # e.g. "L2"/"L3"/"All" — used to build
                                     # the short sheet-tab name
          }
        }

    Sheet layout per figure:
      - Row 1: the full descriptive name (previously used as the sheet tab
        name) written as a title cell.
      - A metadata block (starting row 3) showing rep_idx, rep_p, median_p
        for each pairwise comparison.
      - Raw data follows, one column per group.

    The sheet TAB name is now the short "Lesion_Layer" combination instead
    of the full descriptive name (which could exceed Excel's 31-character
    limit). If two sheets would end up with the same short name, they are
    disambiguated as Lesion_Layer_1, Lesion_Layer_2, etc.
    """
    wb    = Workbook()
    first = True

    hdr_fill  = PatternFill("solid", start_color="2F5496", end_color="2F5496")
    meta_fill = PatternFill("solid", start_color="D9E1F2", end_color="D9E1F2")
    title_font = Font(name="Arial", bold=True, size=11)
    hdr_font  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    meta_font = Font(name="Arial", bold=True, size=9)
    body_font = Font(name="Arial", size=10)
    center    = Alignment(horizontal="center")
    left      = Alignment(horizontal="left")
    thin      = Side(style="thin", color="CCCCCC")
    border    = Border(left=thin, right=thin, top=thin, bottom=thin)
    alt_fill  = PatternFill("solid", start_color="F2F2F2", end_color="F2F2F2")

    fig_names  = list(cdh_records.keys())
    base_names = [
        f"{data.get('lesion_label', '') or 'All'}_{data.get('layer_label', '') or 'All'}"
        for data in cdh_records.values()
    ]
    sheet_names = _make_unique_sheet_names(base_names)

    for fig_name, sheet_name in zip(fig_names, sheet_names):
        data = cdh_records[fig_name]

        if first:
            ws    = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(title=sheet_name)

        rep_info         = data.get("rep_info", {})
        all_group_samples = data.get("all_group_samples", {})

        # ── Row 1: full descriptive name (formerly the sheet tab name) ───────
        title_cell = ws.cell(row=1, column=1, value=fig_name)
        title_cell.font      = title_font
        title_cell.alignment = left

        # ── Metadata rows: one row per pairwise comparison, starting row 3 ───
        meta_headers = ["Comparison", "Rep. Iteration #", "Rep. p-value",
                        "Median p-value", "Δ |rep_p − median_p|"]
        meta_header_row = 3
        for col_i, h in enumerate(meta_headers, start=1):
            c = ws.cell(row=meta_header_row, column=col_i, value=h)
            c.font      = meta_font
            c.fill      = meta_fill
            c.alignment = center
            c.border    = border
            ws.column_dimensions[get_column_letter(col_i)].width = max(len(h) + 4, 20)

        for offset, (pair, pinfo) in enumerate(rep_info.items()):
            meta_row = meta_header_row + 1 + offset
            gA, gB = pair
            delta  = abs(pinfo["rep_p"] - pinfo["median_p"])
            row_vals = [
                f"{gA} vs {gB}",
                pinfo["rep_idx"],
                round(pinfo["rep_p"],    6),
                round(pinfo["median_p"], 6),
                round(delta,             6),
            ]
            for col_i, val in enumerate(row_vals, start=1):
                c = ws.cell(row=meta_row, column=col_i, value=val)
                c.font      = body_font
                c.alignment = center
                c.border    = border

        # Blank separator row (row after last meta row)
        separator_row   = meta_header_row + 1 + len(rep_info)
        data_header_row = separator_row + 1         # raw-data header row
        data_start_row  = data_header_row + 1       # first value row

        # ── Raw data header row ───────────────────────────────────────────────
        groups = list(all_group_samples.keys())
        for col_i, g in enumerate(groups, start=1):
            c = ws.cell(row=data_header_row, column=col_i, value=g)
            c.font      = hdr_font
            c.fill      = hdr_fill
            c.alignment = center
            c.border    = border
            ws.column_dimensions[get_column_letter(col_i)].width = max(len(g) + 4, 18)

        # ── Raw values ────────────────────────────────────────────────────────
        max_len = max((len(v) for v in all_group_samples.values()), default=0)
        for col_i, g in enumerate(groups, start=1):
            vals = list(all_group_samples[g])
            for row_i in range(max_len):
                excel_row = data_start_row + row_i
                val = round(float(vals[row_i]), 6) if row_i < len(vals) else None
                c = ws.cell(row=excel_row, column=col_i, value=val)
                c.font      = body_font
                c.alignment = center
                c.border    = border
                if row_i % 2 == 1:
                    c.fill = alt_fill

        ws.freeze_panes = f"A{data_start_row}"

    wb.save(output_path)
    print(f"  CDH raw-data Excel saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _run_bootstrap(flat_group_data, x_values, n_bootstrap):
    """
    Run bootstrap resampling and compute KS stats for ALL pairwise comparisons.

    Returns
    -------
    bootstrap_cdfs : dict  { group_name: ndarray (n_bootstrap, len(x_values)) }
    bootstrap_ks   : dict  { (groupA, groupB): list of (ks_stat, p_val, samp_A, samp_B) }
    """
    groups         = list(flat_group_data.keys())
    all_pairs      = list(combinations(groups, 2))
    bootstrap_cdfs = {g: [] for g in groups}
    bootstrap_ks   = {pair: [] for pair in all_pairs}

    for _ in range(n_bootstrap):
        boot_samples = {}
        for g, vals in flat_group_data.items():
            samp = np.random.choice(vals, size=len(vals), replace=True)
            boot_samples[g] = samp
            cdf = np.searchsorted(np.sort(samp), x_values, side="right") / len(samp)
            bootstrap_cdfs[g].append(cdf)

        for (gA, gB) in all_pairs:
            ks_stat, p_val = ks_2samp(boot_samples[gA], boot_samples[gB])
            bootstrap_ks[(gA, gB)].append(
                (ks_stat, p_val, boot_samples[gA].copy(), boot_samples[gB].copy())
            )

    for g in bootstrap_cdfs:
        bootstrap_cdfs[g] = np.array(bootstrap_cdfs[g])

    return bootstrap_cdfs, bootstrap_ks


def _compute_representative_info(bootstrap_ks, n_bootstrap):
    """
    Summarise each pairwise comparison from bootstrap_ks.

    Returns
    -------
    dict  { (groupA, groupB): { p_val, idx, median_p, median_ks,
                                 pct_sig_05, pct_sig_01, p_values_boot } }
    """
    representative_info = {}
    for pair, entries in bootstrap_ks.items():
        if not entries:
            continue
        p_values_boot = np.array([x[1] for x in entries])
        ks_stats_boot = np.array([x[0] for x in entries])
        median_p      = float(np.median(p_values_boot))
        median_ks     = float(np.median(ks_stats_boot))
        pct_sig_05    = float(np.mean(p_values_boot < 0.05) * 100)
        pct_sig_01    = float(np.mean(p_values_boot < 0.01) * 100)
        closest_idx   = int(np.argmin(np.abs(p_values_boot - median_p)))
        representative_info[pair] = {
            "p_val":         float(p_values_boot[closest_idx]),
            "idx":           closest_idx,
            "median_p":      median_p,
            "median_ks":     median_ks,
            "pct_sig_05":    pct_sig_05,
            "pct_sig_01":    pct_sig_01,
            "p_values_boot": p_values_boot,
        }
    return representative_info


def _extract_representative_samples(bootstrap_ks, representative_info):
    """
    For each pair, pull the raw resampled arrays from the representative
    iteration (the one whose p-value is closest to the median p-value).

    Returns
    -------
    rep_info_with_samples : dict
        { (groupA, groupB): {
              "rep_idx":  int,
              "rep_p":    float,
              "median_p": float,
              "samp_A":   np.ndarray,
              "samp_B":   np.ndarray,
              "group_A":  str,
              "group_B":  str,
          } }

    all_group_samples : dict
        { group_name: np.ndarray }
        Raw values for EVERY group from the first pair's representative
        iteration.  When multiple pairs exist (>2 groups) each pair may
        choose a slightly different iteration; we use the iteration chosen
        by the FIRST pair as the single representative snapshot for all
        groups, so the columns are internally consistent.
    """
    rep_info_with_samples = {}
    # We'll collect per-group samples keyed by iteration index so we can
    # reconstruct a consistent snapshot.
    # bootstrap_ks stores (ks_stat, p_val, samp_A, samp_B) per iteration.

    # Build a lookup: pair -> list of (samp_A, samp_B) indexed by iteration
    pair_samples = {pair: [(e[2], e[3]) for e in entries]
                    for pair, entries in bootstrap_ks.items()}

    for pair, info in representative_info.items():
        gA, gB  = pair
        rep_idx = info["idx"]
        samp_A, samp_B = pair_samples[pair][rep_idx]
        rep_info_with_samples[pair] = {
            "rep_idx":  rep_idx,
            "rep_p":    info["p_val"],
            "median_p": info["median_p"],
            "samp_A":   samp_A,
            "samp_B":   samp_B,
            "group_A":  gA,
            "group_B":  gB,
        }

    # Build all_group_samples from the first pair's representative iteration.
    # For >2 groups we reconstruct every group's sample from its pair entries.
    # Strategy: use the iteration index of the LEXICOGRAPHICALLY first pair.
    all_group_samples = {}
    if rep_info_with_samples:
        first_pair = next(iter(rep_info_with_samples))
        anchor_idx = rep_info_with_samples[first_pair]["rep_idx"]

        # Collect every group mentioned across all pairs
        all_groups = set()
        for (gA, gB) in bootstrap_ks:
            all_groups.add(gA)
            all_groups.add(gB)

        # For each group, find a pair that contains it and grab that
        # group's sample at anchor_idx.
        for g in all_groups:
            for (gA, gB), entries in bootstrap_ks.items():
                if gA == g:
                    all_group_samples[g] = entries[anchor_idx][2]  # samp_A
                    break
                if gB == g:
                    all_group_samples[g] = entries[anchor_idx][3]  # samp_B
                    break

    return rep_info_with_samples, all_group_samples


def _save_histogram(fig_name, pair, info,
                    n_bootstrap, output_folder, dpi, bar_color="steelblue"):
    gA, gB        = pair
    p_values_boot = info["p_values_boot"]
    median_p      = info["median_p"]
    pct_sig_05    = info["pct_sig_05"]

    hist_fig, hist_ax = plt.subplots(figsize=(1.82, 1.82), dpi=dpi)
    hist_ax.hist(p_values_boot, bins=50, color=bar_color,
                 edgecolor="white", linewidth=0.3)
    hist_ax.axvline(0.05,     color="black", linestyle="--", linewidth=0.5, label="p=0.05")
    hist_ax.axvline(median_p, color="red",   linestyle=":",  linewidth=0.5, label="Median")
    hist_ax.axvline(color="black", linestyle="", linewidth=0.1,
                    label=f"p<0.05: {pct_sig_05:.1f}%")

    hist_ax.set_xlabel("KS p-value", fontsize=6)
    hist_ax.set_ylabel("Count",      fontsize=8)
    hist_ax.yaxis.set_label_position("right")
    hist_ax.yaxis.tick_right()
    hist_ax.set_title(f"{fig_name}\n{gA} vs {gB}", fontsize=6)
    hist_ax.spines["top"].set_visible(False)
    hist_ax.spines["left"].set_visible(False)
    hist_ax.spines["right"].set_visible(True)
    hist_ax.tick_params(labelsize=6, width=1.0, length=4)
    hist_ax.legend(fontsize=4, frameon=False)
    hist_fig.tight_layout()

    pair_tag  = f"{gA}_vs_{gB}"
    hist_key  = f"{fig_name}__hist__{pair_tag}"
    hist_path = os.path.join(output_folder, f"{hist_key}.png")
    hist_fig.savefig(hist_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(hist_fig)
    print(f"  Histogram saved: {hist_path}")
    return hist_key, hist_fig


# ──────────────────────────────────────────────────────────────────────────────
# SHARED DATA LOADER
# ──────────────────────────────────────────────────────────────────────────────

def _load_group_data(subset, filename_col, cd_index, sample_size,
                     label="", missing_log=None, context=None):
    cell_data = {}
    for _, row in subset.iterrows():
        raw_fname = str(row[filename_col])
        col_data  = _lookup_cell(cd_index, raw_fname, sample_size,
                                 missing_log=missing_log, context=context)
        if col_data is not None:
            cell_data[raw_fname] = col_data
    if label:
        print(f"  [{label}] {len(cell_data)} cell(s) found in CD index")
    return cell_data


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _is_numeric(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_cd_analysis(
        df_group,
        comparisons,
        measurement_col,
        main_file=None,
        cd_folder="CD_Data",
        depth_col=None,
        depth_threshold=None,
        training_col=None,
        filename_col=None,
        lesion_col=None,
        group_colors=None,
        sample_size=25,
        dpi=600,
        n_bootstrap=5000,
        fig_size=(1.82, 1.82),
        output_folder="./Plots"):

    figs             = {}
    group_data_out   = {}
    summary_records  = []
    cdh_data_records = {}   # ← now stores rep_info + all_group_samples

    missing_log = []

    os.makedirs(output_folder, exist_ok=True)

    cd_index = _build_cd_index(cd_folder)
    if not cd_index:
        print("⚠️  CD index is empty — no cell data available. Aborting.")
        return figs, group_data_out

    split_by_depth = (depth_col and depth_col in df_group.columns
                      and not df_group[depth_col].dropna().empty)
    use_lesion     = bool(lesion_col and lesion_col in df_group.columns)

    print(f"\n=== CD ANALYSIS ===")
    print(f"use_lesion={use_lesion} | lesion_col={lesion_col}")
    print(f"split_by_depth={split_by_depth} | depth_col={depth_col}")
    print(f"comparisons={comparisons}")

    df_group = df_group.copy()
    if split_by_depth and depth_threshold is not None:
        df_group["DepthLabel"] = df_group[depth_col].apply(
            lambda x: "L2" if x < depth_threshold else "L3"
        )
    else:
        df_group["DepthLabel"] = "All"

    lesion_color_map = {}
    if use_lesion:
        lesion_color_map = load_lesion_colors("lesion_colors.txt")
        if not lesion_color_map:
            lesion_values = sorted(df_group[lesion_col].dropna().unique().tolist(), key=str)
            cmap = plt.cm.get_cmap("tab10", max(len(lesion_values), 1))
            lesion_color_map = {str(lv): cmap(i) for i, lv in enumerate(lesion_values)}

    # ════════════════════════════════════════════════════════════════════════ #
    # SECTION A — training-group comparisons                                  #
    # ════════════════════════════════════════════════════════════════════════ #
    for comp in comparisons:
        lesion_groups = (df_group[lesion_col].dropna().unique()
                         if use_lesion else [None])

        for lesion in lesion_groups:
            lesion_tag   = str(lesion) if lesion is not None else None
            depth_labels = ["L2", "L3"] if split_by_depth else [None]

            for depth_label in depth_labels:
                print(f"\n--- comp={comp} | lesion={lesion_tag} | depth={depth_label} ---")

                comp_group_data = {}

                for g in comp:
                    subset = df_group.copy()

                    if lesion_tag:
                        subset = subset[subset[lesion_col].astype(str) == lesion_tag]

                    train_name = g.split("-")[-1]
                    subset = subset[subset[training_col].astype(str) == train_name]

                    if split_by_depth and depth_label:
                        subset = subset[subset["DepthLabel"] == depth_label]

                    if subset.empty:
                        print(f"  [{g}] empty subset — skipped")
                        continue

                    ctx = {
                        "section":        "group_comparison",
                        "training_group": g,
                        "lesion":         lesion_tag or "",
                        "depth":          depth_label or "",
                    }
                    cell_data = _load_group_data(
                        subset, filename_col, cd_index, sample_size,
                        label=g, missing_log=missing_log, context=ctx)

                    if cell_data:
                        comp_group_data[g] = cell_data

                print(f"  Groups with data: {list(comp_group_data.keys())}")

                if len(comp_group_data) < 2:
                    print("  SKIPPED: fewer than 2 groups have data")
                    continue

                flat_group_data = {}
                for g, cells in comp_group_data.items():
                    flat = []
                    for fname, vals in cells.items():
                        if len(vals) >= sample_size:
                            flat.extend(random.sample(vals, sample_size))
                    flat_group_data[g] = np.array(flat)
                    print(f"  [{g}] flat size: {len(flat_group_data[g])}")

                if len([v for v in flat_group_data.values() if len(v) > 0]) < 2:
                    print("  SKIPPED: fewer than 2 non-empty groups")
                    continue

                all_vals = np.concatenate(list(flat_group_data.values()))
                x_values = np.linspace(all_vals.min(), all_vals.max(), 200)

                bootstrap_cdfs, bootstrap_ks = _run_bootstrap(
                    flat_group_data, x_values, n_bootstrap)

                fig_name = " vs ".join(comp)
                if lesion_tag:
                    fig_name += f"_{lesion_tag}"
                if depth_label:
                    fig_name += f"_{depth_label}"

                representative_info = _compute_representative_info(
                    bootstrap_ks, n_bootstrap)

                # ── Extract raw samples from representative iteration ─────────
                rep_info_with_samples, all_group_samples = \
                    _extract_representative_samples(bootstrap_ks, representative_info)

                groups_in_fig = list(flat_group_data.keys())
                if use_lesion and lesion_tag:
                    base_lesion_color = lesion_color_map.get(str(lesion_tag).strip(), "gray")
                    variants = _vary_lightness(base_lesion_color, len(groups_in_fig))
                    group_color_assigned = {g: c for g, c in zip(groups_in_fig, variants)}
                else:
                    group_color_assigned = {}
                    for g in groups_in_fig:
                        bare = g.split("-")[-1]
                        raw  = (group_colors or {}).get(bare, "gray")
                        group_color_assigned[g] = _color_to_rgb(raw)

                for pair, info in representative_info.items():
                    gA, gB    = pair
                    bar_color = group_color_assigned.get(gB, group_color_assigned.get(gA, "gray"))
                    _save_histogram(fig_name, pair, info,
                                    n_bootstrap, output_folder, dpi,
                                    bar_color=bar_color)

                    summary_records.append({
                        "fig_name":         fig_name,
                        "comparison":       " vs ".join(comp),
                        "lesion":           lesion_tag,
                        "depth":            depth_label,
                        "control_group":    gA,
                        "test_group":       gB,
                        "n_control":        len(flat_group_data.get(gA, [])),
                        "n_test":           len(flat_group_data.get(gB, [])),
                        "median_ks_stat":   info["median_ks"],
                        "median_p":         info["median_p"],
                        "pct_sig_05":       info["pct_sig_05"],
                        "pct_sig_01":       info["pct_sig_01"],
                        "representative_p": info["p_val"],
                    })

                # ── CDH Plot (still uses bootstrap_cdfs for median CDF line) ──
                fig, ax   = plt.subplots(figsize=fig_size, dpi=dpi)

                for g, cdfs in bootstrap_cdfs.items():
                    if cdfs.size == 0:
                        continue
                    median_cdf = np.median(cdfs, axis=0)
                    ci_low     = np.percentile(cdfs, 5,  axis=0)
                    ci_high    = np.percentile(cdfs, 95, axis=0)

                    color     = group_color_assigned.get(g, "black")
                    is_first  = (g == groups_in_fig[0])
                    linestyle = "--" if is_first else "-"
                    lw        = 0.8  if is_first else 1

                    ax.plot(x_values, median_cdf, label=g,
                            color=color, lw=lw, linestyle=linestyle)

                n_texts = [f"{g}: n={len(v)}" for g, v in flat_group_data.items()]
                ax.text(0.98, 0.30, "\n".join(n_texts),
                        transform=ax.transAxes, ha="right", va="top", fontsize=4)

                ks_texts = [
                    f"{gA} vs {gB}: p={info['p_val']:.2g}"
                    for (gA, gB), info in representative_info.items()
                ]
                if ks_texts:
                    ax.text(0.5, -0.4, "\n".join(ks_texts),
                            transform=ax.transAxes, ha="center", va="top", fontsize=4)

                ax.set_xlabel(
                    f"{depth_label if depth_label else 'All'} {measurement_col}",
                    fontsize=8)
                ax.set_ylabel("Cumulative Probability", fontsize=8)
                ax.set_xlim(0, 40)
                ax.set_ylim(0, 1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", which="major", labelsize=6,
                               width=1.0, length=4)
                ax.legend(fontsize=4, frameon=False, loc="upper left")
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.35)

                figs[fig_name]           = fig
                group_data_out[fig_name] = flat_group_data

                # ← store the representative raw data for Excel export
                cdh_data_records[fig_name] = {
                    "rep_info":          rep_info_with_samples,
                    "all_group_samples": all_group_samples,
                    "lesion_label":      lesion_tag or "All",
                    "layer_label":       depth_label or "All",
                }

                cdh_path = os.path.join(output_folder, f"{fig_name}.png")
                fig.savefig(cdh_path, dpi=dpi, bbox_inches="tight", transparent=True)
                plt.close(fig)
                print(f"  CDH figure saved: {cdh_path}")

    # ════════════════════════════════════════════════════════════════════════ #
    # SECTION B — lesion-vs-lesion comparisons per training group             #
    # ════════════════════════════════════════════════════════════════════════ #
    if use_lesion:
        lesion_values   = sorted(df_group[lesion_col].dropna().unique().tolist(), key=str)
        training_values = sorted({
            str(g).split("-")[-1]
            for comp in comparisons
            for g in comp
        })
        depth_labels = ["L2", "L3"] if split_by_depth else [None]

        print(f"\n=== LESION-VS-LESION ===")

        for train_name in training_values:
            for depth_label in depth_labels:
                print(f"\n--- train='{train_name}' | depth='{depth_label}' ---")

                lesion_group_data = {}

                for lesion in lesion_values:
                    lesion_tag = str(lesion)
                    subset     = df_group.copy()

                    if train_name is not None:
                        subset = subset[subset[training_col].astype(str) == str(train_name)]

                    subset = subset[subset[lesion_col].astype(str) == lesion_tag]

                    if split_by_depth and depth_label:
                        subset = subset[subset["DepthLabel"] == depth_label]

                    if subset.empty:
                        print(f"  [{lesion_tag}] empty subset — skipped")
                        continue

                    ctx = {
                        "section":        "lesion_comparison",
                        "training_group": str(train_name) if train_name is not None else "",
                        "lesion":         lesion_tag,
                        "depth":          depth_label or "",
                    }
                    cell_data = _load_group_data(
                        subset, filename_col, cd_index, sample_size,
                        label=lesion_tag, missing_log=missing_log, context=ctx)

                    if cell_data:
                        lesion_group_data[lesion_tag] = cell_data

                if len(lesion_group_data) < 2:
                    print("  SKIPPED: fewer than 2 lesion groups have data")
                    continue

                flat_lesion_data = {}
                for lt, cells in lesion_group_data.items():
                    flat = []
                    for fname, vals in cells.items():
                        if len(vals) >= sample_size:
                            flat.extend(random.sample(vals, sample_size))
                    flat_lesion_data[lt] = np.array(flat)
                    print(f"  [{lt}] flat size: {len(flat_lesion_data[lt])}")

                if len([v for v in flat_lesion_data.values() if len(v) > 0]) < 2:
                    print("  SKIPPED: fewer than 2 non-empty lesion groups")
                    continue

                lesion_keys = list(flat_lesion_data.keys())

                all_vals = np.concatenate(list(flat_lesion_data.values()))
                x_values = np.linspace(all_vals.min(), all_vals.max(), 200)

                bootstrap_cdfs, bootstrap_ks = _run_bootstrap(
                    flat_lesion_data, x_values, n_bootstrap)

                lesion_fig_name = f"LesionComp_{train_name}"
                if depth_label:
                    lesion_fig_name += f"_{depth_label}"

                representative_info = _compute_representative_info(
                    bootstrap_ks, n_bootstrap)

                # ── Extract raw samples from representative iteration ─────────
                rep_info_with_samples, all_group_samples = \
                    _extract_representative_samples(bootstrap_ks, representative_info)

                lesion_color_assigned = {
                    lt: _color_to_rgb(lesion_color_map.get(str(lt).strip(), "gray"))
                    for lt in lesion_keys
                }

                for pair, info in representative_info.items():
                    gA, gB    = pair
                    bar_color = (group_colors.get(str(train_name), "gray")
                                 if group_colors and train_name else "gray")
                    _save_histogram(lesion_fig_name, pair,
                                    info, n_bootstrap, output_folder, dpi,
                                    bar_color=bar_color)

                    summary_records.append({
                        "fig_name":         lesion_fig_name,
                        "comparison":       f"Lesion comparison ({train_name})",
                        "lesion":           f"{gA} vs {gB}",
                        "depth":            depth_label,
                        "control_group":    gA,
                        "test_group":       gB,
                        "n_control":        len(flat_lesion_data.get(gA, [])),
                        "n_test":           len(flat_lesion_data.get(gB, [])),
                        "median_ks_stat":   info["median_ks"],
                        "median_p":         info["median_p"],
                        "pct_sig_05":       info["pct_sig_05"],
                        "pct_sig_01":       info["pct_sig_01"],
                        "representative_p": info["p_val"],
                    })

                fig, ax   = plt.subplots(figsize=fig_size, dpi=dpi)

                for i, (lt, cdfs) in enumerate(bootstrap_cdfs.items()):
                    if cdfs.size == 0:
                        continue
                    median_cdf = np.median(cdfs, axis=0)
                    ci_low     = np.percentile(cdfs, 5,  axis=0)
                    ci_high    = np.percentile(cdfs, 95, axis=0)

                    color     = lesion_color_assigned.get(lt, "black")
                    is_first  = (i == 0)
                    linestyle = "--" if is_first else "-"
                    lw        = 0.8  if is_first else 1

                    ax.plot(x_values, median_cdf, label=lt,
                            color=color, lw=lw, linestyle=linestyle)

                n_texts = [f"{lt}: n={len(v)}" for lt, v in flat_lesion_data.items()]
                ax.text(0.98, 0.30, "\n".join(n_texts),
                        transform=ax.transAxes, ha="right", va="top", fontsize=4)

                ks_texts = [
                    f"{gA} vs {gB}: p={info['p_val']:.2g}"
                    for (gA, gB), info in representative_info.items()
                ]
                if ks_texts:
                    ax.text(0.5, -0.4, "\n".join(ks_texts),
                            transform=ax.transAxes, ha="center", va="top", fontsize=4)

                ax.set_xlabel(
                    f"{depth_label if depth_label else 'All'} {measurement_col}",
                    fontsize=8)
                ax.set_ylabel("Cumulative Probability", fontsize=8)
                ax.set_xlim(0, 40)
                ax.set_ylim(0, 1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="both", which="major", labelsize=6,
                               width=1.0, length=4)
                ax.set_title(f"Train: {train_name}", fontsize=6)
                ax.legend(fontsize=4, frameon=False, loc="upper left")
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.35)

                figs[lesion_fig_name]           = fig
                group_data_out[lesion_fig_name] = flat_lesion_data

                # ← store the representative raw data for Excel export
                cdh_data_records[lesion_fig_name] = {
                    "rep_info":          rep_info_with_samples,
                    "all_group_samples": all_group_samples,
                    "lesion_label":      f"LesionComp-{train_name}",
                    "layer_label":       depth_label or "All",
                }

                cdh_path = os.path.join(output_folder, f"{lesion_fig_name}.png")
                fig.savefig(cdh_path, dpi=dpi, bbox_inches="tight", transparent=True)
                plt.close(fig)
                print(f"  Lesion CDH figure saved: {cdh_path}")

    # ── Write summary outputs ────────────────────────────────────────────────
    if summary_records:
        _write_summary_png(summary_records,
                           os.path.join(output_folder, "CD_Bootstrap_Summary.png"))

    if cdh_data_records:
        _write_cdh_data_excel(cdh_data_records,
                              os.path.join(output_folder, "CD_Plot_Data.xlsx"))

    _write_missing_cells_excel(
        missing_log,
        os.path.join(output_folder, "CD_Missing_Cells.xlsx")
    )

    return figs, group_data_out