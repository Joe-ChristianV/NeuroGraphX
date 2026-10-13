# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:10:22 2025

@author: JoeCh
"""

# Raw_Trace_Module.py
# Refactored for NeuroGraphX GUI integration — no tkinter, no hardcoded paths.

import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def clean_fname(raw_fname):
    """Extract a clean file identifier from the raw filename string."""
    raw_fname = str(raw_fname)
    match = re.match(r"(JC20\d{8}[a-z])", raw_fname, re.IGNORECASE)
    if match:
        return match.group(1)
    return raw_fname.split(" ")[0]


def load_trace_txt(txt_path):
    """
    Load two-column numeric trace data from a .txt file.
    Skips the first 5 header lines (matching original behaviour).
    Returns an (N, 2) numpy array [time, current], or None on failure.
    """
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()[5:]
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        if not rows:
            return None
        return np.array(rows)
    except Exception as e:
        print(f"⚠️  Failed to load {txt_path}: {e}")
        return None


def _darken_color(color, factor):
    """
    Return a darkened version of `color` by multiplying RGB by (1 - factor).
    factor=0.0 → original, factor=1.0 → black.
    """
    try:
        r, g, b, *a = mcolors.to_rgba(color)
        r2 = max(0.0, r * (1.0 - factor))
        g2 = max(0.0, g * (1.0 - factor))
        b2 = max(0.0, b * (1.0 - factor))
        return (r2, g2, b2, a[0] if a else 1.0)
    except Exception:
        return color


def _average_traces_for_group(df_sub, filename_column, all_txt_files,
                               traces_folder, time_window=(3, 20),
                               baseline_window=(4, 5)):
    """
    Load, align, baseline-correct, and average all traces for a subset df.
    Returns (time_axis, mean_trace) or (None, warnings) if no valid traces.
    """
    traces   = []
    warnings = []

    for _, row in df_sub.iterrows():
        raw_fname = str(row[filename_column])
        fname     = clean_fname(raw_fname)

        matches = [f for f in all_txt_files
                   if f.lower().startswith(fname.lower())]
        if not matches:
            msg = f"No .txt file found for '{fname}'"
            warnings.append(msg)
            print(f"⚠️  {msg}")
            continue

        trace_data = load_trace_txt(os.path.join(traces_folder, matches[0]))
        if trace_data is not None:
            traces.append(trace_data)
        else:
            msg = f"No usable numeric data in {matches[0]}"
            warnings.append(msg)
            print(f"⚠️  {msg}")

    if not traces:
        return None, warnings

    min_len        = min(t.shape[0] for t in traces)
    aligned_traces = np.array([t[:min_len, 1] for t in traces])
    time_axis      = traces[0][:min_len, 0]

    # Baseline correction
    bm = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    if bm.any():
        baseline       = np.mean(aligned_traces[:, bm], axis=1)
        aligned_traces = aligned_traces - baseline[:, np.newaxis]

    # Time window
    wm             = (time_axis >= time_window[0]) & (time_axis <= time_window[1])
    time_axis      = time_axis[wm]
    aligned_traces = aligned_traces[:, wm]

    print(f"      {len(traces)} trace(s) averaged")
    return (time_axis, np.mean(aligned_traces, axis=0)), warnings


# ──────────────────────────────────────────────────────────────────────────────
# PLOT HELPER — shared between lesion and no-lesion modes
# ──────────────────────────────────────────────────────────────────────────────

def _plot_and_save(group_traces, comp_name_str, output_folder,
                   layer, dpi, offset_gap=20):
    """
    Render averaged traces side-by-side with x-offsets and save SVG + PNG.

    group_traces : OrderedDict  { display_label: (time_axis, mean_trace, color) }
    Returns list of saved file paths.
    """
    fig, ax = plt.subplots(figsize=(2, 1.35), dpi=dpi)

    x_offset  = 0
    x_offsets = []
    first_min = None

    for i, (lbl, (time_axis, mean_trace, trace_color)) in enumerate(group_traces.items()):
        ax.plot(time_axis + x_offset, mean_trace, lw=1, color=trace_color)
        x_offsets.append(x_offset)

        if i == 0:
            first_min = np.min(mean_trace)

        mid_x      = x_offset + (time_axis[0] + time_axis[-1]) / 2
        # Label above trace: use the bare group name (strip layer prefix)
        bare_label = lbl.split("-", 1)[-1] if lbl.count("-") >= 1 else lbl
        ax.text(mid_x, np.max(mean_trace) + 1, bare_label,
                color=trace_color, ha="center", va="bottom",
                fontsize=8, family="Arial", style="italic")

        x_offset += offset_gap

    # Reference lines
    first_time = next(iter(group_traces.values()))[0]
    last_time  = list(group_traces.values())[-1][0]
    x_start    = first_time[0]
    x_end      = x_offsets[-1] + last_time[-1]

    ax.hlines(0, x_start, x_end, colors="lightgray", linestyles="--", lw=1)
    if first_min is not None:
        ax.hlines(first_min, x_start, x_end,
                  colors="lightgray", linestyles="--", lw=1)

    # Scale bars
    y_min, y_max = ax.get_ylim()
    bar_x = x_end + 5
    bar_y = y_min + 0.2 * (y_max - y_min)
    ax.hlines(bar_y, bar_x - 5, bar_x, color="black", lw=1)
    ax.vlines(bar_x, bar_y, bar_y + 4, color="black", lw=1)
    ax.axis("off")

    if layer:
        ax.text(0.02, 0.98, layer,
                transform=ax.transAxes,
                fontsize=9, family="Arial", style="italic",
                va="top", ha="left", color="#555555")

    saved = []
    for ext, path in [
        (".svg", os.path.join(output_folder, f"AvgTrace_{comp_name_str}.svg")),
        (".png", os.path.join(output_folder, f"AvgTrace_{comp_name_str}.png")),
    ]:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
        saved.append(path)
        print(f"   ✅ {ext.upper()}  → {path}")

    plt.close(fig)
    return saved


# ──────────────────────────────────────────────
# Public entry point called by the GUI
# ──────────────────────────────────────────────

def run_trace_averager(
    df_group,
    comparisons,
    split_value,
    traces_folder,
    output_folder,
    training_column="GrpTrain",
    depth_column="Depth (μm)",
    filename_column="Filename",
    lesion_column=None,
    group_colors=None,
    lesion_colors=None,
    dpi=600,
):
    """
    Average raw traces per group/comparison and save SVG + PNG plots.

    NO LESION MODE
    --------------
    One plot per comparison × layer.
    Traces coloured by group_colors.
    Output: AvgTrace_{G1}_vs_{G2}_{layer}.svg/png

    LESION MODE
    -----------
    For each comparison × layer, all lesion groups for each training group
    are plotted side-by-side on the same figure.
    Within each training group the lesion traces are arranged left→right
    with progressively darker hues of the lesion base colour.
    Output: AvgTrace_{G1}_vs_{G2}_{layer}.svg/png
            (same filename — lesion info is encoded in the trace colours)

    Parameters
    ----------
    df_group        : pandas DataFrame
    comparisons     : list of lists e.g. [["ACC","SAT1"]]
    split_value     : float | None — depth cutoff (L2 < cutoff ≤ L3)
    traces_folder   : path to Raw_Data/Raw_Traces/
    output_folder   : path to Plots/
    training_column : str
    depth_column    : str | None
    filename_column : str
    lesion_column   : str | None — if set, activates lesion mode
    group_colors    : dict {group → colour}
    lesion_colors   : dict {lesion → colour}  used in lesion mode
    dpi             : int

    Returns
    -------
    saved_files  : list[str]
    all_warnings : list[str]
    """

    if group_colors  is None: group_colors  = {}
    if lesion_colors is None: lesion_colors = {}

    os.makedirs(output_folder, exist_ok=True)

    # ── Depth label ──────────────────────────────────────────────────────────
    has_depth = (
        bool(depth_column)
        and depth_column in df_group.columns
        and split_value is not None
    )
    df_work = df_group.copy()
    if has_depth:
        df_work["_DepthLabel"] = df_work[depth_column].apply(
            lambda x: "L2" if float(x) < split_value else "L3"
        )
        layers = ["L2", "L3"]
    else:
        layers = [None]

    # ── Lesion values ────────────────────────────────────────────────────────
    use_lesion = (
        bool(lesion_column)
        and lesion_column in df_work.columns
    )
    if use_lesion:
        lesion_values = sorted(df_work[lesion_column].dropna().astype(str).unique())
        print(f"  Lesion mode active — lesions: {lesion_values}")
    else:
        lesion_values = [None]

    # ── Validate traces folder ───────────────────────────────────────────────
    try:
        all_txt_files = [f for f in os.listdir(traces_folder)
                         if f.lower().endswith(".txt")]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Raw_Traces folder not found:\n{traces_folder}\n\n"
            "Create it and place your .txt trace files inside."
        )

    saved_files  = []
    all_warnings = []

    # ── Main loop: comparison × layer ────────────────────────────────────────
    for comp in comparisons:
        for layer in layers:
            layer_tag  = f"_{layer}" if layer else ""
            layer_desc = f" [{layer}]" if layer else ""
            print(f"\n=== Trace Averager — {' vs '.join(comp)}{layer_desc} ===")

            # group_traces maps display_label → (time_axis, mean_trace, color)
            # In no-lesion mode:  one entry per group
            # In lesion mode:     one entry per (group × lesion), ordered so
            #                     all lesions for G1 come before all for G2
            group_traces = {}

            if not use_lesion:
                # ── NO-LESION MODE ───────────────────────────────────────────
                # One figure per layer: ACC trace + SAT1 trace side by side.
                for g in comp:
                    mask   = df_work[training_column].astype(str) == g
                    if layer:
                        mask &= df_work["_DepthLabel"] == layer
                    df_sub = df_work[mask]

                    if df_sub.empty:
                        msg = (f"No rows for group='{g}'"
                               + (f", layer='{layer}'" if layer else "") + ".")
                        all_warnings.append(msg)
                        print(f"⚠️  {msg}")
                        continue

                    display_label = f"{layer}-{g}" if layer else g
                    print(f"   Loading '{display_label}' ({len(df_sub)} row(s)) …")

                    result, warns = _average_traces_for_group(
                        df_sub, filename_column, all_txt_files, traces_folder)
                    all_warnings.extend(warns)

                    if result is None:
                        msg = f"No valid traces for '{display_label}' — skipping."
                        all_warnings.append(msg)
                        print(f"⚠️  {msg}")
                        continue

                    time_axis, mean_trace = result
                    trace_color = group_colors.get(g, "black")
                    group_traces[display_label] = (time_axis, mean_trace, trace_color)

                if not group_traces:
                    msg = (f"Skipping {' vs '.join(comp)}{layer_desc} "
                           "— no data for any group.")
                    all_warnings.append(msg)
                    print(f"⚠️  {msg}")
                    continue

                comp_name_str = "_vs_".join(comp) + layer_tag

                # CSV
                csv_path = os.path.join(output_folder,
                                        f"AvgTrace_{comp_name_str}.csv")
                ref_time = next(iter(group_traces.values()))[0]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time (ms)"] + list(group_traces.keys()))
                    for i in range(len(ref_time)):
                        row_vals = [ref_time[i]] + [
                            group_traces[lbl][1][i]
                            if i < len(group_traces[lbl][1]) else ""
                            for lbl in group_traces
                        ]
                        writer.writerow(row_vals)
                saved_files.append(csv_path)
                print(f"   ✅ CSV  → {csv_path}")

                paths = _plot_and_save(group_traces, comp_name_str,
                                       output_folder, layer, dpi)
                saved_files.extend(paths)

            else:
                # ── LESION MODE ──────────────────────────────────────────────
                # One figure per lesion per layer.
                # Each figure: one trace per training group (e.g. ACC, SAT1),
                # coloured by the lesion colour for that lesion.
                for lv in lesion_values:
                    safe_lv = str(lv).replace(" ", "_").replace("/", "_")
                    lv_desc = f" [{layer}|{lv}]" if layer else f" [{lv}]"
                    print(f"   --- Lesion: {lv}{layer_desc} ---")

                    lesion_traces = {}   # {display_label: (time, trace, color)}

                    for g in comp:
                        mask = df_work[training_column].astype(str) == g
                        mask &= df_work[lesion_column].astype(str) == str(lv)
                        if layer:
                            mask &= df_work["_DepthLabel"] == layer
                        df_sub = df_work[mask]

                        if df_sub.empty:
                            print(f"⚠️  No rows for group='{g}', "
                                  f"lesion='{lv}'"
                                  + (f", layer='{layer}'" if layer else "")
                                  + ".")
                            continue

                        display_label = f"{layer}-{g}" if layer else g
                        print(f"   Loading '{display_label}' "
                              f"({len(df_sub)} row(s)) …")

                        result, warns = _average_traces_for_group(
                            df_sub, filename_column, all_txt_files,
                            traces_folder)
                        all_warnings.extend(warns)

                        if result is None:
                            print(f"⚠️  No valid traces for '{display_label}'.")
                            continue

                        time_axis, mean_trace = result
                        # Colour = lesion colour; fall back to group colour
                        trace_color = lesion_colors.get(str(lv),
                                      group_colors.get(g, "steelblue"))
                        lesion_traces[display_label] = (time_axis, mean_trace,
                                                         trace_color)

                    if not lesion_traces:
                        print(f"⚠️  No data for lesion '{lv}'{layer_desc} — skipping.")
                        continue

                    # Filename encodes both layer and lesion
                    comp_name_str = "_vs_".join(comp) + layer_tag + f"_{safe_lv}"

                    # CSV
                    csv_path = os.path.join(output_folder,
                                            f"AvgTrace_{comp_name_str}.csv")
                    ref_time = next(iter(lesion_traces.values()))[0]
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Time (ms)"] + list(lesion_traces.keys()))
                        for i in range(len(ref_time)):
                            row_vals = [ref_time[i]] + [
                                lesion_traces[lbl][1][i]
                                if i < len(lesion_traces[lbl][1]) else ""
                                for lbl in lesion_traces
                            ]
                            writer.writerow(row_vals)
                    saved_files.append(csv_path)
                    print(f"   ✅ CSV  → {csv_path}")

                    paths = _plot_and_save(lesion_traces, comp_name_str,
                                           output_folder, layer, dpi)
                    saved_files.extend(paths)

    return saved_files, all_warnings