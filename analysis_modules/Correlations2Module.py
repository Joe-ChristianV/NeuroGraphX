# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Correlations2Module.py
======================
Simplified correlation graphs pairing per-window lick metrics against
animal-level covariates.  All graphs split by lesion.

TWO output figures per (comparison × lesion × layer × measure):

  Figure A  — _corr2_behaviour/   (no qEPSC)
    Row 0: Lick Diff vs Reward Size          (if water_ml_col set)
    Row 1: Lick Diff vs Expected Value       (if water_ml_col set)
    Row 2: Lick Diff vs Water Probability    (groups combined, line = lesion)
    Row 3: Lick Diff vs ILI  (same window)   (groups combined, line = lesion)

  Figure B  — _corr2_qepsc/        (qEPSC correlations)
    Row 0: Lick Diff vs qEPSC  (by group)
    Row 1: Lick Diff vs qEPSC  (all animals)
    Row 2: ILI        vs qEPSC  (by group)

All panels within a figure share the same Y axis.
All panels within a row share the same X axis.
Scatter fill = group colour.  Regression line = lesion colour.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr

import statsmodels.api as sm

from .BehaviorTrialsModule import (
    _load_and_merge,
    _valid_days,
    _segment_trials,
    _lick_hz_all,
    _align_days,
    _extract_selected_groups,
    _animal_lick_difference,
    _is_acclimation_group,
    WIN_DEFS,
    WIN_NAMES,
)
from .CorrelationModule import (
    _norm,
    _is_always_water,
    _compute_water_probability,
    _compute_total_water_trials,
    _build_bar,
    _apply_min_cells,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SIG_ALPHA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# PER-WINDOW ILI  (inter-lick interval = mean time between lick-on events)
# ─────────────────────────────────────────────────────────────────────────────

def _ili_all(seg_ts: np.ndarray, seg_lick: np.ndarray, ref: float) -> np.ndarray:
    """
    Return array of shape (4,) with mean ILI (s) for each WIN_DEF window.
    ILI = mean time between consecutive lick-on events (seg_lick == 2)
    within the window.  Returns np.nan for windows with fewer than 2 licks.
    """
    lick_ts = seg_ts[seg_lick == 2]
    rel     = lick_ts - ref
    out     = np.full(4, np.nan, dtype=np.float64)
    for j, (w0, w1) in enumerate(WIN_DEFS):
        win_licks = rel[(rel >= w0) & (rel < w1)]
        if len(win_licks) >= 2:
            out[j] = float(np.diff(np.sort(win_licks)).mean())
    return out


def _animal_ili_per_window(files, group=None):
    """
    Compute mean per-window ILI for one animal using the same
    day-selection logic as _lick_diff_per_window.

    Returns dict: win_def -> ILI (s, float) or None on failure.
    """
    alignment = "forward" if _is_acclimation_group(group) else "reversed"

    df = _load_and_merge(files)
    if df is None or df.empty:
        return None
    valid  = _valid_days(df)
    trials = _segment_trials(df)
    if not trials:
        return None

    water = [t for t in trials if     t[3] and t[4] is not None and t[2] in valid]
    blank = [t for t in trials if not t[3] and t[4] is not None and t[2] in valid]

    # Pool water + blank trials for ILI (ILI is a licking rhythm metric,
    # not condition-specific in the same way lick-diff is)
    all_trials = water + blank

    ili_by_day: dict = defaultdict(list)   # day -> list of shape-(4,) arrays
    for seg_ts, seg_lick, day, _, ref in all_trials:
        arr = _ili_all(seg_ts, seg_lick, ref)
        ili_by_day[day].append(arr)

    all_days = sorted(ili_by_day.keys())
    if not all_days:
        return None

    ordered = _align_days(all_days, alignment)

    # Pick the correct slot (mirrors _lick_diff_per_window exactly)
    if alignment == "forward":
        slot = 1 if len(ordered) > 1 else 0
    else:
        slot = 0

    if slot >= len(ordered):
        return None

    day = ordered[slot]
    arrs = [a for a in ili_by_day[day] if not np.all(np.isnan(a))]
    if not arrs:
        return None

    # nanmean across trials for that day
    stacked = np.stack(arrs)   # (n_trials, 4)
    mean_ili = np.nanmean(stacked, axis=0)  # (4,)

    return {win_def: float(mean_ili[j]) for j, win_def in enumerate(WIN_DEFS)}


# ─────────────────────────────────────────────────────────────────────────────
# LICK-DIFF EXTRACTION  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _lick_diff_per_window(files, group=None):
    """
    Compute per-window lick-diff (water − blank) using the same
    day-selection logic as BehaviorTrialsModule.
    Returns dict: win_def -> float, or None on failure.
    """
    alignment = "forward" if _is_acclimation_group(group) else "reversed"
    diff_dict, _ = _animal_lick_difference(files, alignment=alignment)
    if diff_dict is None or not diff_dict:
        return None
    slot = (1 if 1 in diff_dict else 0) if alignment == "forward" else 0
    if slot not in diff_dict:
        return None
    win_vals = diff_dict[slot]
    return {win_def: win_vals[j] for j, win_def in enumerate(WIN_DEFS)}


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_scalar_vs_lickdiff(animals, lick_diff, scalar_dict, group_of):
    """
    X = scalar (reward size / EV / water prob), Y = lick diff.
    Returns list of 4 dicts (one per window): {x, y, labels}
    """
    rows = []
    for win_def in WIN_DEFS:
        x, y, labels_out = [], [], []
        for a in animals:
            sv = scalar_dict.get(a, np.nan)
            ld = (lick_diff.get(a) or {}).get(win_def, np.nan)
            if np.isnan(float(sv)) or np.isnan(float(ld)):
                continue
            x.append(float(sv))
            y.append(float(ld))
            labels_out.append(group_of.get(a, "?"))
        rows.append(dict(x=x, y=y, labels=labels_out))
    return rows


def _collect_lickdiff_vs_ili(animals, lick_diff, ili_cache, group_of):
    """
    X = ILI for the SAME window, Y = lick diff for the SAME window.
    Returns list of 4 dicts.
    """
    rows = []
    for win_def in WIN_DEFS:
        x, y, labels_out = [], [], []
        for a in animals:
            ili_val = (ili_cache.get(a) or {}).get(win_def, np.nan)
            ld      = (lick_diff.get(a) or {}).get(win_def, np.nan)
            if np.isnan(float(ili_val)) or np.isnan(float(ld)):
                continue
            x.append(float(ili_val))
            y.append(float(ld))
            labels_out.append(group_of.get(a, "?"))
        rows.append(dict(x=x, y=y, labels=labels_out))
    return rows


def _collect_qepsc_vs_metric(animals, metric_cache, bar, measure, layer,
                               group_of, split_by_comparison):
    """
    X = qEPSC, Y = metric (lick_diff or ILI).
    metric_cache: dict animal -> {win_def: float}
    Returns list of 4 dicts.
    """
    rows = []
    for win_def in WIN_DEFS:
        x, y, labels_out = [], [], []
        for a in animals:
            if layer not in bar or a not in bar[layer]:
                continue
            if measure not in bar[layer][a]:
                continue
            qval = bar[layer][a][measure][0]
            mv   = (metric_cache.get(a) or {}).get(win_def, np.nan)
            if np.isnan(float(qval)) or np.isnan(float(mv)):
                continue
            x.append(float(qval))
            y.append(float(mv))
            labels_out.append(group_of.get(a, "?") if split_by_comparison else "all")
        rows.append(dict(x=x, y=y, labels=labels_out))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PANEL DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def _draw_panel(ax, x, y, labels, group_colors,
                xlabel, ylabel, title,
                xlim, ylim, line_color="#000000",
                split_by_comparison=True):
    """
    One scatter panel.
    - Fill   = group colour  (group_colors dict)
    - Line   = line_color    (= lesion colour)
    - Legend on every panel
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y   = x[mask], y[mask]
    labels = [l for l, m in zip(labels, mask) if m]

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    n = len(x)
    if n == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        return

    if split_by_comparison:
        seen = {}
        for xi, yi, li in zip(x, y, labels):
            col = group_colors.get(li, "#808080")
            if li not in seen:
                seen[li] = col
                ax.scatter(xi, yi, color=col, s=40, zorder=3, alpha=0.85,
                           edgecolors="none", label=li)
            else:
                ax.scatter(xi, yi, color=col, s=40, zorder=3, alpha=0.85,
                           edgecolors="none")
        if seen:
            ax.legend(fontsize=6, loc="best", framealpha=0.5, frameon=True)
    else:
        ax.scatter(x, y, color="#808080", s=40, zorder=3, alpha=0.85,
                   edgecolors="none")

    if n >= 3:
        r, p   = pearsonr(x, y)
        X1     = sm.add_constant(x)
        ols    = sm.OLS(y, X1).fit()
        xs     = np.linspace(xlim[0], xlim[1], 300)
        pred   = ols.predict(sm.add_constant(xs))
        ls     = "-" if p < SIG_ALPHA else "--"
        ax.plot(xs, pred, color=line_color, linestyle=ls, linewidth=2.0, zorder=4)
        ax.text(0.97, 0.97, f"r={r:.2f}\np={p:.3g}\nn={n}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.6))
    else:
        ax.text(0.97, 0.97, f"n<3\nn={n}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED AXIS LIMIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ylim_from_rows(row_defs):
    all_y = [yi for _, _, data_rows, _ in row_defs
             for d in data_rows for yi in d["y"]]
    if all_y:
        ypad = max(0.5, (max(all_y) - min(all_y)) * 0.15)
        return (min(all_y) - ypad, max(all_y) + ypad)
    return (-1, 1)


def _xlims_from_rows(row_defs):
    xlims = []
    for _, _, data_rows, _ in row_defs:
        all_x = [xi for d in data_rows for xi in d["x"]]
        if all_x:
            xpad = max(0.05, (max(all_x) - min(all_x)) * 0.12)
            xlims.append((min(all_x) - xpad, max(all_x) + xpad))
        else:
            xlims.append((0, 1))
    return xlims


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDER  (generic N-row × 4-col)
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure(suptitle, row_defs, group_colors, line_color):
    """
    row_defs: list of (row_label, x_label, data_rows[4], split_by_comparison)
    All panels share Y; rows share X.
    """
    n_rows = len(row_defs)
    ylim   = _ylim_from_rows(row_defs)
    xlims  = _xlims_from_rows(row_defs)

    fig, axes = plt.subplots(n_rows, 4,
                              figsize=(12, 3 * n_rows),
                              squeeze=False)
    fig.suptitle(suptitle, fontsize=12, y=1.01)

    for row_i, (row_label, x_label, data_rows, split_flag) in enumerate(row_defs):
        for col_i, (win_name, d) in enumerate(zip(WIN_NAMES, data_rows)):
            ax     = axes[row_i][col_i]
            ylabel = row_label if col_i == 0 else ""
            _draw_panel(
                ax=ax,
                x=d["x"], y=d["y"], labels=d["labels"],
                group_colors=group_colors,
                xlabel=x_label,
                ylabel=ylabel,
                title=win_name,
                xlim=xlims[row_i],
                ylim=ylim,
                line_color=line_color,
                split_by_comparison=split_flag,
            )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A  — behaviour correlations (no qEPSC)
# ─────────────────────────────────────────────────────────────────────────────

def _make_behaviour_fig(title, animals, lick_diff_cache, ili_cache,
                         reward_size, expected_value, water_prob,
                         group_of, group_colors, line_color, has_water_ml):
    """
    Rows:
      0: Lick Diff vs Reward Size     (if has_water_ml)
      1: Lick Diff vs Expected Value  (if has_water_ml)
      2: Lick Diff vs Water Prob      (groups combined → split_flag=False)
      3: Lick Diff vs ILI (same win)  (groups combined → split_flag=False)
    """
    row_defs = []

    if has_water_ml:
        row_defs.append((
            "Lick Diff vs\nReward Size",
            "Reward Size (µL/trial)",
            _collect_scalar_vs_lickdiff(animals, lick_diff_cache, reward_size, group_of),
            True,
        ))
        row_defs.append((
            "Lick Diff vs\nExpected Value",
            "Expected Value (µL × P(Water))",
            _collect_scalar_vs_lickdiff(animals, lick_diff_cache, expected_value, group_of),
            True,
        ))

    row_defs.append((
        "Lick Diff vs\nWater Probability",
        "P(Water)",
        _collect_scalar_vs_lickdiff(animals, lick_diff_cache, water_prob, group_of),
        False,   # groups combined
    ))
    row_defs.append((
        "Lick Diff vs\nILI (same window)",
        "ILI (s)",
        _collect_lickdiff_vs_ili(animals, lick_diff_cache, ili_cache, group_of),
        False,   # groups combined
    ))

    return _build_figure(
        suptitle=f"{title}  —  Behaviour Correlations",
        row_defs=row_defs,
        group_colors=group_colors,
        line_color=line_color,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B  — qEPSC correlations
# ─────────────────────────────────────────────────────────────────────────────

def _make_qepsc_fig(title, animals, lick_diff_cache, ili_cache,
                     bar, measure, layer, group_of, group_colors,
                     cell_counts, line_color):
    """
    Rows:
      0: Lick Diff vs qEPSC  (by group)
      1: Lick Diff vs qEPSC  (all animals combined)
      2: ILI        vs qEPSC  (by group)
    """
    qepsc_animals = (
        _apply_min_cells(animals, cell_counts, layer, measure)
        if cell_counts is not None else animals
    )

    row_defs = [
        (
            f"Lick Diff vs\n{measure} (by group)",
            f"{measure} (pA)",
            _collect_qepsc_vs_metric(
                qepsc_animals, lick_diff_cache, bar, measure, layer,
                group_of, split_by_comparison=True),
            True,
        ),
        (
            f"Lick Diff vs\n{measure} (all animals)",
            f"{measure} (pA)",
            _collect_qepsc_vs_metric(
                qepsc_animals, lick_diff_cache, bar, measure, layer,
                group_of, split_by_comparison=False),
            False,
        ),
        (
            f"ILI vs\n{measure} (by group)",
            f"{measure} (pA)",
            _collect_qepsc_vs_metric(
                qepsc_animals, ili_cache, bar, measure, layer,
                group_of, split_by_comparison=True),
            True,
        ),
    ]

    return _build_figure(
        suptitle=f"{title}  |  {layer}  |  {measure}  —  qEPSC Correlations",
        row_defs=row_defs,
        group_colors=group_colors,
        line_color=line_color,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, path, dpi=300):
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    except Exception as e:
        print(f"[Corr2] Could not save {path}: {e}")
    finally:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_correlation2(df_master, animal_id_col, training_col,
                     bar_measure_cols, comparisons, training_data_dir,
                     output_folder, lesion_col=None, lesion_filter=None,
                     depth_col=None, depth_threshold=None,
                     water_ml_col=None,
                     group_colors=None, lesion_colors=None):
    """
    Generate Correlations 2 figures.

    Outputs (inside output_folder):
        _corr2_behaviour/   Figure A per comparison × lesion (behaviour only)
        _corr2_qepsc/       Figure B per comparison × lesion × layer × measure
    """
    group_colors  = group_colors  or {}
    lesion_colors = lesion_colors or {}

    out_beh   = os.path.join(output_folder, "_corr2_behaviour")
    out_qepsc = os.path.join(output_folder, "_corr2_qepsc")
    os.makedirs(out_beh,   exist_ok=True)
    os.makedirs(out_qepsc, exist_ok=True)

    # ── Prepare master data ───────────────────────────────────────────────────
    df = df_master.copy()
    groups = [_norm(g) for g in _extract_selected_groups(comparisons)]
    df[animal_id_col] = df[animal_id_col].astype(str)
    df[training_col]  = df[training_col].astype(str)
    df = df[df[training_col].isin(groups)]

    bar_measure_cols = [c for c in bar_measure_cols if c in df.columns]
    if not bar_measure_cols:
        print("[Corr2] No valid measure columns found in data.")
        return

    # ── Registry ─────────────────────────────────────────────────────────────
    registry = defaultdict(dict)
    for g in groups:
        gdf = df[df[training_col] == g]
        for a in gdf[animal_id_col].unique():
            a     = _norm(a)
            files = glob.glob(os.path.join(training_data_dir, a, "*.txt"))
            if files:
                registry[g][a] = (files, g)

    flat_registry = {}
    for g in registry:
        for a, (f, grp) in registry[g].items():
            flat_registry[a] = (f, grp)

    # ── Per-animal caches ─────────────────────────────────────────────────────
    lick_diff_cache    = {}
    ili_cache          = {}
    water_prob         = {}
    total_water_trials = {}
    water_ml_lookup    = {}

    for a, (f, grp) in flat_registry.items():
        lick_diff_cache[a]    = _lick_diff_per_window(f, grp)
        ili_cache[a]          = _animal_ili_per_window(f, grp)
        water_prob[a]         = _compute_water_probability(f, grp)
        total_water_trials[a] = _compute_total_water_trials(f, grp)

    if water_ml_col and water_ml_col in df.columns:
        water_ml_lookup = (
            df.groupby(animal_id_col)[water_ml_col]
            .apply(lambda s: np.nanmean(pd.to_numeric(s, errors="coerce")))
            .to_dict()
        )

    # ── Reward Size & Expected Value ──────────────────────────────────────────
    reward_size    = {}
    expected_value = {}
    for a in flat_registry:
        wml = float(water_ml_lookup.get(a, np.nan))
        twt = float(total_water_trials.get(a, np.nan))
        wp  = float(water_prob.get(a, np.nan))
        reward_size[a]    = (wml / twt) * 1000 if (not np.isnan(wml) and not np.isnan(twt) and twt > 0) else np.nan
        expected_value[a] = wml * wp             if (not np.isnan(wml) and not np.isnan(wp))             else np.nan

    # ── Bar (qEPSC) ───────────────────────────────────────────────────────────
    bar, cell_counts = _build_bar(df, animal_id_col, bar_measure_cols,
                                   depth_col, depth_threshold)

    # ── Lesion values ─────────────────────────────────────────────────────────
    if lesion_col and lesion_col in df.columns:
        lesion_vals = (lesion_filter if lesion_filter
                       else sorted(df[lesion_col].dropna().astype(str).unique()))
    else:
        lesion_vals = [None]

    # ── Animal → group mapping ────────────────────────────────────────────────
    group_of = {}
    for g in groups:
        for a in df[df[training_col] == g][animal_id_col].astype(str).apply(_norm).unique():
            group_of[a] = g

    has_wml = bool(water_ml_col and water_ml_col in df.columns)

    # ── Iterate ───────────────────────────────────────────────────────────────
    for comp_groups_raw in comparisons:
        comp_group_labels = [_norm(g) for g in _extract_selected_groups([comp_groups_raw])]
        comp_group_labels = [g for g in comp_group_labels if g in registry]
        if not comp_group_labels:
            continue

        comp_tag = "_vs_".join(comp_group_labels)

        for lesion in lesion_vals:
            lesion_tag = str(lesion) if lesion is not None else "ALL"
            line_color = lesion_colors.get(str(lesion), "#000000") if lesion is not None else "#000000"

            # Animals for this comparison + lesion
            comp_animals = []
            for g in comp_group_labels:
                gdf = df[df[training_col] == g]
                if lesion is not None and lesion_col and lesion_col in df.columns:
                    gdf = gdf[gdf[lesion_col] == lesion]
                sub_a = set(map(_norm, gdf[animal_id_col].astype(str).unique()))
                reg_a = set(map(_norm, registry[g].keys()))
                comp_animals += sorted(sub_a & reg_a)

            if not comp_animals:
                continue

            tag_base = f"{comp_tag}_{lesion_tag}"

            # ── Figure A: behaviour ───────────────────────────────────────────
            fig_beh = _make_behaviour_fig(
                title      = f"{comp_tag} | {lesion_tag}",
                animals    = comp_animals,
                lick_diff_cache = lick_diff_cache,
                ili_cache  = ili_cache,
                reward_size    = reward_size,
                expected_value = expected_value,
                water_prob = water_prob,
                group_of   = group_of,
                group_colors   = group_colors,
                line_color = line_color,
                has_water_ml   = has_wml,
            )
            _save_fig(fig_beh, os.path.join(out_beh, f"{tag_base}.png"))

            # ── Figure B: qEPSC (one per layer × measure) ─────────────────────
            for layer in bar:
                for measure in bar_measure_cols:
                    tag = f"{tag_base}_{layer}_{measure}"
                    fig_q = _make_qepsc_fig(
                        title      = f"{comp_tag} | {lesion_tag}",
                        animals    = comp_animals,
                        lick_diff_cache = lick_diff_cache,
                        ili_cache  = ili_cache,
                        bar        = bar,
                        measure    = measure,
                        layer      = layer,
                        group_of   = group_of,
                        group_colors   = group_colors,
                        cell_counts    = cell_counts,
                        line_color = line_color,
                    )
                    _save_fig(fig_q, os.path.join(out_qepsc, f"{tag}.png"))

    print(f"[Corr2] Done — figures saved under {output_folder}")