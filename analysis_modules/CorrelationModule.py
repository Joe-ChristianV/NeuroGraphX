# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 08:20:43 2026

@author: JoeCh
"""
# ===================== CorrelationModule.py =====================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr

import statsmodels.api as sm

from .BehaviorTrialsModule import (
    _load_and_merge,
    _valid_days,
    _segment_trials,
    _classify_trial,
    _trial_start_time,
    _lick_hz,
    _extract_selected_groups,
    WIN_DEFS,
    WIN_NAMES,
)

# =========================================================
# CONSTANTS
# =========================================================

WINDOWS   = WIN_NAMES
SIG_ALPHA = 0.05

COLORS = {
    "water": "blue",
    "blank": "red",
    "delta": "black",
}

CONDITIONS = ["water", "blank", "delta"]

# Groups that receive only water trials by experimental design.
# Returning 1.0 directly is both correct and guards against edge
# cases where trial parsing drops trials for these animals.
_ALWAYS_WATER_GROUPS = {"1", "100ACC", "ACC100"}

def _is_always_water(group):
    """
    Return True if *group* is a 100%-water training group.

    Handles all common label variants:
      "1", "100", "100%", "100 %", "100ACC", "ACC100",
      "100 ACC", "ACC 100", and case-insensitive forms.
    """
    if group is None:
        return False
    s = str(group).strip().upper().replace(" ", "").replace("%", "")
    # Exact set membership (normalised)
    _norm_set = {"1", "100", "100ACC", "ACC100"}
    return s in _norm_set


# =========================================================
# UTIL
# =========================================================

def _norm(x):
    return str(x).strip()


# =========================================================
# LICK EXTRACTION
# =========================================================

def _extract_lick_stats(files, group=None):
    """
    Extract per-window lick-rate stats for the last valid day.

    For always-water groups (100 %, ACC100, etc.) every trial is a water
    trial by experimental design.  Trials the classifier labels as 'blank'
    for these animals are simply water trials where the hardware did not
    log water delivery.  We therefore:
      • Put ALL trials into the water bucket.
      • Leave the blank bucket empty  →  blank mean = NaN, delta = NaN.
    This prevents spurious blank/delta values contaminating the
    lick-rate correlation plots for 100 % animals.
    """

    df = _load_and_merge(files)
    if df is None or df.empty:
        return None

    valid = _valid_days(df)
    if not valid:
        return None

    trials = _segment_trials(df)
    if not trials:
        return None

    t0           = df["timestamp"].iloc[0]
    always_water = _is_always_water(group)

    water = defaultdict(list)
    blank = defaultdict(list)

    for t in trials:
        # _segment_trials returns tuples: (seg_ts, seg_lick, day, is_water, ref)
        if isinstance(t, tuple) and len(t) == 5 and isinstance(t[0], np.ndarray):
            seg_ts, seg_lick, day, is_water, ref = t
        else:
            # Legacy DataFrame path (fallback)
            day = int((t["timestamp"].iloc[0] - t0) // 86400)
            is_water = _classify_trial(t) == "water"
            ref = _trial_start_time(t)
            seg_ts = t["timestamp"].values
            seg_lick = t["lick"].values
            t = (seg_ts, seg_lick, day, is_water, ref)

        if day not in valid:
            continue
        if always_water or is_water:
            water[day].append(t)
        else:
            blank[day].append(t)

    last_day = max(valid)

    def _vals(trial_list, win):
        out = []
        for t in trial_list:
            seg_ts, seg_lick, day, is_water, ref = t
            if ref is None:
                continue
            out.append(_lick_hz(seg_ts, seg_lick, ref, win))
        return np.array(out, dtype=float)

    result = {}

    for win in WIN_DEFS:

        w = _vals(water[last_day], win)
        b = _vals(blank[last_day], win)

        def mean_sem(x):
            if len(x) == 0:
                return np.nan, np.nan
            return np.nanmean(x), (np.nanstd(x) / np.sqrt(len(x))) if len(x) > 1 else 0

        result[("water", win)] = mean_sem(w)
        result[("blank", win)] = mean_sem(b)

        if len(w) and len(b):
            result[("delta", win)] = (np.nanmean(w) - np.nanmean(b), np.nan)
        else:
            result[("delta", win)] = (np.nan, np.nan)

    return result


# =========================================================
# BAR DATA
# =========================================================

def _build_bar(df, animal_id_col, cols, depth_col, depth_thr):
    """
    Returns (bar, cell_counts).

    bar[layer][animal][measure]        = (mean, sem)
    cell_counts[layer][animal][measure] = n_valid_cells

    cell_counts is used downstream to drop animals with < 2 valid
    cells for a given layer+measure combination.
    """
    layers = ["L2", "L3"] if depth_col else [None]
    out    = {l: {} for l in layers}
    counts = {l: {} for l in layers}

    df[animal_id_col] = df[animal_id_col].astype(str)

    for l in layers:

        sub = df.copy()

        if depth_col:
            if l == "L2":
                sub = sub[sub[depth_col] < depth_thr]
            else:
                sub = sub[sub[depth_col] >= depth_thr]

        for aid, grp in sub.groupby(animal_id_col):

            aid  = _norm(aid)
            row  = {}
            crow = {}

            for c in cols:
                vals = pd.to_numeric(grp[c], errors="coerce").dropna()
                n    = len(vals)
                if n == 0:
                    print(f"[WARNING] Column '{c}' is entirely NaN "
                          f"for animal '{aid}' in layer '{l}'")
                    row[c]  = (np.nan, np.nan)
                    crow[c] = 0
                else:
                    row[c]  = (
                        float(np.nanmean(vals)),
                        float(np.nanstd(vals) / np.sqrt(n)) if n > 1 else 0.0,
                    )
                    crow[c] = n

            out[l][aid]    = row
            counts[l][aid] = crow

    return out, counts

#Minimum Cell Requirement for Graphing
def _apply_min_cells(animals, cell_counts, layer, measure, min_cells=1):
    """
    Return the subset of *animals* that have >= min_cells valid cells
    for the given layer+measure.  Animals missing from cell_counts
    entirely are also dropped.
    """
    kept    = []
    dropped = []
    for a in animals:
        n = cell_counts.get(layer, {}).get(a, {}).get(measure, 0)
        if n >= min_cells:
            kept.append(a)
        else:
            dropped.append((a, n))
    if dropped:
        for a, n in dropped:
            print(f"  [MinCells] Dropped '{a}' from {layer}/{measure} "
                  f"(n_cells={n} < {min_cells})")
    return kept


# =========================================================
# WATER PROBABILITY
# =========================================================

def _compute_water_probability(files, group):
    """
    Return P(water trial) for an animal.

    Animals in always-water groups (100 %, ACC100, etc.) receive only
    water trials by experimental design, so we return 1.0 directly.
    """
    if _is_always_water(group):
        return 1.0

    df = _load_and_merge(files)
    if df is None or df.empty:
        return np.nan

    trials = _segment_trials(df)
    if not trials:
        return np.nan

    # _segment_trials returns tuples: (seg_ts, seg_lick, day, is_water, ref)
    if trials and isinstance(trials[0], tuple) and len(trials[0]) == 5 and isinstance(trials[0][0], np.ndarray):
        labels = ["water" if t[3] else "blank" for t in trials]
    else:
        labels = [_classify_trial(t) for t in trials]
    water  = np.sum(np.array(labels) == "water")
    total  = len(labels)

    return water / total if total > 0 else np.nan


# =========================================================
# TOTAL WATER TRIALS (across all valid days)
# =========================================================

def _compute_total_water_trials(files, group=None):
    """
    Count water trials across ALL valid days.

    For always-water groups the hardware does not log water delivery on
    ~20 % of trials, so _classify_trial labels those as 'blank'.
    We treat ALL parsed valid-day trials as water for these animals —
    matching the P(Water) = 1.0 shortcut — so that the reward-size
    denominator is consistent with what the animal actually received.

    Returns an int (>= 0) or np.nan if data cannot be parsed.
    """
    df = _load_and_merge(files)
    if df is None or df.empty:
        return np.nan

    valid = _valid_days(df)
    if not valid:
        return np.nan

    trials = _segment_trials(df)
    if not trials:
        return np.nan

    t0           = df["timestamp"].iloc[0]
    always_water = _is_always_water(group)

    # _segment_trials returns tuples: (seg_ts, seg_lick, day, is_water, ref)
    _tuple_format = (trials and isinstance(trials[0], tuple)
                     and len(trials[0]) == 5 and isinstance(trials[0][0], np.ndarray))

    total_water = 0
    for t in trials:
        if _tuple_format:
            seg_ts, seg_lick, day, is_water, ref = t
        else:
            day = int((t["timestamp"].iloc[0] - t0) // 86400)
            is_water = _classify_trial(t) == "water"
        if day not in valid:
            continue
        if always_water or is_water:
            total_water += 1

    return total_water


# =========================================================
# SCATTER CORE  (lick-rate correlation figures)
# =========================================================

def _scatter(ax, x, y, xerr=None, yerr=None, color="black", condition=None,
             ylim=(10, 22), xlim_delta=(-10, 10), xlim_other=(0, 10)):
    """
    Scatter with optional error bars and Pearson correlation.
    Axis limits are keyword parameters so callers can override
    them without risk of silently clipping data.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if xerr is not None:
        xerr = np.array(xerr)[mask]
    if yerr is not None:
        yerr = np.array(yerr)[mask]

    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt='o', markersize=5,
                ecolor='gray', capsize=2,
                alpha=0.75, color=color)

    if len(x) >= 3:
        r, p = pearsonr(x, y)
        m, b = np.polyfit(x, y, 1)
        xs   = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(xs, m * xs + b, 'k-' if p < SIG_ALPHA else 'k--')
        ax.text(0.95, 0.95, f"r={r:.2f}\np={p:.2g}",
                transform=ax.transAxes, ha="right", va="top")
    else:
        ax.text(0.95, 0.95, "n<3",
                transform=ax.transAxes, ha="right", va="top")

    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 2))

    if condition == "delta":
        ax.set_xlim(*xlim_delta)
        ax.set_xticks(np.arange(xlim_delta[0], xlim_delta[1] + 1, 2))
    else:
        ax.set_xlim(*xlim_other)
        ax.set_xticks(np.arange(xlim_other[0], xlim_other[1] + 1, 2))


# =========================================================
# FIGURE BUILDER  (lick-rate correlations)
# =========================================================

def _make_fig(title, animals, lick, bar, measure, layer, out, cell_counts=None):

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(f"{title} | {layer} | {measure}")

    # Apply 2-cell minimum before any other filtering
    if cell_counts is not None:
        animals = _apply_min_cells(animals, cell_counts, layer, measure)

    valid = [
        a for a in animals
        if (a in lick and lick[a] is not None)
        and (layer in bar and a in bar[layer])
        and (measure in bar[layer][a])
    ]

    if not valid:
        plt.close(fig)
        return

    # zip WIN_NAMES and WIN_DEFS together to avoid index-coupling bugs
    for r, cond in enumerate(CONDITIONS):
        for c, (win_name, win_def) in enumerate(zip(WINDOWS, WIN_DEFS)):

            ax = axes[r][c]
            x, y, xe, ye = [], [], [], []

            for a in valid:
                if (cond, win_def) not in lick[a]:
                    continue
                lmean, lsem = lick[a][(cond, win_def)]
                bmean, bsem = bar[layer][a][measure]
                x.append(lmean)
                y.append(bmean)
                xe.append(lsem)
                ye.append(bsem)

            _scatter(ax, x, y, xe, ye, COLORS[cond], cond)
            ax.set_title(f"{cond} | {win_name}")
            ax.set_xlabel(f"{win_name}-{cond} ")
            ax.set_ylabel(f"{measure}")

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# =========================================================
# WATER PROBABILITY FIGURE
# =========================================================

def _water_prob_fig(lesion, layer, animals, water_prob, bar, measure,
                    outpath, df_raw=None, animal_id_col=None,
                    depth_col=None, depth_threshold=None,
                    cell_counts=None, single_group=False, animal_group=None):
    """
    Plots P(water) % (X, inverted) vs bar measure (Y).

    Two modes
    ---------
    animal-average (default, df_raw=None):
        One point per animal using the pre-averaged bar dict.

    by-cell (df_raw supplied):
        One point per cell (master-sheet row).

    single_group=True
    -----------------
    When only one training group is in the comparison, runs a purely
    animal-level plot (one point per animal, labelled) with a single
    linear regression only — no binning, no quadratic.

    Statistics
    ----------
    OLS on raw individual points (animals or cells).

      Linear:    y = B0 + B1*x            (always)
      Quadratic: y = B0 + B1*x + B2*x²   (when n >= 5, multi-group only)
      Best model by AIC highlighted; other shown faded.

    X-axis: INVERTED (100 % left, 0 % right), range 110 to -10.
    """

    # Apply 2-cell minimum
    if cell_counts is not None:
        animals = _apply_min_cells(animals, cell_counts, layer, measure)

    # --------------------------------------------------
    # 1. COLLECT RAW POINTS
    # --------------------------------------------------
    x_raw, y_raw, a_labels = [], [], []

    if df_raw is not None and animal_id_col is not None and not single_group:
        # --- By-cell mode (multi-group only) ---
        sub = df_raw.copy()
        sub[animal_id_col] = sub[animal_id_col].astype(str).apply(_norm)

        if depth_col and depth_col in sub.columns:
            if layer == "L2":
                sub = sub[sub[depth_col] < depth_threshold]
            elif layer == "L3":
                sub = sub[sub[depth_col] >= depth_threshold]

        for _, row in sub.iterrows():
            a  = _norm(row[animal_id_col])
            if a not in animals:
                continue
            wp = water_prob.get(a, np.nan)
            if np.isnan(wp):
                continue
            mval = pd.to_numeric(row.get(measure, np.nan), errors="coerce")
            if np.isnan(mval):
                continue
            x_raw.append(wp * 100.0)
            y_raw.append(float(mval))
            a_labels.append(a)

    else:
        # --- Animal-average mode (always used for single_group) ---
        for a in animals:
            wp = water_prob.get(a, np.nan)
            if np.isnan(wp):
                continue
            if layer not in bar or a not in bar[layer]:
                continue
            if measure not in bar[layer][a]:
                continue
            mval = bar[layer][a][measure][0]
            if np.isnan(mval):
                continue
            x_raw.append(wp * 100.0)
            y_raw.append(mval)
            a_labels.append(a)

    x_raw = np.array(x_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)

    mask  = ~(np.isnan(x_raw) | np.isnan(y_raw))
    x_raw    = x_raw[mask]
    y_raw    = y_raw[mask]
    a_labels = [a_labels[i] for i, m in enumerate(mask) if m]
    n_pts    = len(x_raw)

    pt_label = "Animals" if (single_group or df_raw is None) else "Cells"

    min_n = 3 if single_group else 4
    if n_pts < min_n:
        fig, ax = plt.subplots()
        ax.set_title(
            f"{lesion} | {layer} | {measure} "
            f"(insufficient data, n={n_pts} {pt_label.lower()})"
        )
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return

    # ---------------------------------------------------------------
    # SINGLE-GROUP MODE: animal-level scatter + linear regression only
    # ---------------------------------------------------------------
    if single_group:
        fig, ax = plt.subplots(figsize=(5, 4))

        # Colour points by group if provided
        if animal_group:
            unique_grps = list(dict.fromkeys(animal_group.get(a, "?") for a in a_labels))
            cmap = plt.cm.get_cmap("tab10", max(len(unique_grps), 1))
            g_color = {g: cmap(i) for i, g in enumerate(unique_grps)}
            for i, (xi, yi, ai) in enumerate(zip(x_raw, y_raw, a_labels)):
                grp = animal_group.get(ai, "?")
                ax.scatter(xi, yi, color=g_color[grp], s=50, zorder=3, alpha=0.85)
                ax.annotate(ai, (xi, yi), textcoords="offset points",
                            xytext=(4, 4), fontsize=6, color="dimgray")
        else:
            for xi, yi, ai in zip(x_raw, y_raw, a_labels):
                ax.scatter(xi, yi, color="black", s=50, zorder=3, alpha=0.85)
                ax.annotate(ai, (xi, yi), textcoords="offset points",
                            xytext=(4, 4), fontsize=6, color="dimgray")

        # Linear fit
        X1 = sm.add_constant(x_raw)
        m1 = sm.OLS(y_raw, X1).fit()
        xs   = np.linspace(110, -10, 500)
        pred = m1.predict(sm.add_constant(xs))
        r, p = pearsonr(x_raw, y_raw)
        ls   = "-" if p < SIG_ALPHA else "--"
        ax.plot(xs, pred, color="black", linestyle=ls, linewidth=2, zorder=4,
                label=f"r={r:.2f}, p={p:.3g}")

        ax.set_xlim(110, -10)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_ylim(12, 18)
        ax.set_yticks(np.arange(12, 19, 2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("P(Water) %")
        ax.set_ylabel("qEPSC (pA)")
        ax.set_title(f"{lesion} | {layer} | {measure}")
        ax.legend(fontsize=8)

        # Stat annotation
        labels_lin = ["B0 (intercept)", "B1 (P(Water))"]
        lines = ["-- Linear fit --",
                 f"  {'Param':<18} {'Value':>9}  {'SE':>9}  {'t':>8}  {'p':>8}"]
        for lbl, val, se, tv, pv in zip(
            labels_lin, m1.params, m1.bse, m1.tvalues, m1.pvalues
        ):
            sig = " *" if pv < SIG_ALPHA else "  "
            lines.append(f"  {lbl:<18} {val:>9.4f}  {se:>9.4f}  "
                         f"{tv:>8.3f}  {pv:>8.4f}{sig}")
        dof_m, dof_r = int(m1.df_model), int(m1.df_resid)
        lines.append(f"  N={n_pts} animals  df(model)={dof_m}  df(resid)={dof_r}  "
                     f"R²={m1.rsquared:.4f}  Adj.R²={m1.rsquared_adj:.4f}")
        lines.append(f"  ANOVA: F({dof_m},{dof_r})={m1.fvalue:.3f}  "
                     f"Prob>F={m1.f_pvalue:.4f}"
                     + (" *" if m1.f_pvalue < SIG_ALPHA else ""))

        fig.text(0.01, -0.22, "\n".join(lines),
                 transform=ax.transAxes, fontsize=6, fontfamily="monospace",
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.5))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.28)
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        return

    # --------------------------------------------------
    # 2. BIN FOR DISPLAY ONLY
    #    x SEM and y SEM are shown as error bars on the
    #    bin-mean markers.  They do NOT enter the fit.
    # --------------------------------------------------
    bins = {
        20:  (10,    30),
        50:  (40,    60),
        80:  (70,    90),
        100: (90,  100.1),
    }

    bx_plot, by, bxerr, byerr = [], [], [], []

    for center, (lo, hi) in bins.items():
        bmask = (x_raw >= lo) & (x_raw < hi)
        n = int(np.sum(bmask))
        if n == 0:
            continue
        xvals = x_raw[bmask]
        yvals = y_raw[bmask]
        bx_plot.append(center)
        by.append(np.nanmean(yvals))
        byerr.append(np.nanstd(yvals) / np.sqrt(n) if n > 1 else 0.0)
        bxerr.append(np.nanstd(xvals) / np.sqrt(n) if n > 1 else 0.0)

    bx_plot = np.array(bx_plot, dtype=float)
    by      = np.array(by,      dtype=float)
    bxerr   = np.array(bxerr,   dtype=float)
    byerr   = np.array(byerr,   dtype=float)
    n_bins  = len(bx_plot)

    if n_bins < 2:
        fig, ax = plt.subplots()
        ax.set_title(f"{lesion} | {layer} | {measure} (< 2 bins populated)")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return

    # --------------------------------------------------
    # 3. FIT ON RAW INDIVIDUAL POINTS
    #    N = n_pts  (animals or cells)
    #    df_resid = n_pts - n_params
    #    This matches graphing software when given one row per subject.
    # --------------------------------------------------
    fits = []

    X1 = sm.add_constant(x_raw)
    m1 = sm.OLS(y_raw, X1).fit()
    fits.append({
        "name":   "Linear",
        "model":  m1,
        "color":  "black",
        "labels": ["B0 (intercept)", "B1 (x)"],
    })

    if n_pts >= 5:
        X2 = sm.add_constant(np.column_stack([x_raw, x_raw ** 2]))
        m2 = sm.OLS(y_raw, X2).fit()
        fits.append({
            "name":   "Quadratic",
            "model":  m2,
            "color":  "blue",
            "labels": ["B0 (intercept)", "B1 (x)", "B2 (x^2)"],
        })

    best_idx = int(np.argmin([f["model"].aic for f in fits]))

    # --------------------------------------------------
    # 4. FIGURE
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))

    # Raw individual points — behind everything
    ax.scatter(
        x_raw, y_raw,
        color="gray",
        alpha=0.35,
        s=20,
        zorder=2,
        label=f"{pt_label} (n={n_pts})",
    )

    # Bin means with x and y SEM error bars — purely display
    ax.errorbar(
        bx_plot, by,
        xerr=bxerr,
        yerr=byerr,
        fmt='o',
        color='black',
        ecolor='gray',
        capsize=3,
        markersize=7,
        zorder=4,
        label="Bin mean ± SEM",
    )

    # Inverted x-axis: 100 % left, 0 % right
    ax.set_xlim(110, -10)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_ylim(12, 18)
    ax.set_yticks(np.arange(12, 19, 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("P(Water) %")
    ax.set_ylabel("qEPSC (pA)")
    ax.set_title(f"{lesion} | {layer} | {measure}")

    xs = np.linspace(110, -10, 500)

    # --------------------------------------------------
    # 5. PLOT FITS + STAT ANNOTATION
    # --------------------------------------------------
    annotation_blocks = []

    for i, f in enumerate(fits):
        m       = f["model"]
        is_best = (i == best_idx)
        alpha   = 1.0 if is_best else 0.25
        lw      = 2.5 if is_best else 1.5
        ls      = "-" if is_best else "--"

        if f["name"] == "Linear":
            xs_design = sm.add_constant(xs)
        else:
            xs_design = sm.add_constant(np.column_stack([xs, xs ** 2]))

        pred = m.predict(xs_design)
        ax.plot(xs, pred,
                linestyle=ls, color=f["color"],
                alpha=alpha, linewidth=lw,
                zorder=5,
                label=f["name"] + (" *" if is_best else ""))

        lines  = []
        marker = " * (best AIC)" if is_best else ""
        lines.append(f"-- {f['name']}{marker} --")
        lines.append(
            f"  {'Param':<18} {'Value':>9}  {'SE':>9}  {'t':>8}  {'p':>8}"
        )
        for label, val, se, tv, pv in zip(
            f["labels"], m.params, m.bse, m.tvalues, m.pvalues
        ):
            sig = " *" if pv < SIG_ALPHA else "  "
            lines.append(
                f"  {label:<18} {val:>9.4f}  {se:>9.4f}  "
                f"{tv:>8.3f}  {pv:>8.4f}{sig}"
            )

        dof_m  = int(m.df_model)
        dof_r  = int(m.df_resid)
        rss    = float(m.ssr)
        r2     = float(m.rsquared)
        r2_adj = float(m.rsquared_adj)
        fval   = float(m.fvalue)
        fp     = float(m.f_pvalue)
        sig_f  = " *" if fp < SIG_ALPHA else "  "

        lines.append(
            f"  N={n_pts} {pt_label.lower()}  "
            f"df(model)={dof_m}  df(resid)={dof_r}  "
            f"RSS={rss:.5f}  R2={r2:.4f}  Adj.R2={r2_adj:.4f}"
        )
        lines.append(
            f"  ANOVA: F({dof_m},{dof_r})={fval:.3f}  "
            f"Prob>F={fp:.4f}{sig_f}"
        )

        annotation_blocks.append("\n".join(lines))

    full_annotation = "\n\n".join(annotation_blocks)

    fig.text(
        0.01, -0.25,
        full_annotation,
        transform=ax.transAxes,
        fontsize=6,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.5),
    )

    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38 if n_pts >= 5 else 0.26)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# =========================================================
# REWARD SIZE FIGURE
# =========================================================

def _reward_size_fig(lesion, layer, animals, reward_size, bar, measure, outpath,
                     cell_counts=None, animal_group=None):
    """
    Plots Reward Size (X) vs Animal qEPSC / bar measure (Y).

    Reward Size per animal  =  (Water_mL / Tot.WaterTrials_across_all_days) * 1000
    where Water_mL comes from the master-sheet column supplied by the caller
    and Tot.WaterTrials is counted from the corresponding behaviour files.

    One point per animal.  Pearson r + linear regression shown on the plot.
    When animal_group (dict: animal -> group label) is supplied, points are
    coloured by group and a legend is added.
    """

    # Apply 2-cell minimum
    if cell_counts is not None:
        animals = _apply_min_cells(animals, cell_counts, layer, measure)

    # --------------------------------------------------
    # 1. COLLECT RAW ANIMAL-LEVEL DATA
    # --------------------------------------------------
    x_raw, y_raw, grp_raw = [], [], []

    for a in animals:
        rs = reward_size.get(a, np.nan)
        if np.isnan(rs):
            continue
        if layer not in bar or a not in bar[layer]:
            continue
        if measure not in bar[layer][a]:
            continue
        mval = bar[layer][a][measure][0]
        if np.isnan(mval):
            continue
        x_raw.append(rs)
        y_raw.append(mval)
        grp_raw.append(animal_group.get(a, "Unknown") if animal_group else None)

    x_raw = np.array(x_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)

    mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
    x_raw    = x_raw[mask]
    y_raw    = y_raw[mask]
    grp_raw  = [g for g, m in zip(grp_raw, mask) if m]
    n_animals = len(x_raw)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_ylim(12, 18)
    ax.set_yticks(np.arange(12, 19, 2))
    ax.set_title(f"{lesion} | {layer} | {measure}")
    ax.set_xlabel("Reward Size (Water mL / Tot. Water Trials × 1000)")
    ax.set_ylabel(measure)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_animals < 3:
        ax.text(0.5, 0.5, f"Insufficient data (n={n_animals})",
                transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        return

    # --------------------------------------------------
    # 2. SCATTER — one point per animal, coloured by group
    # --------------------------------------------------
    if animal_group and any(g is not None for g in grp_raw):
        unique_groups = list(dict.fromkeys(grp_raw))  # stable insertion order
        cmap   = plt.cm.get_cmap("tab10", len(unique_groups))
        g_color = {g: cmap(i) for i, g in enumerate(unique_groups)}
        for g in unique_groups:
            idx = [i for i, gr in enumerate(grp_raw) if gr == g]
            ax.scatter(x_raw[idx], y_raw[idx],
                       color=g_color[g], alpha=0.75, s=40, zorder=3,
                       label=f"{g} (n={len(idx)})")
    else:
        ax.scatter(x_raw, y_raw,
                   color="black", alpha=0.65, s=30, zorder=3,
                   label=f"Animals (n={n_animals})")

    # --------------------------------------------------
    # 3. LINEAR FIT + STATS
    # --------------------------------------------------
    X1 = sm.add_constant(x_raw)
    m1 = sm.OLS(y_raw, X1).fit()

    xs   = np.linspace(np.nanmin(x_raw), np.nanmax(x_raw), 300)
    pred = m1.predict(sm.add_constant(xs))

    r, p = pearsonr(x_raw, y_raw)
    ls   = "-" if p < SIG_ALPHA else "--"
    ax.plot(xs, pred, color="black", linestyle=ls, linewidth=2, zorder=4,
            label=f"r={r:.2f}, p={p:.3g}")

    ax.legend(fontsize=8, loc="best")

    # --------------------------------------------------
    # 4. STAT ANNOTATION  (matches _water_prob_fig style)
    # --------------------------------------------------
    labels = ["B0 (intercept)", "B1 (Reward Size)"]
    lines  = ["-- Linear fit --",
              f"  {'Param':<22} {'Value':>9}  {'SE':>9}  {'t':>8}  {'p':>8}"]

    for label, val, se, tv, pv in zip(
        labels, m1.params, m1.bse, m1.tvalues, m1.pvalues
    ):
        sig = " *" if pv < SIG_ALPHA else "  "
        lines.append(
            f"  {label:<22} {val:>9.4f}  {se:>9.4f}  "
            f"{tv:>8.3f}  {pv:>8.4f}{sig}"
        )

    dof_m  = int(m1.df_model)
    dof_r  = int(m1.df_resid)
    r2     = float(m1.rsquared)
    r2_adj = float(m1.rsquared_adj)
    fval   = float(m1.fvalue)
    fp     = float(m1.f_pvalue)
    sig_f  = " *" if fp < SIG_ALPHA else "  "

    lines.append(
        f"  N={n_animals}  df(model)={dof_m}  df(resid)={dof_r}  "
        f"R²={r2:.4f}  Adj.R²={r2_adj:.3f}"
    )
    lines.append(
        f"  ANOVA: F({dof_m},{dof_r})={fval:.3f}  Prob>F={fp:.3f}{sig_f}"
    )

    fig.text(
        0.01, -0.25,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=6,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# =========================================================
# REWARD SIZE FIGURE
# =========================================================
def _expected_value_fig(
        lesion, layer, animals,
        expected_value, bar, measure,
        outpath,
        cell_counts=None,
        animal_group=None):
    """
    Plots Reward Size (X) vs Animal qEPSC / bar measure (Y).

    Reward Size per animal  =  (Water_mL / Tot.WaterTrials_across_all_days) * 1000
    where Water_mL comes from the master-sheet column supplied by the caller
    and Tot.WaterTrials is counted from the corresponding behaviour files.

    One point per animal.  Pearson r + linear regression shown on the plot.
    When animal_group (dict: animal -> group label) is supplied, points are
    coloured by group and a legend is added.
    """

    # Apply 2-cell minimum
    if cell_counts is not None:
        animals = _apply_min_cells(animals, cell_counts, layer, measure)

    # --------------------------------------------------
    # 1. COLLECT RAW ANIMAL-LEVEL DATA
    # --------------------------------------------------
    x_raw, y_raw, grp_raw = [], [], []

    for a in animals:
        ev = expected_value.get(a, np.nan)
        if np.isnan(ev):
            continue
        if layer not in bar or a not in bar[layer]:
            continue
        if measure not in bar[layer][a]:
            continue
        mval = bar[layer][a][measure][0]
        if np.isnan(mval):
            continue
        x_raw.append(ev)
        y_raw.append(mval)
        grp_raw.append(animal_group.get(a, "Unknown") if animal_group else None)

    x_raw = np.array(x_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)

    mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
    x_raw    = x_raw[mask]
    y_raw    = y_raw[mask]
    grp_raw  = [g for g, m in zip(grp_raw, mask) if m]
    n_animals = len(x_raw)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_ylim(12, 18)
    ax.set_yticks(np.arange(12, 19, 2))
    ax.set_title(f"{lesion} | {layer} | {measure}")
    ax.set_xlabel("Expected Value (Water mL x P(Water))")
    ax.set_ylabel(measure)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_animals < 3:
        ax.text(0.5, 0.5, f"Insufficient data (n={n_animals})",
                transform=ax.transAxes, ha="center", va="center")
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        return

    # --------------------------------------------------
    # 2. SCATTER — one point per animal, coloured by group
    # --------------------------------------------------
    if animal_group and any(g is not None for g in grp_raw):
        unique_groups = list(dict.fromkeys(grp_raw))  # stable insertion order
        cmap   = plt.cm.get_cmap("tab10", len(unique_groups))
        g_color = {g: cmap(i) for i, g in enumerate(unique_groups)}
        for g in unique_groups:
            idx = [i for i, gr in enumerate(grp_raw) if gr == g]
            ax.scatter(x_raw[idx], y_raw[idx],
                       color=g_color[g], alpha=0.75, s=40, zorder=3,
                       label=f"{g} (n={len(idx)})")
    else:
        ax.scatter(x_raw, y_raw,
                   color="black", alpha=0.65, s=30, zorder=3,
                   label=f"Animals (n={n_animals})")

    # --------------------------------------------------
    # 3. LINEAR FIT + STATS
    # --------------------------------------------------
    X1 = sm.add_constant(x_raw)
    m1 = sm.OLS(y_raw, X1).fit()

    xs   = np.linspace(np.nanmin(x_raw), np.nanmax(x_raw), 300)
    pred = m1.predict(sm.add_constant(xs))

    r, p = pearsonr(x_raw, y_raw)
    ls   = "-" if p < SIG_ALPHA else "--"
    ax.plot(xs, pred, color="black", linestyle=ls, linewidth=2, zorder=4,
            label=f"r={r:.2f}, p={p:.3g}")

    ax.legend(fontsize=8, loc="best")

    # --------------------------------------------------
    # 4. STAT ANNOTATION  (matches _water_prob_fig style)
    # --------------------------------------------------
    labels = ["B0 (intercept)", "B1 (Expected Value)"]
    lines  = ["-- Linear fit --",
              f"  {'Param':<22} {'Value':>9}  {'SE':>9}  {'t':>8}  {'p':>8}"]

    for label, val, se, tv, pv in zip(
        labels, m1.params, m1.bse, m1.tvalues, m1.pvalues
    ):
        sig = " *" if pv < SIG_ALPHA else "  "
        lines.append(
            f"  {label:<22} {val:>9.4f}  {se:>9.4f}  "
            f"{tv:>8.3f}  {pv:>8.4f}{sig}"
        )

    dof_m  = int(m1.df_model)
    dof_r  = int(m1.df_resid)
    r2     = float(m1.rsquared)
    r2_adj = float(m1.rsquared_adj)
    fval   = float(m1.fvalue)
    fp     = float(m1.f_pvalue)
    sig_f  = " *" if fp < SIG_ALPHA else "  "

    lines.append(
        f"  N={n_animals}  df(model)={dof_m}  df(resid)={dof_r}  "
        f"R²={r2:.3f}  Adj.R²={r2_adj:.3f}"
    )
    lines.append(
        f"  ANOVA: F({dof_m},{dof_r})={fval:.3f}  Prob>F={fp:.3f}{sig_f}"
    )

    fig.text(
        0.01, -0.25,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=6,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN ENTRY
# =========================================================

def run_correlation(df_master, animal_id_col, training_col,
                    bar_measure_cols, comparisons, training_data_dir,
                    output_folder, lesion_col=None, lesion_filter=None,
                    depth_col=None, depth_threshold=None,
                    water_ml_col=None):

    outdir      = os.path.join(output_folder, "_correlations")
    waterdir    = os.path.join(output_folder, "_water_prob")
    rewarddir   = os.path.join(output_folder, "_reward_size")
    evdir = os.path.join(output_folder, "_expected_value")

    os.makedirs(outdir,    exist_ok=True)
    os.makedirs(waterdir,  exist_ok=True)
    os.makedirs(rewarddir, exist_ok=True)
    os.makedirs(evdir, exist_ok=True)

    df = df_master.copy()

    groups = [_norm(g) for g in _extract_selected_groups(comparisons)]

    df[animal_id_col] = df[animal_id_col].astype(str)
    df[training_col]  = df[training_col].astype(str)
    df = df[df[training_col].isin(groups)]

    bar_measure_cols = [c for c in bar_measure_cols if c in df.columns]

    registry = defaultdict(dict)

    # -------------------------------------------------
    # BUILD REGISTRY
    # Store (files, group) so the group label is
    # available when computing water probability.
    # -------------------------------------------------
    for g in groups:
        gdf = df[df[training_col] == g]
        for a in gdf[animal_id_col].unique():
            a     = _norm(a)
            files = glob.glob(os.path.join(training_data_dir, a, "*.txt"))
            if files:
                registry[g][a] = (files, g)

    # -------------------------------------------------
    # FLATTEN  (animal -> (files, group))
    # -------------------------------------------------
    flat_registry = {}
    for g in registry:
        for a, (f, grp) in registry[g].items():
            flat_registry[a] = (f, grp)

    # -------------------------------------------------
    # LICK STATS + WATER PROBABILITY
    # -------------------------------------------------
    lick_cache        = {}
    water_prob        = {}
    total_water_trials = {}
    total_blank_trials = {}

    for a, (f, grp) in flat_registry.items():
        lick_cache[a]         = _extract_lick_stats(f, grp)
        water_prob[a]         = _compute_water_probability(f, grp)
        total_water_trials[a] = _compute_total_water_trials(f, grp)
        # Blank trials: always-water animals have 0 by design
        if _is_always_water(grp):
            total_blank_trials[a] = 0
        else:
            twt = total_water_trials[a]
            if not (isinstance(twt, float) and np.isnan(twt)):
                df_a = _load_and_merge(f)
                if df_a is not None and not df_a.empty:
                    trials_a = _segment_trials(df_a)
                    valid_a  = _valid_days(df_a)
                    if trials_a and valid_a:
                        t0_a = df_a["timestamp"].iloc[0]
                        # _segment_trials returns tuples: (seg_ts, seg_lick, day, is_water, ref)
                        _tfmt = (isinstance(trials_a[0], tuple) and len(trials_a[0]) == 5
                                 and isinstance(trials_a[0][0], np.ndarray))
                        total_all = sum(
                            1 for t in trials_a
                            if (t[2] if _tfmt else int((t["timestamp"].iloc[0] - t0_a) // 86400)) in valid_a
                        )
                        total_blank_trials[a] = max(0, total_all - int(twt))
                    else:
                        total_blank_trials[a] = np.nan
                else:
                    total_blank_trials[a] = np.nan
            else:
                total_blank_trials[a] = np.nan

    # -------------------------------------------------
    # REWARD SIZE  per animal
    #   = (Water_mL from master sheet / Tot.WaterTrials) * 1000
    # Only computed when water_ml_col is provided and present in df.
    # -------------------------------------------------
    reward_size    = {}
    water_ml_lookup = {}

    if water_ml_col and water_ml_col in df.columns:
        # Build a lookup: animal_id -> mean Water_mL across master-sheet rows
        water_ml_lookup = (
            df.groupby(animal_id_col)[water_ml_col]
            .apply(lambda s: np.nanmean(pd.to_numeric(s, errors="coerce")))
            .to_dict()
        )
        for a in flat_registry:
            wml = water_ml_lookup.get(a, np.nan)
            twt = total_water_trials.get(a, np.nan)
            if (not np.isnan(wml)) and (not np.isnan(twt)) and twt > 0:
                reward_size[a] = (wml / twt) * 1000
            else:
                reward_size[a] = np.nan

    bar, cell_counts = _build_bar(df, animal_id_col, bar_measure_cols,
                              depth_col, depth_threshold)
    # -------------------------------------------------
    # EXPECTED VALUE
    #   = Water_mL x P(Water)
    # -------------------------------------------------
    expected_value = {}

    for a in flat_registry:
        wp  = water_prob.get(a, np.nan)
        wml = water_ml_lookup.get(a, np.nan)

        if not np.isnan(wp) and not np.isnan(wml):
            expected_value[a] = wml * wp
        else:
            expected_value[a] = np.nan
    # -------------------------------------------------
    # LICK-RATE CORRELATION FIGURES  (one figure per comparison)
    # -------------------------------------------------
    for comp_idx, comp_groups in enumerate(comparisons):
        comp_group_labels = [_norm(g) for g in _extract_selected_groups([comp_groups])]
        # filter to groups that actually exist in the data
        comp_group_labels = [g for g in comp_group_labels if g in registry]
        if not comp_group_labels:
            continue

        comp_tag = "_vs_".join(comp_group_labels)

        for lesion in (
            (lesion_filter if lesion_filter else df[lesion_col].dropna().unique())
            if lesion_col and lesion_col in df.columns else [None]
        ):
            lesion_tag = str(lesion) if lesion is not None else "ALL"

            # Collect animals for this comparison only
            comp_animals = []
            for g in comp_group_labels:
                gdf = df[df[training_col] == g]
                if lesion is not None:
                    gdf = gdf[gdf[lesion_col] == lesion]
                sub_a = set(map(_norm, gdf[animal_id_col].astype(str).unique()))
                reg_a = set(map(_norm, registry[g].keys()))
                comp_animals += sorted(sub_a & reg_a)

            for layer in bar:
                for measure in bar_measure_cols:
                    out1 = os.path.join(
                        outdir, f"{comp_tag}_{lesion_tag}_{layer}_{measure}.png"
                    )
                    _make_fig(comp_tag, comp_animals, lick_cache, bar, measure,
                              layer, out1, cell_counts)

    # -------------------------------------------------
    # WATER PROBABILITY FIGURES  (per comparison, per lesion)
    # -------------------------------------------------
    waterdir_cell = os.path.join(output_folder, "_water_prob_bycell")
    os.makedirs(waterdir_cell, exist_ok=True)

    lesions_wp = (
        (lesion_filter if lesion_filter else list(df[lesion_col].dropna().unique()))
        if lesion_col and lesion_col in df.columns else [None]
    )

    for comp_idx, comp_groups in enumerate(comparisons):
        comp_group_labels = [_norm(g) for g in _extract_selected_groups([comp_groups])]
        comp_group_labels = [g for g in comp_group_labels if g in registry]
        if not comp_group_labels:
            continue

        comp_tag     = "_vs_".join(comp_group_labels)
        single_group = len(comp_group_labels) == 1

        for lesion in lesions_wp:
            lesion_tag = str(lesion) if lesion is not None else "ALL"

            subdf_base = df if lesion is None else df[df[lesion_col] == lesion]

            # Animals for this comparison
            comp_animals = []
            for g in comp_group_labels:
                gdf = subdf_base[subdf_base[training_col] == g]
                sub_a = set(map(_norm, gdf[animal_id_col].astype(str).unique()))
                reg_a = set(map(_norm, registry[g].keys()))
                comp_animals += sorted(sub_a & reg_a)

            # Filter subdf to only this comparison's animals
            subdf_comp = subdf_base[
                subdf_base[animal_id_col].astype(str).apply(_norm).isin(set(comp_animals))
            ]

            for layer in bar:
                for measure in bar_measure_cols:
                    out2 = os.path.join(
                        waterdir, f"{comp_tag}_{lesion_tag}_{layer}_{measure}.png"
                    )
                    _water_prob_fig(
                        f"{comp_tag} | {lesion_tag}", layer, comp_animals,
                        water_prob, bar, measure, out2,
                        cell_counts=cell_counts,
                        single_group=single_group,
                        animal_group={
                            a: g for g in comp_group_labels
                            for a in map(_norm,
                                subdf_base[subdf_base[training_col] == g
                                           ][animal_id_col].astype(str).unique())
                        },
                    )

                    out2c = os.path.join(
                        waterdir_cell,
                        f"{comp_tag}_{lesion_tag}_{layer}_{measure}.png"
                    )
                    _water_prob_fig(
                        f"{comp_tag} | {lesion_tag}", layer, comp_animals,
                        water_prob, bar, measure, out2c,
                        df_raw=subdf_comp,
                        animal_id_col=animal_id_col,
                        depth_col=depth_col,
                        depth_threshold=depth_threshold,
                        single_group=single_group,
                        animal_group={
                            a: g for g in comp_group_labels
                            for a in map(_norm,
                                subdf_base[subdf_base[training_col] == g
                                           ][animal_id_col].astype(str).unique())
                        },
                    )

    # -------------------------------------------------
    # REWARD SIZE FIGURES  (per comparison)
    # -------------------------------------------------
    lesions_rs = (
        (lesion_filter if lesion_filter else list(df[lesion_col].dropna().unique()))
        if lesion_col and lesion_col in df.columns else [None]
    )

    if water_ml_col and water_ml_col in df.columns:

        n_valid_rs = sum(1 for v in reward_size.values() if not np.isnan(v))
        print(f"[RewardSize] {n_valid_rs}/{len(reward_size)} animals "
              f"have a valid reward size value.")
        for a, v in reward_size.items():
            if np.isnan(v):
                twt = total_water_trials.get(a, np.nan)
                wml = water_ml_lookup.get(a, np.nan)
                print(f"  [SKIP] {a}: Water_mL={wml!r}  "
                      f"Tot.WaterTrials={twt!r}  -> RewardSize=NaN")

        for comp_idx, comp_groups in enumerate(comparisons):
            comp_group_labels = [_norm(g) for g in _extract_selected_groups([comp_groups])]
            comp_group_labels = [g for g in comp_group_labels if g in registry]
            if not comp_group_labels:
                continue

            comp_tag = "_vs_".join(comp_group_labels)

            # animal->group map for this comparison
            comp_animal_group = {}
            for g in comp_group_labels:
                for a in df[df[training_col] == g][animal_id_col].astype(str).apply(_norm).unique():
                    comp_animal_group[a] = g

            for lesion in lesions_rs:
                lesion_tag = str(lesion) if lesion is not None else "ALL"
                subdf = df if lesion is None else df[df[lesion_col] == lesion]

                comp_animals = []
                for g in comp_group_labels:
                    gdf = subdf[subdf[training_col] == g]
                    sub_a = set(map(_norm, gdf[animal_id_col].astype(str).unique()))
                    reg_a = set(map(_norm, registry[g].keys()))
                    comp_animals += sorted(sub_a & reg_a)

                for layer in bar:
                    for measure in bar_measure_cols:
                        out3 = os.path.join(
                            rewarddir, f"{comp_tag}_{lesion_tag}_{layer}_{measure}.png"
                        )
                        _reward_size_fig(
                            f"{comp_tag} | {lesion_tag}", layer, comp_animals,
                            reward_size, bar, measure, out3,
                            cell_counts=cell_counts,
                            animal_group=comp_animal_group,
                        )

    elif water_ml_col:
        print(f"[RewardSize] Column '{water_ml_col}' not found in master sheet. "
              f"Available columns: {list(df.columns)}")

    # -------------------------------------------------
    # EXPECTED VALUE FIGURES  (per comparison)
    # -------------------------------------------------
    for comp_idx, comp_groups in enumerate(comparisons):
        comp_group_labels = [_norm(g) for g in _extract_selected_groups([comp_groups])]
        comp_group_labels = [g for g in comp_group_labels if g in registry]
        if not comp_group_labels:
            continue

        comp_tag = "_vs_".join(comp_group_labels)

        comp_animal_group = {}
        for g in comp_group_labels:
            for a in df[df[training_col] == g][animal_id_col].astype(str).apply(_norm).unique():
                comp_animal_group[a] = g

        for lesion in lesions_rs:
            lesion_tag = str(lesion) if lesion is not None else "ALL"
            subdf = df if lesion is None else df[df[lesion_col] == lesion]

            comp_animals = []
            for g in comp_group_labels:
                gdf = subdf[subdf[training_col] == g]
                sub_a = set(map(_norm, gdf[animal_id_col].astype(str).unique()))
                reg_a = set(map(_norm, registry[g].keys()))
                comp_animals += sorted(sub_a & reg_a)

            for layer in bar:
                for measure in bar_measure_cols:
                    out_ev = os.path.join(
                        evdir, f"{comp_tag}_{lesion_tag}_{layer}_{measure}.png"
                    )
                    _expected_value_fig(
                        f"{comp_tag} | {lesion_tag}", layer, comp_animals,
                        expected_value, bar, measure, out_ev,
                        cell_counts=cell_counts,
                        animal_group=comp_animal_group,
                    )

    # -------------------------------------------------
    # EXCEL EXPORT  — single workbook for all graphs
    # -------------------------------------------------
    animals_by_group = {}
    for g in groups:
        gdf = df[df[training_col] == g]
        g_animals = sorted(
            set(map(_norm, gdf[animal_id_col].astype(str).unique()))
            & set(map(_norm, registry[g].keys()))
        )
        animals_by_group[g] = g_animals

    # Rebuild animal_group in case it was defined inside a conditional block
    _animal_group_export = {}
    for g in groups:
        gdf_tmp = df[df[training_col] == g]
        for a in gdf_tmp[animal_id_col].astype(str).apply(_norm).unique():
            _animal_group_export[a] = g

    export_correlation_excel(
        output_folder=output_folder,
        groups=groups,
        animals_by_group=animals_by_group,
        lick_cache=lick_cache,
        bar=bar,
        bar_measure_cols=bar_measure_cols,
        water_prob=water_prob,
        reward_size=reward_size,
        expected_value=expected_value,
        cell_counts=cell_counts,
        df_raw=df,
        animal_id_col=animal_id_col,
        depth_col=depth_col,
        depth_threshold=depth_threshold,
        animal_group=_animal_group_export,
        lesion_col=lesion_col,
        total_blank_trials=total_blank_trials,
        total_water_trials=total_water_trials,
        water_ml_lookup=water_ml_lookup,
    )



# =========================================================
# EXCEL EXPORT
# =========================================================

# ── style helpers ──────────────────────────────────────────────────────────────

_HEADER_FILL   = PatternFill("solid", start_color="2F5496", end_color="2F5496")
_HEADER_FONT   = Font(name="Arial", bold=True, color="FFFFFF", size=10)
_SUBHEAD_FILL  = PatternFill("solid", start_color="D9E1F2", end_color="D9E1F2")
_SUBHEAD_FONT  = Font(name="Arial", bold=True, size=10)
_BODY_FONT     = Font(name="Arial", size=10)
_THIN          = Side(style="thin", color="BFBFBF")
_THIN_BORDER   = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_SINGLE_CELL_FILL = PatternFill("solid", start_color="FF0000", end_color="FF0000")


def _hdr(ws, row, col, value):
    c = ws.cell(row=row, column=col, value=value)
    c.font  = _HEADER_FONT
    c.fill  = _HEADER_FILL
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = _THIN_BORDER
    return c


def _sub(ws, row, col, value):
    c = ws.cell(row=row, column=col, value=value)
    c.font  = _SUBHEAD_FONT
    c.fill  = _SUBHEAD_FILL
    c.alignment = Alignment(horizontal="center", vertical="center")
    c.border = _THIN_BORDER
    return c


def _val(ws, row, col, value):
    c = ws.cell(row=row, column=col, value=value)
    c.font   = _BODY_FONT
    c.border = _THIN_BORDER
    c.alignment = Alignment(horizontal="center")
    if isinstance(value, float) and not np.isnan(value):
        c.number_format = "0.0000"
    return c


def _val_red(ws, row, col, value):
    """Like _val but fills the cell red (n=1 cell average warning)."""
    c = _val(ws, row, col, value)
    c.fill = _SINGLE_CELL_FILL
    c.font = Font(name="Arial", size=10, color="FFFFFF", bold=True)
    return c


def _auto_width(ws, min_w=8, max_w=30):
    for col_cells in ws.columns:
        length = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in col_cells
        )
        ws.column_dimensions[get_column_letter(col_cells[0].column)].width = \
            min(max(length + 2, min_w), max_w)


# ── sheet builders ─────────────────────────────────────────────────────────────

def _sheet_lick_correlation(wb, groups, animals_by_group,
                             lick_cache, bar, bar_measure_cols, cell_counts,
                             animal_lesion=None):
    """
    Sheet: Lick-Rate Correlations
    One row per animal. Each row contains the animal's licking Hz for every
    (condition × window) combination, followed by the L2 and L3 bar averages.
    Columns: Lesion | Group | Animal | <cond_win Lick_Hz> ... | L2_Bar_Mean | L2_Bar_SEM | L3_Bar_Mean | L3_Bar_SEM
    """
    ws = wb.create_sheet("Lick-Rate Correlations")

    # Build column headers: fixed metadata + one lick column per (cond, window)
    fixed_headers = ["Lesion", "Group", "Animal"]
    lick_col_keys = []  # (cond, win_name, win_def)
    for cond in CONDITIONS:
        for win_name, win_def in zip(WIN_NAMES, WIN_DEFS):
            lick_col_keys.append((cond, win_name, win_def))

    lick_headers = [f"{cond}_{win} Hz" for cond, win, _ in lick_col_keys]

    layer_headers = []
    for layer in (bar.keys() if bar else []):
        layer_headers += [f"{layer}_Bar_Mean", f"{layer}_Bar_SEM"]

    all_headers = fixed_headers + lick_headers + layer_headers
    for c, h in enumerate(all_headers, 1):
        _hdr(ws, 1, c, h)

    row = 2
    seen_animals = set()
    for g, animals in animals_by_group.items():
        # Collect all valid animals across all layers/measures for this group
        valid_set = set(animals)
        if cell_counts:
            for layer in bar:
                for measure in bar_measure_cols:
                    valid_set &= set(_apply_min_cells(animals, cell_counts, layer, measure))
        valid_animals = [
            a for a in animals
            if a in valid_set
            and a in lick_cache and lick_cache[a] is not None
            and a not in seen_animals
        ]
        for a in valid_animals:
            seen_animals.add(a)
            lesion_val = (animal_lesion or {}).get(a, "")
            row_data = [lesion_val, g, a]

            # Lick Hz columns
            for cond, win_name, win_def in lick_col_keys:
                lmean = None
                if lick_cache[a] and (cond, win_def) in lick_cache[a]:
                    v = lick_cache[a][(cond, win_def)][0]
                    lmean = None if (isinstance(v, float) and np.isnan(v)) else v
                row_data.append(lmean)

            # L2 then L3 bar averages
            layer_avg_info = []  # (bmean, bsem, single_cell)
            for layer in (bar.keys() if bar else []):
                if layer in bar and a in bar[layer]:
                    means = [bar[layer][a][m][0] for m in bar_measure_cols
                             if m in bar[layer][a] and not np.isnan(bar[layer][a][m][0])]
                    sems  = [bar[layer][a][m][1] for m in bar_measure_cols
                             if m in bar[layer][a] and not np.isnan(bar[layer][a][m][1])]
                    bmean = float(np.nanmean(means)) if means else None
                    bsem  = float(np.nanmean(sems))  if sems  else None
                    n_cells = [cell_counts.get(layer, {}).get(a, {}).get(m, 0)
                               for m in bar_measure_cols
                               if m in (bar.get(layer, {}).get(a, {}))]
                    single_cell = bool(n_cells) and all(n == 1 for n in n_cells)
                else:
                    bmean, bsem, single_cell = None, None, False
                layer_avg_info.append((bmean, bsem, single_cell))
                row_data += [bmean, bsem]

            # Write fixed + lick columns normally
            lick_end_col = len(fixed_headers) + len(lick_col_keys)
            for col_i, v in enumerate(row_data[:lick_end_col], 1):
                _val(ws, row, col_i, v)

            # Write L2/L3 mean+sem with red fill when n==1
            col_i = lick_end_col + 1
            for bmean, bsem, single_cell in layer_avg_info:
                writer = _val_red if single_cell else _val
                writer(ws, row, col_i,     bmean)
                writer(ws, row, col_i + 1, bsem)
                col_i += 2

            row += 1

    _auto_width(ws)
    ws.freeze_panes = "A2"


def _sheet_water_prob(wb, animals, water_prob, bar, bar_measure_cols,
                       cell_counts, df_raw=None, animal_id_col=None,
                       depth_col=None, depth_threshold=None,
                       animal_lesion=None, animal_group=None):
    """
    Sheet: Water Probability (by cell)
    By-cell mode: one row per (layer, measure, cell/row) with Lesion and
    Group columns added.
    Animal-average fallback: one row per (layer, measure, animal).
    """

    ws = wb.create_sheet("Water Prob by Cell")

    if df_raw is not None:
        headers = ["Lesion", "Group", "Layer", "Measure", "Animal", "P(Water)_%",
                   "Bar_Value (cell-level)"]
    else:
        headers = ["Lesion", "Group", "Layer", "Measure", "Animal", "P(Water)_%",
                   "Bar_Mean", "Bar_SEM"]

    for c, h in enumerate(headers, 1):
        _hdr(ws, 1, c, h)

    row = 2
    for layer in bar:
        for measure in bar_measure_cols:
            valid = _apply_min_cells(animals, cell_counts, layer, measure) \
                if cell_counts else animals

            if df_raw is not None and animal_id_col:
                sub = df_raw.copy()
                sub[animal_id_col] = sub[animal_id_col].astype(str).apply(_norm)
                if depth_col and depth_col in sub.columns:
                    if layer == "L2":
                        sub = sub[sub[depth_col] < depth_threshold]
                    elif layer == "L3":
                        sub = sub[sub[depth_col] >= depth_threshold]
                for _, r in sub.iterrows():
                    a  = _norm(r[animal_id_col])
                    if a not in valid:
                        continue
                    wp = water_prob.get(a, np.nan)
                    if np.isnan(wp):
                        continue
                    mval = pd.to_numeric(r.get(measure, np.nan), errors="coerce")
                    if np.isnan(mval):
                        continue
                    lesion_val = (animal_lesion or {}).get(a, "")
                    group_val  = (animal_group  or {}).get(a, "")
                    for c_i, v in enumerate(
                        [lesion_val, group_val, layer, measure, a, round(wp * 100, 4), float(mval)], 1
                    ):
                        _val(ws, row, c_i, v)
                    row += 1
            else:
                for a in valid:
                    wp = water_prob.get(a, np.nan)
                    if np.isnan(wp):
                        continue
                    if layer not in bar or a not in bar[layer]:
                        continue
                    if measure not in bar[layer][a]:
                        continue
                    bmean, bsem = bar[layer][a][measure]
                    if np.isnan(bmean):
                        continue
                    lesion_val = (animal_lesion or {}).get(a, "")
                    group_val  = (animal_group  or {}).get(a, "")
                    for c_i, v in enumerate(
                        [lesion_val, group_val, layer, measure, a, round(wp * 100, 4), bmean, bsem], 1
                    ):
                        _val(ws, row, c_i,
                             None if (isinstance(v, float) and np.isnan(v)) else v)
                    row += 1

    _auto_width(ws)
    ws.freeze_panes = "A2"


def _sheet_water_prob_by_animal(wb, animals, water_prob, animal_group=None):
    """
    Sheet: Water Probability by Animal
    One row per animal — no layer/measure breakdown since P(Water) is
    an animal-level value, not cell-level.
    Columns: Animal | Group | P(Water)_%
    """

    ws = wb.create_sheet("Water Prob by Animal")

    headers = ["Animal", "Group","layer", "measure", "P(Water)_%"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 1, c, h)

    row = 2
    for a in sorted(animals):
        wp = water_prob.get(a, np.nan)
        if isinstance(wp, float) and np.isnan(wp):
            continue
        grp = animal_group.get(a, "") if animal_group else ""
        for c_i, v in enumerate(
            [a, grp, round(wp * 100, 4)], 1
        ):
            _val(ws, row, c_i, v)
        row += 1

    _auto_width(ws)
    ws.freeze_panes = "A2"


def _sheet_reward_size(wb, animals, reward_size, bar, bar_measure_cols,
                        cell_counts, animal_group=None):
    """
    Sheet: Reward Size
    One row per (layer, measure, animal).
    """

    ws = wb.create_sheet("Reward Size")

    headers = ["Layer", "Measure", "Animal", "Group",
               "Reward_Size (µL)", "Bar_Mean", "Bar_SEM"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 1, c, h)

    row = 2
    for layer in bar:
        for measure in bar_measure_cols:
            valid = _apply_min_cells(animals, cell_counts, layer, measure) \
                if cell_counts else animals
            for a in valid:
                rs = reward_size.get(a, np.nan)
                if np.isnan(rs):
                    continue
                if layer not in bar or a not in bar[layer]:
                    continue
                if measure not in bar[layer][a]:
                    continue
                bmean, bsem = bar[layer][a][measure]
                if np.isnan(bmean):
                    continue
                grp = animal_group.get(a, "") if animal_group else ""
                for c_i, v in enumerate(
                    [layer, measure, a, grp, rs, bmean, bsem], 1
                ):
                    _val(ws, row, c_i,
                         None if (isinstance(v, float) and np.isnan(v)) else v)
                row += 1

    _auto_width(ws)
    ws.freeze_panes = "A2"


def _sheet_expected_value(wb, animals, expected_value, bar, bar_measure_cols,
                           cell_counts, animal_group=None, water_prob=None):
    """
    Sheet: Expected Value
    One row per (layer, measure, animal).
    """

    ws = wb.create_sheet("Expected Value")

    headers = ["Layer", "Measure", "Animal", "Group",
               "P(Water)_%", "Expected_Value", "Bar_Mean", "Bar_SEM"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 1, c, h)

    row = 2
    for layer in bar:
        for measure in bar_measure_cols:
            valid = _apply_min_cells(animals, cell_counts, layer, measure) \
                if cell_counts else animals
            for a in valid:
                ev = expected_value.get(a, np.nan)
                if np.isnan(ev):
                    continue
                if layer not in bar or a not in bar[layer]:
                    continue
                if measure not in bar[layer][a]:
                    continue
                bmean, bsem = bar[layer][a][measure]
                if np.isnan(bmean):
                    continue
                grp = animal_group.get(a, "") if animal_group else ""
                wp_pct = round(water_prob[a] * 100, 4) \
                    if (water_prob and a in water_prob
                        and not np.isnan(water_prob[a])) else None
                for c_i, v in enumerate(
                    [layer, measure, a, grp, wp_pct, ev, bmean, bsem], 1
                ):
                    _val(ws, row, c_i,
                         None if (isinstance(v, float) and np.isnan(v)) else v)
                row += 1

    _auto_width(ws)
    ws.freeze_panes = "A2"


def _sheet_per_lesion(wb, lesion, animals, animal_group, water_prob,
                       water_ml_lookup, total_water_trials, total_blank_trials,
                       reward_size, expected_value, bar, cell_counts=None,
                       bar_measure_cols=None):
    """
    One tab per lesion.
    Columns: Animal | Training Group | P(W) | Water (mL) | Total Water Trials |
             Total Blank Trials | RS | Expected Value | L2 Animal Avg | L3 Animal Avg
    L2/L3 averages are means across all bar_measure_cols for that layer.
    Blank cells when data not available.
    """
    # Sanitise lesion name for sheet title (Excel 31-char limit, no special chars)
    safe_name = str(lesion)[:28].replace("/", "-").replace("\\", "-").replace("*", "").replace("?", "").replace("[", "").replace("]", "").replace(":", "-")
    ws = wb.create_sheet(safe_name)

    headers = ["Animal", "Training Group", "P(W)", "Water (mL)",
               "Total Water Trials", "Total Blank Trials", "RS",
               "Expected Value", "L2 Animal Avg", "L3 Animal Avg"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 1, c, h)

    row = 2
    for a in sorted(animals):
        grp = (animal_group or {}).get(a, "")
        wp  = water_prob.get(a, np.nan) if water_prob else np.nan
        wp_val = round(wp * 100, 4) if (wp is not None and not np.isnan(wp)) else None

        wml = (water_ml_lookup or {}).get(a, np.nan)
        wml_val = None if (wml is None or (isinstance(wml, float) and np.isnan(wml))) else wml

        twt = (total_water_trials or {}).get(a, np.nan)
        twt_val = None if (twt is None or (isinstance(twt, float) and np.isnan(twt))) else twt

        tbt = (total_blank_trials or {}).get(a, np.nan)
        tbt_val = None if (tbt is None or (isinstance(tbt, float) and np.isnan(tbt))) else tbt

        rs = (reward_size or {}).get(a, np.nan)
        rs_val = None if (rs is None or (isinstance(rs, float) and np.isnan(rs))) else rs

        ev = (expected_value or {}).get(a, np.nan)
        ev_val = None if (ev is None or (isinstance(ev, float) and np.isnan(ev))) else ev

        # L2 average across all measures
        l2_avg = None
        l2_single = False
        measures_used = bar_measure_cols or []
        if bar and "L2" in bar and a in bar["L2"]:
            vals = [v[0] for v in bar["L2"][a].values()
                    if not (isinstance(v[0], float) and np.isnan(v[0]))]
            l2_avg = float(np.nanmean(vals)) if vals else None
            n_cells_l2 = [cell_counts.get("L2", {}).get(a, {}).get(m, 0)
                          for m in measures_used if m in bar["L2"].get(a, {})]
            l2_single = bool(n_cells_l2) and all(n == 1 for n in n_cells_l2)

        # L3 average across all measures
        l3_avg = None
        l3_single = False
        if bar and "L3" in bar and a in bar["L3"]:
            vals = [v[0] for v in bar["L3"][a].values()
                    if not (isinstance(v[0], float) and np.isnan(v[0]))]
            l3_avg = float(np.nanmean(vals)) if vals else None
            n_cells_l3 = [cell_counts.get("L3", {}).get(a, {}).get(m, 0)
                          for m in measures_used if m in bar["L3"].get(a, {})]
            l3_single = bool(n_cells_l3) and all(n == 1 for n in n_cells_l3)

        # Write fixed columns (cols 1-8)
        for col_i, v in enumerate(
            [a, grp, wp_val, wml_val, twt_val, tbt_val, rs_val, ev_val], 1
        ):
            _val(ws, row, col_i, v)
        # Write L2/L3 averages with red fill when n==1
        ((_val_red if l2_single else _val)(ws, row, 9,  l2_avg))
        ((_val_red if l3_single else _val)(ws, row, 10, l3_avg))
        row += 1

    _auto_width(ws)
    ws.freeze_panes = "A2"


def _sheet_readme(wb):
    ws = wb.create_sheet("README", 0)
    lines = [
        ("Correlation Module — Data Export", True),
        ("", False),
        ("This workbook contains the raw data underlying every figure produced by", False),
        ("run_correlation().  Tab layout:", False),
        ("", False),
        ("Tab 1 — README (this sheet)", True),
        ("  Describes all tabs and column definitions.", False),
        ("", False),
        ("Tab 2 — Lick-Rate Correlations", True),
        ("  One row per animal.  All licking Hz values for that animal appear in", False),
        ("  a single row (wide format), followed by L2 and L3 bar averages.", False),
        ("  Columns: Lesion | Group | Animal |", False),
        ("    <condition>_<window> Hz  (one column per condition × window) |", False),
        ("    L2_Bar_Mean | L2_Bar_SEM | L3_Bar_Mean | L3_Bar_SEM", False),
        ("", False),
        ("Tab 3 — Water Prob by Cell", True),
        ("  One row per cell (master-sheet row) in by-cell mode, or one row per", False),
        ("  animal in animal-average mode.  Lesion column included.", False),
        ("  Columns: Lesion | Group | Layer | Measure | Animal | P(Water)_% | Bar_Value", False),
        ("", False),
        ("Tab 4+ — Per-Lesion Tabs  (one tab per unique lesion value)", True),
        ("  Columns: Animal | Training Group | P(W) | Water (mL) |", False),
        ("    Total Water Trials | Total Blank Trials | RS |", False),
        ("    Expected Value | L2 Animal Avg | L3 Animal Avg", False),
        ("  L2/L3 Animal Avg = mean bar value across all measures for that layer.", False),
        ("  Cells are blank when data are not available for that animal.", False),
        ("", False),
        ("Notes", True),
        ("  • All values are means ± SEM across cells for that animal & layer.", False),
        ("  • P(W) / P(Water) is expressed as a percentage (0–100).", False),
        ("  • RS (Reward Size) = (Water_mL / Total_Water_Trials) × 1000  (µL/trial).", False),
        ("  • Expected_Value = Water_mL × P(Water).", False),
        ("  • Animals with < 2 valid cells for a given layer+measure are excluded.", False),
        ("  • If an animal has no L2 or L3 data the corresponding cells are blank.", False),
    ]
    ws.column_dimensions["A"].width = 80
    for r, (text, bold) in enumerate(lines, 1):
        c = ws.cell(row=r, column=1, value=text)
        c.font = Font(name="Arial", bold=bold, size=10)


# ── public entry point ─────────────────────────────────────────────────────────

def export_correlation_excel(
        output_folder,
        groups,
        animals_by_group,   # dict: group -> list[animal_id]
        lick_cache,
        bar,
        bar_measure_cols,
        water_prob,
        reward_size,
        expected_value,
        cell_counts,
        df_raw=None,
        animal_id_col=None,
        depth_col=None,
        depth_threshold=None,
        animal_group=None,
        lesion_col=None,
        total_blank_trials=None,
        total_water_trials=None,
        water_ml_lookup=None,
        filename="correlation_data.xlsx",
):
    """
    Write a single Excel workbook with one sheet per graph type so that
    all figures produced by run_correlation() can be replotted externally.

    Parameters
    ----------
    output_folder     : str  — same folder passed to run_correlation()
    groups            : list[str]
    animals_by_group  : dict  group -> [animal_id, ...]
    lick_cache        : dict  animal -> lick stats dict (from _extract_lick_stats)
    bar               : dict  layer -> animal -> measure -> (mean, sem)
    bar_measure_cols  : list[str]
    water_prob        : dict  animal -> float
    reward_size       : dict  animal -> float
    expected_value    : dict  animal -> float
    cell_counts       : dict  layer -> animal -> measure -> int
    df_raw            : pd.DataFrame | None   — master sheet (for by-cell water-prob)
    animal_id_col     : str | None
    depth_col         : str | None
    depth_threshold   : float | None
    animal_group      : dict | None   animal -> group label
    filename          : str
    """
    wb = Workbook()
    # Remove default sheet
    del wb[wb.sheetnames[0]]

    _sheet_readme(wb)

    all_animals = sorted({a for g_animals in animals_by_group.values()
                           for a in g_animals})

    # Build lesion lookup: animal -> lesion label
    animal_lesion = {}
    if df_raw is not None and animal_id_col and lesion_col and lesion_col in df_raw.columns:
        for _, row in df_raw[[animal_id_col, lesion_col]].drop_duplicates().iterrows():
            a = _norm(str(row[animal_id_col]))
            animal_lesion[a] = str(row[lesion_col]) if pd.notna(row[lesion_col]) else ""

    _sheet_lick_correlation(
        wb, groups, animals_by_group,
        lick_cache, bar, bar_measure_cols, cell_counts,
        animal_lesion=animal_lesion,
    )

    _sheet_water_prob(
        wb, all_animals, water_prob, bar, bar_measure_cols, cell_counts,
        df_raw=df_raw, animal_id_col=animal_id_col,
        depth_col=depth_col, depth_threshold=depth_threshold,
        animal_lesion=animal_lesion, animal_group=animal_group,
    )

    # Collect unique lesions for per-lesion tabs
    if animal_lesion:
        all_lesions = sorted({v for v in animal_lesion.values() if v})
    elif df_raw is not None and lesion_col and lesion_col in df_raw.columns:
        all_lesions = sorted(df_raw[lesion_col].dropna().astype(str).unique())
    else:
        all_lesions = []

    for lesion in all_lesions:
        lesion_animals = [a for a in all_animals if animal_lesion.get(a, "") == lesion]
        _sheet_per_lesion(
            wb, lesion, lesion_animals, animal_group, water_prob,
            water_ml_lookup, total_water_trials, total_blank_trials,
            reward_size, expected_value, bar,
            cell_counts=cell_counts,
            bar_measure_cols=bar_measure_cols,
        )

    outpath = os.path.join(output_folder, filename)
    wb.save(outpath)
    print(f"[ExcelExport] Saved → {outpath}")
    return outpath