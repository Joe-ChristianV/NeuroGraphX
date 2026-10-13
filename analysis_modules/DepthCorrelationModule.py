# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 07:29:52 2026

@author: JoeCh
"""

"""
DepthCorrelationModule.py
Regression Discontinuity Tree (RDT) analysis.

For each (training_group, lesion) combination present in the comparisons,
a full RDT pipeline is run and three figures are returned:
  - scatter : depth vs measurement with regression line + cutoff
  - tree    : decision tree diagram  (requires MIN_RDT_N samples)
  - bar     : bar + jitter chart     (requires MIN_RDT_N samples)

Sample-size gating
------------------
  MIN_SCATTER_N   (default 2) : minimum to attempt a scatter + linear fit.
  MIN_RDT_N       (default 10): minimum to run tree, bootstrap, permutation,
                                CV, and bar chart.  Below this threshold the
                                scatter is still produced; tree_fig / bar_fig
                                are returned as None and a warning is emitted
                                via the `warnings` list in each result dict.

Coloring convention
-------------------
  fill  (facecolor) : group_colors[training_group]
  border (edgecolor): lesion_colors[lesion_value]   (if lesion col selected)
"""

import warnings as _warnings_module
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.stats import mannwhitneyu, linregress, ttest_ind


# ---------------------------------------------------------------------------
# Thresholds  (change here or pass overrides to run_rdt_analysis)
# ---------------------------------------------------------------------------
MIN_SCATTER_N = 2    # need at least 2 points for linregress
MIN_RDT_N     = 10   # need enough for CV folds, bootstrap, and split stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(messages: list, text: str):
    """Append a diagnostic message and also emit a Python warning."""
    messages.append(text)
    _warnings_module.warn(text, stacklevel=3)


def _safe_linregress(x, y):
    """
    Wrapper around scipy linregress that returns NaNs instead of raising
    when x has zero variance (all depths identical) or n < 2.
    """
    if len(x) < 2 or np.std(x) == 0:
        return 0.0, float(np.mean(y)), float("nan"), float("nan"), float("nan")
    return linregress(x, y)


def _bootstrap_ci(model, X, y, n_bootstrap=1000):
    estimates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y), len(y), replace=True)
        model.fit(X[idx], y[idx])
        estimates.append(model.predict(X).mean())
    return np.percentile(estimates, 2.5), np.percentile(estimates, 97.5)


def _permutation_test_tstat(X_flat, y, cutoff, n_permutations=1000):
    mask_above = X_flat > cutoff
    mask_below = ~mask_above
    y_above, y_below = y[mask_above], y[mask_below]

    if len(y_above) < 2 or len(y_below) < 2:
        return 1.0, 0.0, 0.0

    se = np.sqrt(y_above.var(ddof=1) / len(y_above) +
                 y_below.var(ddof=1) / len(y_below))
    orig_t = (y_above.mean() - y_below.mean()) / se if se > 0 else 0.0

    perm_tstats = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        ya, yb = y_perm[mask_above], y_perm[mask_below]
        if len(ya) < 2 or len(yb) < 2:
            perm_tstats.append(0.0)
            continue
        se_p = np.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
        perm_tstats.append((ya.mean() - yb.mean()) / se_p if se_p > 0 else 0.0)

    p_val = (np.sum(np.abs(perm_tstats) >= np.abs(orig_t)) + 1) / (n_permutations + 1)
    tau = y_above.mean() - y_below.mean()
    return p_val, orig_t, tau


def _fit_tree(X, y, n_cv_folds=5):
    """
    Fit a decision tree with cross-validated depth selection.
    Automatically reduces cv folds when sample size is small.
    """
    n = len(y)
    cv = min(n_cv_folds, n)          # never ask for more folds than samples
    max_d = max(1, min(10, n // 2))  # don't allow depth > n/2

    param_grid = {"max_depth": range(1, max_d + 1)}
    grid = GridSearchCV(
        DecisionTreeRegressor(),
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X, y)
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _make_bar_fig(group_below, group_above, training_group,
                  fill_color, mw_pval, outcome_col,
                  label_below="L2", label_above="L3", dpi=600):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(2, 2), dpi=dpi)

    avg_below = group_below.mean()
    avg_above = group_above.mean()

    ax.bar([label_below, label_above],
           [avg_below, avg_above],
           color=fill_color, alpha=0.5, edgecolor=fill_color)

    # Jittered black scatter points
    for i, grp_data in enumerate([group_below.values, group_above.values]):
        x = np.random.normal(i, 0.05, size=len(grp_data))
        ax.scatter(x, grp_data, color="black", alpha=0.5, s=15, zorder=3)

    # MW significance bar
    y_sig = 21.5
    ax.plot([0, 1], [y_sig, y_sig], color="black", lw=0.75)
    ax.text(0.5, y_sig, f"MW p={mw_pval:.3f}",
            ha="center", va="bottom", fontsize=4, color="black")

    # Mean value labels
    for i, m in enumerate([avg_below, avg_above]):
        ax.text(i, 10.2, f"{m:.2f}",
                ha="center", va="bottom", fontsize=6)

    # Sample size labels
    for i, grp_data in enumerate([group_below, group_above]):
        ax.text(i, 10.2 + 0.8, f"n={len(grp_data)}",
                ha="center", va="bottom", fontsize=6, color="black")

    ax.set_ylim(10, 22)
    ax.set_yticks(np.arange(10, 23, 4))
    ax.tick_params(axis="both", labelsize=6, width=1)
    ax.set_xlabel("", fontsize=8)
    ax.set_ylabel(f"{training_group} {outcome_col}", fontsize=8)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def _make_scatter_fig(df_subset, running_col, outcome_col,
                      chosen_cutoff, slope, intercept, p_value,
                      fill_color, edge_color, title,
                      show_cutoff=True):
    """
    Always produces a figure.  
    - If p_value is NaN the legend shows 'p=N/A'.
    - chosen_cutoff line is only drawn when show_cutoff=True.
    """
    fig = Figure(figsize=(2, 2), dpi=600)
    ax = fig.add_subplot(111)

    ax.scatter(df_subset[running_col].values,
               df_subset[outcome_col].values,
               color=fill_color,
               edgecolors=edge_color,
               linewidths=0.8,
               alpha=0.8, s=20, zorder=3)

    if show_cutoff and chosen_cutoff is not None:
        ax.axvline(x=chosen_cutoff, color="black", alpha=0.6,
                   linestyle="--", linewidth=0.5,
                   label=f"Cutoff: {chosen_cutoff:.2f} µm")

    x_vals = df_subset[running_col].values
    if len(x_vals) >= MIN_SCATTER_N and not np.isnan(slope):
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        p_label = f"p={p_value:.4f}" if not np.isnan(p_value) else "p=N/A"
        ax.plot(x_line, intercept + slope * x_line,
                color="red", linestyle="--", linewidth=0.5,
                label=f"Linear fit ({p_label})")

    # Attempt fixed axis limits; fall back to auto if data is outside range
    x_data_min, x_data_max = x_vals.min(), x_vals.max()
    y_data_min, y_data_max = (df_subset[outcome_col].values.min(),
                               df_subset[outcome_col].values.max())

    ax.set_xlim(min(100, x_data_min * 0.95), max(450, x_data_max * 1.05))
    ax.set_ylim(min(10, y_data_min * 0.95),  max(22, y_data_max * 1.05))

    ax.set_xticks(np.arange(100, 451, 75))
    ax.set_yticks(np.arange(10, 23, 4))

    ax.tick_params(axis="both", labelsize=6, width=1)
    ax.set_xlabel(f"{running_col}", color="black", fontsize=8)
    ax.set_ylabel(f"{outcome_col}", color="black", fontsize=8)
    ax.set_title(title, color="black", fontsize=8)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels_l = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(facecolor="white", edgecolor="black",
                           labelcolor="black", fontsize=4)
        for line in legend.get_lines():
            line.set_linewidth(0.5)

    fig.tight_layout()
    return fig


def _make_tree_fig(best_tree, running_col, title):
    # plot_tree requires a pyplot-managed figure for get_renderer
    plt.ioff()
    fig, ax = plt.subplots(figsize=(2, 1.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("blue")
    plot_tree(best_tree,
              feature_names=[running_col],
              filled=True, rounded=True,
              ax=ax, fontsize=4)
    ax.set_title(title, color="black", fontsize=6)
    fig.tight_layout()
    plt.close(fig)   # detach from pyplot; Figure object stays alive
    return fig


# ---------------------------------------------------------------------------
# Public: find_optimal_cutoff  (called by GUI before full run to show user)
# ---------------------------------------------------------------------------

def find_optimal_cutoff(df: pd.DataFrame,
                        running_col: str,
                        outcome_col: str) -> float:
    X = df[running_col].values.reshape(-1, 1)
    y = df[outcome_col].values
    return float(_fit_tree(X, y).tree_.threshold[0])


# ---------------------------------------------------------------------------
# Public: run_rdt_analysis
# ---------------------------------------------------------------------------

def run_rdt_analysis(df: pd.DataFrame,
                     running_col: str,
                     outcome_col: str,
                     training_col,
                     lesion_col,
                     comparisons: list,
                     group_colors: dict,
                     lesion_colors: dict,
                     cutoff=None,
                     use_rdt_cutoff: bool = True,
                     n_bootstrap: int = 1000,
                     n_permutations: int = 1000,
                     label_below: str = "L2",
                     label_above: str = "L3",
                     min_scatter_n: int = MIN_SCATTER_N,
                     min_rdt_n: int = MIN_RDT_N) -> list:
    """
    Run a full RDT for every (training_group, lesion_value) combination
    found in the selected comparisons.

    Parameters
    ----------
    df            : Full (optionally depth-filtered) DataFrame from GUI.
    running_col   : Depth column (X / running variable).
    outcome_col   : Measurement column (Y / outcome).
    training_col  : Training-group column name, or None.
    lesion_col    : Lesion column name, or None.
    comparisons   : List of lists of group label strings,
                    e.g. [["SAT1", "SAT2"], ["SAT1", "ACC"]].
    group_colors  : {group_label: color} from group_colors.txt.
    lesion_colors : {lesion_label: color} from lesion_colors.txt.
    cutoff        : Manual cutoff value (used when use_rdt_cutoff=False).
    use_rdt_cutoff: If True, use the tree's optimal threshold.
    n_bootstrap   : Bootstrap iterations.
    n_permutations: Permutation test iterations.
    label_below   : X-tick label for below-cutoff group.
    label_above   : X-tick label for above-cutoff group.
    min_scatter_n : Minimum n to attempt scatter + linear fit (default 2).
    min_rdt_n     : Minimum n to run tree / bar / full stats (default 10).

    Returns
    -------
    List of dicts, one per (group, lesion) combo, each with:
        "label"         : str  e.g. "SAT1 – lesion_A"
        "group"         : training group label
        "lesion"        : lesion value or None
        "scatter_fig"   : Figure  (None if n < min_scatter_n)
        "tree_fig"      : Figure  (None if n < min_rdt_n)
        "bar_fig"       : Figure  (None if n < min_rdt_n)
        "optimal_cutoff": float   (None if tree not run)
        "chosen_cutoff" : float   (None if tree not run)
        "stats"         : dict of computed statistics (partial if RDT skipped)
        "warnings"      : list[str] of human-readable diagnostic messages
        "n"             : int  number of valid observations in this subset
    """
    DEFAULT_FILL = "#A3BE8C"
    DEFAULT_EDGE = "#ECEFF4"

    # Flatten all groups from all comparisons, preserving order, no duplicates
    seen = set()
    all_groups = []
    for comp in comparisons:
        for g in comp:
            if g not in seen:
                seen.add(g)
                all_groups.append(g)

    # Lesion values to iterate over
    if lesion_col and lesion_col in df.columns:
        lesion_values = sorted(df[lesion_col].dropna().astype(str).unique())
    else:
        lesion_values = [None]

    results = []

    for training_group in all_groups:
        for lesion_val in lesion_values:

            warn_msgs = []  # collects all diagnostics for this combo

            # ── subset ───────────────────────────────────────────────────
            mask = pd.Series([True] * len(df), index=df.index)
            if training_col and training_col in df.columns:
                mask &= df[training_col].astype(str) == str(training_group)
            if lesion_val is not None:
                mask &= df[lesion_col].astype(str) == str(lesion_val)

            df_sub = df[mask].dropna(subset=[running_col, outcome_col]).copy()
            n = len(df_sub)

            label = str(training_group)
            if lesion_val is not None:
                label += f" – {lesion_val}"

            # ── too few for anything ──────────────────────────────────────
            if n < min_scatter_n:
                msg = (f"[{label}] SKIPPED entirely: only {n} valid "
                       f"observation(s) after filtering "
                       f"(need ≥ {min_scatter_n} for a scatter plot).")
                _log(warn_msgs, msg)
                results.append({
                    "label": label, "group": training_group,
                    "lesion": lesion_val,
                    "scatter_fig": None, "tree_fig": None, "bar_fig": None,
                    "optimal_cutoff": None, "chosen_cutoff": None,
                    "stats": {}, "warnings": warn_msgs, "n": n,
                })
                continue

            X = df_sub[running_col].values.reshape(-1, 1)
            y = df_sub[outcome_col].values

            # ── colors ───────────────────────────────────────────────────
            fill_color = group_colors.get(str(training_group), DEFAULT_FILL)
            edge_color = (lesion_colors.get(str(lesion_val), DEFAULT_EDGE)
                          if lesion_val is not None else fill_color)

            # ── linear regression (always attempted) ─────────────────────
            slope, intercept, r_value, p_value, std_err = _safe_linregress(
                X.flatten(), y)

            if np.isnan(p_value):
                _log(warn_msgs,
                     f"[{label}] Linear regression p-value is NaN "
                     f"(likely zero variance in {running_col}). "
                     f"Fit line will be omitted from scatter.")

            # ── RDT-capable branch ────────────────────────────────────────
            rdt_available = n >= min_rdt_n

            if not rdt_available:
                msg = (f"[{label}] Partial output (n={n}): scatter + linear "
                       f"fit produced, but tree / bar chart / full RDT stats "
                       f"require ≥ {min_rdt_n} observations. "
                       f"Tree and bar figures will be None.")
                _log(warn_msgs, msg)

            optimal_cutoff = None
            chosen_cutoff  = None
            best_tree      = None
            cv_scores      = None
            lb = ub        = None

            if rdt_available:
                try:
                    best_tree      = _fit_tree(X, y)
                    optimal_cutoff = float(best_tree.tree_.threshold[0])
                    chosen_cutoff  = (optimal_cutoff if use_rdt_cutoff
                                      else float(cutoff))
                    lb, ub         = _bootstrap_ci(best_tree, X, y,
                                                    n_bootstrap)
                    cv_scores      = cross_val_score(
                        best_tree, X, y,
                        cv=min(5, n),
                        scoring="neg_mean_squared_error")
                except Exception as exc:
                    msg = (f"[{label}] Tree fitting failed despite n={n}: "
                           f"{exc}. Tree / bar figures will be None.")
                    _log(warn_msgs, msg)
                    rdt_available = False

            # If manual cutoff was requested but RDT failed, try to honour it
            if not rdt_available and cutoff is not None:
                chosen_cutoff = float(cutoff)

            # ── group stats (only when we have a cutoff and enough data) ──
            below = above = None
            tau = mw_p = t_stat = perm_p = orig_t = None

            if chosen_cutoff is not None:
                below = df_sub[df_sub[running_col] <= chosen_cutoff][outcome_col]
                above = df_sub[df_sub[running_col] >  chosen_cutoff][outcome_col]

                if len(below) < 2 or len(above) < 2:
                    msg = (f"[{label}] Cutoff {chosen_cutoff:.2f} creates a "
                           f"split with < 2 samples on one side "
                           f"(below n={len(below)}, above n={len(above)}). "
                           f"Group stats / bar chart will be skipped.")
                    _log(warn_msgs, msg)
                    below = above = None

            if below is not None:
                tau              = above.mean() - below.mean()
                _, mw_p          = mannwhitneyu(below, above,
                                                alternative="two-sided")
                t_stat, _        = ttest_ind(above, below, equal_var=False)
                perm_p, orig_t, _ = _permutation_test_tstat(
                    X.flatten(), y, chosen_cutoff, n_permutations)

            # ── figures ───────────────────────────────────────────────────
            scatter_fig = _make_scatter_fig(
                df_sub, running_col, outcome_col,
                chosen_cutoff, slope, intercept, p_value,
                fill_color, edge_color,
                title=f"RDT: {label}",
                show_cutoff=(chosen_cutoff is not None),
            )

            tree_fig = None
            bar_fig  = None

            if rdt_available and best_tree is not None:
                try:
                    tree_fig = _make_tree_fig(
                        best_tree, running_col,
                        title=f"Decision Tree: {label}",
                    )
                except Exception as exc:
                    _log(warn_msgs,
                         f"[{label}] Tree figure generation failed: {exc}")

            if below is not None and rdt_available:
                try:
                    bar_fig = _make_bar_fig(
                        below, above,
                        training_group=str(training_group),
                        fill_color=fill_color,
                        mw_pval=mw_p,
                        outcome_col=outcome_col,
                        label_below=label_below,
                        label_above=label_above,
                    )
                except Exception as exc:
                    _log(warn_msgs,
                         f"[{label}] Bar figure generation failed: {exc}")

            # ── stats dict ────────────────────────────────────────────────
            stats = {
                "n": n,
                "slope":        slope,
                "intercept":    intercept,
                "r_value":      r_value,
                "linreg_p":     p_value,
                "cv_mse_mean":  cv_scores.mean() if cv_scores is not None
                                else None,
                "bootstrap_ci": (lb, ub) if lb is not None else None,
                "mean_below":   below.mean()  if below is not None else None,
                "mean_above":   above.mean()  if above is not None else None,
                "sem_below":    below.sem()   if below is not None else None,
                "sem_above":    above.sem()   if above is not None else None,
                "n_below":      len(below)    if below is not None else None,
                "n_above":      len(above)    if above is not None else None,
                "tau":          tau,
                "mw_p":         mw_p,
                "t_stat":       t_stat,
                "perm_p":       perm_p,
            }

            results.append({
                "label":          label,
                "group":          training_group,
                "lesion":         lesion_val,
                "scatter_fig":    scatter_fig,
                "tree_fig":       tree_fig,
                "bar_fig":        bar_fig,
                "optimal_cutoff": optimal_cutoff,
                "chosen_cutoff":  chosen_cutoff,
                "stats":          stats,
                "warnings":       warn_msgs,
                "n":              n,
            })

    return results