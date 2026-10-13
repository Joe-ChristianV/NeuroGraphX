# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 13:55:16 2025

@author: JoeCh
"""

# analysis_modules/BarGraphModule.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
import hashlib

# ---------------- Name Shortening: Safe for Excel & File Saving ----------------
def shorten_name(name, max_len=31):
    if name is None:
        return ""
    name = str(name)
    if len(name) <= max_len:
        return name
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"{name[:max_len - 7]}_{hash_suffix}"

# ---------------- Load colors ----------------
def load_colors(path):
    if not path or not os.path.exists(path):
        return {}
    colors = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            colors[key.strip()] = val.strip()
    return colors

# ---------------- Helper: color lookup ----------------
def get_bar_color(group_name, group_colors):
    training_part = group_name.split("-")[-1]
    return group_colors.get(training_part, "gray")

def get_edge_color(group_name, lesion_colors, use_lesion=True):
    if use_lesion:
        lesion_part = group_name.split("-")[0]
        return lesion_colors.get(lesion_part, "black")
    else:
        return "gray"


# ──────────────────────────────────────────────────────────────────────────────
# BROKEN AXIS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _add_break_marks(ax_bottom, ax_top, d=0.015):
    """
    Draw diagonal // break marks on the LEFT spine only
    between ax_bottom (top edge) and ax_top (bottom edge).
    No marks on the right side.
    """
    # Bottom axis — left side only
    kwargs = dict(transform=ax_bottom.transAxes, color='k',
                  clip_on=False, linewidth=0.75)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Top axis — left side only
    kwargs.update(transform=ax_top.transAxes)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)


def _needs_break(check_arr, y_max=22):
    """Return True if any value in check_arr exceeds y_max."""
    if len(check_arr) == 0:
        return False
    return float(np.max(check_arr)) > y_max


def _outlier_tick(check_arr, y_max=22):
    """
    Return the single extra tick needed above the break.
    Rounds up to the next integer above the highest outlier value.
    e.g. max=24.5 → tick=25
    """
    data_max = float(np.max(check_arr))
    return int(np.ceil(data_max))


# ──────────────────────────────────────────────────────────────────────────────
# CORE DRAWING — shared by both plot functions
# ──────────────────────────────────────────────────────────────────────────────

def _draw_bars(ax, categories, values, data_points, errors,
               bar_width, x_pos, get_color_fn, get_ec_fn, mean_y=10):
    """
    Draw bars, error bars, jittered scatter, and mean text onto `ax`.
    Called once (single axis) or twice (once per broken sub-axis).
    matplotlib clips everything to the current axis limits automatically.
    """
    for i, g in enumerate(categories):
        val       = values[i]
        bar_color = get_color_fn(g)
        ec        = get_ec_fn(g)

        ax.bar(x_pos[i], val, color=bar_color, alpha=0.6,
               width=bar_width, align='center')
        ax.bar(x_pos[i], val, color='none', edgecolor=ec,
               linewidth=1, width=bar_width, align='center')

        # Error bar (upward only, SEM style)
        if errors and g in errors:
            err   = errors[g]
            top   = val + err
            cap_w = bar_width * 0.3
            ax.vlines(x_pos[i], val, top, color=ec, linewidth=1.5)
            ax.hlines(top,
                      x_pos[i] - cap_w / 2,
                      x_pos[i] + cap_w / 2,
                      color=ec, linewidth=1.5)

        # Jittered scatter
        if data_points and g in data_points:
            pts = np.asarray(data_points[g])
            ax.scatter(
                x_pos[i] + (np.random.rand(len(pts)) - 0.5) * bar_width * 0.5,
                pts, s=6, color=ec, alpha=0.85, zorder=3
            )

        # Mean value text — only draw where it fits (bottom panel / single panel)
        lo, hi = ax.get_ylim()
        if lo <= mean_y <= hi:
            ax.text(x_pos[i], mean_y, f"{val:.3g}",
                    ha="center", va="bottom", fontsize=4, color="black")


# ──────────────────────────────────────────────────────────────────────────────
# BROKEN AXIS FIGURE BUILDER
# Same total figure size as the normal plot. The top panel is a small sliver
# containing only 1 extra tick (the rounded-up max). The bottom panel gets
# the remaining height ratio so the normal range looks identical.
# ──────────────────────────────────────────────────────────────────────────────

def _make_broken_figure(fig_w, fig_h, dpi, y_min, y_max_normal, top_tick):
    """
    Create a figure with two axes of the same total size as the normal figure.
    The top panel is a small sliver (1 tick worth of space relative to the
    bottom panel's range), the bottom panel fills the rest.

    Returns (fig, ax_top, ax_bot)
    """
    normal_range = y_max_normal - y_min          # e.g. 12  (10→22)
    top_range    = top_tick - y_max_normal        # e.g.  3  (22→25)

    # Height ratio: bottom gets normal_range units, top gets top_range units
    # but we cap the top ratio so it never dominates
    top_ratio    = top_range / normal_range       # e.g. 0.25
    bottom_ratio = 1.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),          # ← identical size to normal plot
        dpi=dpi,
        gridspec_kw={
            "height_ratios": [top_ratio, bottom_ratio],
            "hspace": 0.0,               # no gap — spines touch
        }
    )
    return fig, ax_top, ax_bot


def _style_broken_axes(ax_top, ax_bot):
    """Apply spine / tick styling for a broken-axis pair."""
    for ax in (ax_top, ax_bot):
        ax.spines['right'].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.75)
        ax.grid(False)

    # Top panel: remove BOTH horizontal spines
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['top'].set_visible(False)   # ← ADD THIS LINE
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Bottom panel: remove top spine (break line)
    ax_bot.spines['top'].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# bar_plot  — training-group comparison bars
# ──────────────────────────────────────────────────────────────────────────────

def bar_plot(categories, values, data_points=None, errors=None,
             title="", xlabel="", ylabel="",
             group_colors=None, dpi=600, edge_colors=None):

    display_labels = [g.split("-")[-1] for g in categories]

    def sort_key(x):
        try:
            return -float(x)
        except Exception:
            return float("inf")

    sorted_idx     = sorted(range(len(display_labels)),
                            key=lambda i: sort_key(display_labels[i]))
    categories     = [categories[i]     for i in sorted_idx]
    display_labels = [display_labels[i] for i in sorted_idx]
    values         = [values[i]         for i in sorted_idx]
    if errors:
        errors = {categories[i]: errors[categories[i]] for i in range(len(categories))}
    if data_points:
        data_points = {categories[i]: data_points[categories[i]] for i in range(len(categories))}

    n         = len(categories)
    bar_width = 0.30
    spacing   = 0.10
    x_pos     = np.arange(n) * (bar_width + spacing)

    def _color(g):
        return get_bar_color(g, group_colors) if group_colors else "gray"
    def _ec(g):
        return edge_colors[g] if edge_colors and g in edge_colors else _color(g)

    # Collect all points + error tops to decide whether a break is needed
    all_pts  = (np.concatenate([np.asarray(v) for v in data_points.values()])
                if data_points else np.array([]))
    err_tops = np.array([np.mean(data_points[g]) + (errors[g] if errors and g in errors else 0)
                         for g in categories if data_points and g in data_points])
    check    = np.concatenate([all_pts, err_tops]) if len(err_tops) else all_pts

    Y_MIN        = 10
    Y_MAX_NORMAL = 22
    fig_w        = n * bar_width + (n - 1) * spacing
    fig_h        = 1.5   # always the same height

    # ── n-label helper ────────────────────────────────────────────────────────
    def _draw_n_labels(ax_target):
        for i, g in enumerate(categories):
            n_val = len(data_points[g]) if data_points and g in data_points else "n/a"
            ax_target.text(x_pos[i], -0.2, f"n{n_val}",
                           ha="center", va="top", fontsize=6,
                           transform=ax_target.get_xaxis_transform())

    def _set_xticks(ax_target, labels):
        ax_target.set_xticks(x_pos)
        ax_target.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)

    if not _needs_break(check, Y_MAX_NORMAL):
        # ── Standard single-axis ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.75)
        ax.grid(False)

        ax.set_ylim(Y_MIN, Y_MAX_NORMAL)
        ax.set_yticks(np.arange(Y_MIN, Y_MAX_NORMAL + 1, 4))
        _draw_bars(ax, categories, values, data_points, errors,
                   bar_width, x_pos, _color, _ec, mean_y=Y_MIN)
        _draw_n_labels(ax)
        _set_xticks(ax, display_labels)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='x', labelsize=7)

        return fig, ax, x_pos

    else:
        # ── Broken-axis ───────────────────────────────────────────────────────
        top_tick = _outlier_tick(check, Y_MAX_NORMAL)   # e.g. 25 if max=24.5

        fig, ax_top, ax_bot = _make_broken_figure(
            fig_w, fig_h, dpi, Y_MIN, Y_MAX_NORMAL, top_tick)

        _style_broken_axes(ax_top, ax_bot)

        # Bottom panel: normal range with normal ticks
        ax_bot.set_ylim(Y_MIN, Y_MAX_NORMAL)
        ax_bot.set_yticks(np.arange(Y_MIN, Y_MAX_NORMAL + 1, 4))

        # Top panel: just the one extra tick above the break
        # Give it a little breathing room above the tick
        BREAK_PAD = 0.8   # controls how far above 22 the break sits
        
        # Bottom axis stops slightly ABOVE the break
        ax_bot.set_ylim(Y_MIN, Y_MAX_NORMAL + BREAK_PAD)
        
        # Top axis starts slightly ABOVE the break
        ax_top.set_ylim(Y_MAX_NORMAL + BREAK_PAD, top_tick + 0)

        # Draw onto both panels (matplotlib clips to each axis range)
        for ax in (ax_top, ax_bot):
            _draw_bars(ax, categories, values, data_points, errors,
                       bar_width, x_pos, _color, _ec, mean_y=Y_MIN)

        # Break marks on left spine only
        _add_break_marks(ax_bot, ax_top)

        # x labels + n labels on bottom panel only
        _set_xticks(ax_bot, display_labels)
        ax_top.set_xticks(x_pos)
        ax_top.set_xticklabels([], fontsize=0)
        _draw_n_labels(ax_bot)

        ax_bot.set_xlabel(xlabel, fontsize=8)
        fig.text(-0.15, 0.5, ylabel, va="center", rotation="vertical", fontsize=8)
        ax_top.set_title(title, fontsize=6)
        for ax in (ax_top, ax_bot):
            ax.tick_params(axis='y', labelsize=6)
            ax.tick_params(axis='x', labelsize=7)

        return fig, ax_bot, x_pos


# ──────────────────────────────────────────────────────────────────────────────
# lesion_bar_plot  — lesion comparison bars
# ──────────────────────────────────────────────────────────────────────────────

def lesion_bar_plot(categories, values, data_points=None, errors=None,
                    title="", xlabel="", ylabel="",
                    lesion_colors=None, dpi=600):

    def normalize_label(x):
        return str(x).strip().replace(" ", "").lower()

    def lesion_sort_key(x):
        norm = normalize_label(x)
        return (0, "") if norm == "vehicle" else (1, norm)

    sorted_idx = sorted(range(len(categories)),
                        key=lambda i: lesion_sort_key(categories[i]))
    categories = [categories[i] for i in sorted_idx]
    values     = [values[i]     for i in sorted_idx]
    if errors:
        errors = {categories[i]: errors[categories[i]] for i in range(len(categories))}
    if data_points:
        data_points = {categories[i]: data_points[categories[i]] for i in range(len(categories))}

    n         = len(categories)
    bar_width = 0.30
    spacing   = 0.10
    x_pos     = np.arange(n) * (bar_width + spacing)
    lc        = lesion_colors or {}

    def _color(g):
        return lc.get(str(g), "gray")
    def _ec(g):
        return _color(g)

    all_pts  = (np.concatenate([np.asarray(v) for v in data_points.values()])
                if data_points else np.array([]))
    err_tops = np.array([np.mean(data_points[g]) + (errors[g] if errors and g in errors else 0)
                         for g in categories if data_points and g in data_points])
    check    = np.concatenate([all_pts, err_tops]) if len(err_tops) else all_pts

    Y_MIN        = 10
    Y_MAX_NORMAL = 22
    fig_w        = max(1.0, n * bar_width + (n - 1) * spacing)
    fig_h        = 1.5   # always the same height

    def _draw_n_labels(ax_target):
        for i, g in enumerate(categories):
            n_val = len(data_points[g]) if data_points and g in data_points else "n/a"
            ax_target.text(x_pos[i], -0.2, f"n{n_val}",
                           ha="center", va="top", fontsize=6,
                           transform=ax_target.get_xaxis_transform())

    def _set_xticks(ax_target, labels):
        ax_target.set_xticks(x_pos)
        ax_target.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)

    if not _needs_break(check, Y_MAX_NORMAL):
        # ── Standard single-axis ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.75)
        ax.grid(False)

        ax.set_ylim(Y_MIN, Y_MAX_NORMAL)
        ax.set_yticks(np.arange(Y_MIN, Y_MAX_NORMAL + 1, 4))
        _draw_bars(ax, categories, values, data_points, errors,
                   bar_width, x_pos, _color, _ec, mean_y=Y_MIN)
        _draw_n_labels(ax)
        _set_xticks(ax, [str(c) for c in categories])
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='x', labelsize=7)

        return fig, ax, x_pos

    else:
        # ── Broken-axis ───────────────────────────────────────────────────────
        top_tick = _outlier_tick(check, Y_MAX_NORMAL)

        fig, ax_top, ax_bot = _make_broken_figure(
            fig_w, fig_h, dpi, Y_MIN, Y_MAX_NORMAL, top_tick)

        _style_broken_axes(ax_top, ax_bot)

        ax_bot.set_ylim(Y_MIN, Y_MAX_NORMAL)
        ax_bot.set_yticks(np.arange(Y_MIN, Y_MAX_NORMAL + 1, 4))
        
        # Top panel: just the one extra tick above the break
        # Give it a little breathing room above the tick
        BREAK_PAD = 0.8   # controls how far above 22 the break sits
        
        # Bottom axis stops slightly ABOVE the break
        ax_bot.set_ylim(Y_MIN, Y_MAX_NORMAL + BREAK_PAD)
        
        # Top axis starts slightly ABOVE the break
        ax_top.set_ylim(Y_MAX_NORMAL + BREAK_PAD, top_tick + 0)
        # ax_top.set_ylim(Y_MAX_NORMAL, top_tick + 0.5)
        # ax_top.set_yticks([top_tick])

        for ax in (ax_top, ax_bot):
            _draw_bars(ax, categories, values, data_points, errors,
                       bar_width, x_pos, _color, _ec, mean_y=Y_MIN)

        _add_break_marks(ax_bot, ax_top)

        _set_xticks(ax_bot, [str(c) for c in categories])
        ax_top.set_xticks(x_pos)
        ax_top.set_xticklabels([], fontsize=0)
        _draw_n_labels(ax_bot)

        ax_bot.set_xlabel(xlabel, fontsize=8)
        fig.text(-0.15, 0.5, ylabel, va="center", rotation="vertical", fontsize=8)
        ax_top.set_title(title, fontsize=6)
        for ax in (ax_top, ax_bot):
            ax.tick_params(axis='y', labelsize=6)
            ax.tick_params(axis='x', labelsize=7)

        return fig, ax_bot, x_pos


# ---------------- Statistics -----------------
def run_stats(data_dict):
    groups = list(data_dict.keys())
    values = [data_dict[g] for g in groups if len(data_dict[g]) > 0]
    results = []

    if len(groups) < 2:
        return "Not enough groups", pd.DataFrame()

    if len(groups) == 2:
        g1, g2 = groups
        stat, p = mannwhitneyu(values[0], values[1], alternative="two-sided")
        results.append({"Test": "Mann-Whitney", "Groups": f"{g1} vs {g2}",
                         "Statistic": stat, "p": p})
    else:
        fval, pval = f_oneway(*values)
        results.append({"Test": "ANOVA", "Groups": ", ".join(groups),
                         "Statistic": fval, "p": pval})

        pairs, pvals = [], []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                stat, p = ttest_ind(data_dict[g1], data_dict[g2], equal_var=False)
                pairs.append(f"{g1} vs {g2}")
                pvals.append(p)

        reject_holm, pvals_holm, _, _ = multipletests(pvals, method="holm")
        for pair, pc in zip(pairs, pvals_holm):
            results.append({"Test": "Sidak-Holm", "Groups": pair,
                             "Statistic": np.nan, "p": pc})

        reject_bonf, pvals_bonf, _, _ = multipletests(pvals, method="bonferroni")
        for pair, pc in zip(pairs, pvals_bonf):
            results.append({"Test": "Bonferroni", "Groups": pair,
                             "Statistic": np.nan, "p": pc})

    return "Stats computed", pd.DataFrame(results)


# ---------------- Table image helper ----------------
def _save_table_image(df_table, filename, title, dpi=600):
    fig, ax = plt.subplots(
        figsize=(max(6, len(df_table) * 0.8), max(2, len(df_table) * 0.5)),
        dpi=dpi)
    ax.axis('tight')
    ax.axis('off')
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.text(0.5, 1.02, title, fontsize=10, fontweight='bold',
            ha='center', va='bottom', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — lesion-vs-lesion bar graphs per training group
# ══════════════════════════════════════════════════════════════════════════════

def _run_lesion_bargraphs(df, measurement_col, training_col,
                          lesion_col, depth_col, depth_threshold,
                          animal_id_col, lesion_colors, group_colors,
                          base_plot_dir, writer, comparisons, dpi=600):

    training_values = sorted({
        str(g).split("-")[-1]
        for comp in comparisons
        for g in comp
    })
    lesion_values_from_comp = sorted({
        str(g).split("-")[0]
        for comp in comparisons
        for g in comp
        if "-" in str(g)
    })
    lesion_values = (lesion_values_from_comp
                     if lesion_values_from_comp
                     else sorted(df[lesion_col].dropna().unique().tolist(), key=str))

    split_by_depth = (depth_col and depth_col in df.columns
                      and not df[depth_col].dropna().empty)
    depth_labels = ["L2", "L3"] if split_by_depth else [None]

    print(f"\n=== LESION BAR GRAPHS (per training group) ===")
    print(f"  training_values (from comparisons): {training_values}")
    print(f"  lesion_values   (from comparisons): {lesion_values}")

    for train_name in training_values:
        for depth_label in depth_labels:
            print(f"\n--- train='{train_name}' | depth='{depth_label}' ---")

            cell_data   = {}
            animal_data = {}

            for lesion in lesion_values:
                lesion_tag = str(lesion)
                subset = df[df[training_col].astype(str) == str(train_name)].copy()
                subset = subset[subset[lesion_col].astype(str) == lesion_tag]

                if split_by_depth and depth_label:
                    if depth_label == "L2":
                        subset = subset[subset[depth_col] < depth_threshold]
                    else:
                        subset = subset[subset[depth_col] >= depth_threshold]

                if subset.empty:
                    print(f"  [{lesion_tag}] empty — skipped")
                    continue

                vals = subset[measurement_col].dropna().values
                if len(vals) == 0:
                    continue
                cell_data[lesion_tag] = vals
                print(f"  [{lesion_tag}] {len(vals)} cell(s)")

                if animal_id_col and animal_id_col in subset.columns:
                    per_animal = (subset.groupby(animal_id_col)[measurement_col]
                                  .mean().dropna())
                    if len(per_animal) > 0:
                        animal_data[lesion_tag] = per_animal.values

            if len(cell_data) < 2:
                print("  SKIPPED: fewer than 2 lesion groups with data")
                continue

            fig_key  = f"LesionComp_{train_name}_{depth_label}"
            safe_key = shorten_name(fig_key)

            _, res_df = run_stats(cell_data)

            fig_cell, _, _ = lesion_bar_plot(
                categories=list(cell_data.keys()),
                values=[np.mean(cell_data[k]) for k in cell_data],
                data_points=cell_data,
                errors={k: np.std(cell_data[k], ddof=1) / np.sqrt(len(cell_data[k]))
                        for k in cell_data},
                title=f"{train_name} {depth_label} Cell (lesions)",
                ylabel=measurement_col,
                lesion_colors=lesion_colors,
                dpi=dpi,
            )
            cell_path = os.path.join(base_plot_dir, f"{safe_key}_graph.png")
            fig_cell.savefig(cell_path, dpi=dpi, bbox_inches='tight', transparent=True)
            plt.close(fig_cell)
            print(f"  Cell bar saved: {cell_path}")

            if animal_data:
                _, res_df_animal = run_stats(animal_data)

                fig_animal, _, _ = lesion_bar_plot(
                    categories=list(animal_data.keys()),
                    values=[np.mean(animal_data[k]) for k in animal_data],
                    data_points=animal_data,
                    errors={k: np.std(animal_data[k], ddof=1) / np.sqrt(len(animal_data[k]))
                            for k in animal_data},
                    title=f"{train_name} {depth_label} Animal (lesions)",
                    ylabel=measurement_col,
                    lesion_colors=lesion_colors,
                    dpi=dpi,
                )
                animal_path = os.path.join(base_plot_dir, f"{safe_key}_N.png")
                fig_animal.savefig(animal_path, dpi=dpi, bbox_inches='tight', transparent=True)
                plt.close(fig_animal)
                print(f"  Animal bar saved: {animal_path}")

            summary_rows = [{"Lesion": k, "N": len(v), "Mean": np.mean(v),
                              "SEM": np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0}
                             for k, v in cell_data.items()]
            summary_df = pd.DataFrame(summary_rows)
            num_cols = summary_df.select_dtypes(include=[np.number]).columns
            summary_df[num_cols] = summary_df[num_cols].apply(
                lambda x: x.map(lambda y: float(f"{y:.3g}")))

            _save_table_image(summary_df,
                              os.path.join(base_plot_dir, f"{safe_key}_table.png"),
                              title=f"{train_name} {depth_label} Lesion Summary", dpi=dpi)
            summary_df.to_excel(writer,
                                sheet_name=shorten_name(f"{safe_key}_Summary"),
                                index=False)

            if not res_df.empty:
                stats_df = res_df.copy()
                num_cols = stats_df.select_dtypes(include=[np.number]).columns
                stats_df[num_cols] = stats_df[num_cols].apply(
                    lambda x: x.map(lambda y: float(f"{y:.3g}")))
                _save_table_image(stats_df,
                                  os.path.join(base_plot_dir, f"{safe_key}_stats_table.png"),
                                  title=f"{train_name} {depth_label} Lesion Stats", dpi=dpi)
                stats_df.to_excel(writer,
                                  sheet_name=shorten_name(f"{safe_key}_Stats"),
                                  index=False)

            if animal_data:
                summary_rows_a = [{"Lesion": k, "N": len(v), "Mean": np.mean(v),
                                    "SEM": np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0}
                                   for k, v in animal_data.items()]
                summary_df_a = pd.DataFrame(summary_rows_a)
                num_cols = summary_df_a.select_dtypes(include=[np.number]).columns
                summary_df_a[num_cols] = summary_df_a[num_cols].apply(
                    lambda x: x.map(lambda y: float(f"{y:.3g}")))
                _save_table_image(summary_df_a,
                                  os.path.join(base_plot_dir, f"{safe_key}_N_table.png"),
                                  title=f"{train_name} {depth_label} Lesion N Summary", dpi=dpi)
                summary_df_a.to_excel(writer,
                                      sheet_name=shorten_name(f"{safe_key}_N_Summary"),
                                      index=False)

                if not res_df_animal.empty:
                    stats_df_a = res_df_animal.copy()
                    num_cols = stats_df_a.select_dtypes(include=[np.number]).columns
                    stats_df_a[num_cols] = stats_df_a[num_cols].apply(
                        lambda x: x.map(lambda y: float(f"{y:.3g}")))
                    _save_table_image(stats_df_a,
                                      os.path.join(base_plot_dir, f"{safe_key}_N_stats_table.png"),
                                      title=f"{train_name} {depth_label} Lesion N Stats", dpi=dpi)
                    stats_df_a.to_excel(writer,
                                        sheet_name=shorten_name(f"{safe_key}_N_Stats"),
                                        index=False)


# ---------------- Main Analysis -----------------
def run_bargraph_analysis(df, measurement_col, training_col,
                          depth_col=None, depth_threshold=None,
                          animal_id_col=None, comparisons=None,
                          lesion_col=None,
                          group_colors_path=None,
                          lesion_colors_path=None,
                          output_dir=None):

    group_colors  = load_colors(group_colors_path)  if group_colors_path  else {}
    lesion_colors = load_colors(lesion_colors_path) if lesion_colors_path else {}

    all_results   = []
    base_plot_dir = os.path.join(output_dir, "Plots") if output_dir else "Plots"
    os.makedirs(base_plot_dir, exist_ok=True)

    split_by_depth = depth_col in df.columns and not df[depth_col].dropna().empty
    use_lesion     = bool(lesion_col and lesion_col in df.columns)

    excel_path = os.path.join(base_plot_dir, "Bar.xlsx")
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:

        # ══════════════════════════════════════════════════════════════════════
        # SECTION A — training-group comparisons
        # ══════════════════════════════════════════════════════════════════════
        for comp in comparisons:
            lesion_groups = df[lesion_col].dropna().unique() if use_lesion else [None]
            for lesion in lesion_groups:
                lesion_tag   = str(lesion) if lesion is not None else None
                depth_labels = ["L2", "L3"] if split_by_depth else [None]

                for depth_label in depth_labels:
                    data_dict        = {}
                    data_dict_animal = {}

                    for g in comp:
                        subset = df.copy()
                        if lesion_tag:
                            subset = subset[subset[lesion_col].astype(str) == lesion_tag]
                        train_name = g.split("-")[-1]
                        subset = subset[subset[training_col].astype(str) == train_name]
                        if split_by_depth:
                            if depth_label == "L2":
                                subset = subset[subset[depth_col] < depth_threshold]
                            elif depth_label == "L3":
                                subset = subset[subset[depth_col] >= depth_threshold]
                        if subset.empty:
                            continue

                        grp_name = ""
                        if lesion_tag:
                            grp_name += lesion_tag + "-"
                        if depth_label:
                            grp_name += depth_label + "-"
                        grp_name += train_name

                        vals = subset[measurement_col].dropna().values
                        if len(vals) == 0:
                            continue
                        data_dict[grp_name] = vals

                        if animal_id_col and animal_id_col in subset.columns:
                            per_animal = subset.groupby(animal_id_col)[measurement_col].mean().dropna()
                            if len(per_animal) > 0:
                                data_dict_animal[grp_name] = per_animal.values

                    if not data_dict:
                        continue

                    _, res_df        = run_stats(data_dict)
                    _, res_df_animal = run_stats(data_dict_animal)

                    edge_colors = {}
                    for grp in data_dict.keys():
                        if lesion_tag:
                            edge_colors[grp] = lesion_colors.get(lesion_tag, "black")
                        else:
                            edge_colors[grp] = get_bar_color(grp, group_colors)

                    raw_name  = "_".join(filter(None, [lesion_tag, depth_label] +
                                                [g.split("-")[-1] for g in comp])).replace(" ", "_")
                    safe_name = shorten_name(raw_name)

                    fig_graph, ax_graph, _ = bar_plot(
                        categories=list(data_dict.keys()),
                        values=[np.mean(data_dict[k]) for k in data_dict],
                        data_points=data_dict,
                        errors={k: np.std(data_dict[k], ddof=1) / np.sqrt(len(data_dict[k]))
                                for k in data_dict},
                        title=f"{lesion_tag or ''} {depth_label or ''} Cell",
                        group_colors=group_colors,
                        edge_colors=edge_colors,
                        ylabel=measurement_col,
                    )

                    fig_animal, ax_animal, _ = bar_plot(
                        categories=list(data_dict_animal.keys()),
                        values=[np.mean(data_dict_animal[k]) for k in data_dict_animal],
                        data_points=data_dict_animal,
                        errors={k: np.std(data_dict_animal[k], ddof=1) / np.sqrt(len(data_dict_animal[k]))
                                for k in data_dict_animal},
                        title=f"{lesion_tag or ''} {depth_label or ''} Animal",
                        group_colors=group_colors,
                        edge_colors=edge_colors,
                        ylabel=measurement_col,
                    )

                    fig_graph.savefig(os.path.join(base_plot_dir, f"{safe_name}_graph.png"),
                                      dpi=600, bbox_inches='tight', transparent=True)
                    fig_animal.savefig(os.path.join(base_plot_dir, f"{safe_name}_N.png"),
                                       dpi=600, bbox_inches='tight', transparent=True)
                    plt.close(fig_graph)
                    plt.close(fig_animal)

                    summary_rows = []
                    for grp, vals in data_dict.items():
                        n    = len(vals)
                        mean = np.mean(vals)
                        sem  = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0
                        summary_rows.append({"Group": grp, "N": n, "Mean": mean, "SEM": sem})
                    summary_df_table = pd.DataFrame(summary_rows)
                    num_cols = summary_df_table.select_dtypes(include=[np.number]).columns
                    summary_df_table[num_cols] = summary_df_table[num_cols].apply(
                        lambda x: x.map(lambda y: float(f"{y:.3g}")))

                    _save_table_image(summary_df_table,
                                      os.path.join(base_plot_dir, f"{safe_name}_table.png"),
                                      title=f"{'_'.join(comp)} Summary")
                    summary_df_table.to_excel(writer,
                                              sheet_name=shorten_name(f"{safe_name}_Summary"),
                                              index=False)

                    if not res_df.empty:
                        stats_df_table = res_df.copy()
                        num_cols = stats_df_table.select_dtypes(include=[np.number]).columns
                        stats_df_table[num_cols] = stats_df_table[num_cols].apply(
                            lambda x: x.map(lambda y: float(f"{y:.3g}")))
                        _save_table_image(stats_df_table,
                                          os.path.join(base_plot_dir, f"{safe_name}_stats_table.png"),
                                          title=f"{'_'.join(comp)} Stats")
                        stats_df_table.to_excel(writer,
                                                sheet_name=shorten_name(f"{safe_name}_Stats"),
                                                index=False)

                    if data_dict_animal:
                        summary_rows_animal = []
                        for grp, vals in data_dict_animal.items():
                            n    = len(vals)
                            mean = np.mean(vals)
                            sem  = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0
                            summary_rows_animal.append({"Group": grp, "N": n,
                                                        "Mean": mean, "SEM": sem})
                        summary_df_animal = pd.DataFrame(summary_rows_animal)
                        num_cols = summary_df_animal.select_dtypes(include=[np.number]).columns
                        summary_df_animal[num_cols] = summary_df_animal[num_cols].apply(
                            lambda x: x.map(lambda y: float(f"{y:.3g}")))

                        _save_table_image(summary_df_animal,
                                          os.path.join(base_plot_dir, f"{safe_name}_N_table.png"),
                                          title=f"{'_'.join(comp)} N Summary")
                        summary_df_animal.to_excel(writer,
                                                   sheet_name=shorten_name(f"{safe_name}_N_Summary"),
                                                   index=False)

                        if not res_df_animal.empty:
                            stats_df_animal = res_df_animal.copy()
                            num_cols = stats_df_animal.select_dtypes(include=[np.number]).columns
                            stats_df_animal[num_cols] = stats_df_animal[num_cols].apply(
                                lambda x: x.map(lambda y: float(f"{y:.3g}")))
                            _save_table_image(stats_df_animal,
                                              os.path.join(base_plot_dir,
                                                           f"{safe_name}_N_stats_table.png"),
                                              title=f"{'_'.join(comp)} N Stats")
                            stats_df_animal.to_excel(writer,
                                                     sheet_name=shorten_name(f"{safe_name}_N_Stats"),
                                                     index=False)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION B — lesion-vs-lesion bar graphs per training group
        # ══════════════════════════════════════════════════════════════════════
        if use_lesion:
            _run_lesion_bargraphs(
                df=df,
                measurement_col=measurement_col,
                training_col=training_col,
                lesion_col=lesion_col,
                depth_col=depth_col,
                depth_threshold=depth_threshold,
                animal_id_col=animal_id_col,
                lesion_colors=lesion_colors,
                group_colors=group_colors,
                base_plot_dir=base_plot_dir,
                writer=writer,
                comparisons=comparisons,
                dpi=600,
            )

    return all_results, excel_path