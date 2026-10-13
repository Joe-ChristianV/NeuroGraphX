# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:22:45 2026

@author: JoeCh
"""

# FigureAssemblyModule.py
# Assembles publication figures from plots saved in the Plots/ folder.
#
# Layout derived from Example_Layout.pptx measurements:
#
#   Row 0 (scatter): plots butted together, zero wspace, offset to align with
#                    data cols 2–5 (trace onward). Schematic col has no scatter.
#   Row 1 (L2):  Schematic | Trace | CDH | Cell Bar | Animal Bar
#   Row 2 (L3):  same
#
# Width ratios (from PPTX): schematic=0.60, trace=1.19, cdh=1.0, bar=1.0, bar=1.0
# All adjacent panels butt together (wspace≈0, zero gap between images).
#
# Schematics are loaded from <plots_folder>/../Schematics/
# Labels only shown when image slot is empty (NOT FOUND).

import os
import glob
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT CONSTANTS  (from PPTX measurements)
# ──────────────────────────────────────────────────────────────────────────────

# Width ratios for the 5 data columns
# schematic : trace : cdh : cell_bar : animal_bar
_COL_WIDTHS = [0.60, 1.19, 1.0, 1.0, 1.0]

# Row height ratios: scatter row is slightly shorter than the data rows
# From PPTX: scatter h=1.97", data rows h≈2.02" → ratio ≈ 0.975
_ROW_HEIGHTS = [0.975, 1.0, 1.0]

# In the PPTX the scatter plots start at the left edge of the TRACE column,
# not the schematic column.  This means scatter col 1 aligns with data col 2.
# We replicate this by giving the scatter row a dummy blank slot for col 0
# (schematic position) then packing the scatter plots from col 1 onward.
_SCATTER_START_COL = 1   # scatter plots begin at this data column index


# ──────────────────────────────────────────────────────────────────────────────
# FILENAME HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _shorten_name(name, max_len=31):
    if name is None:
        return ""
    name = str(name)
    if len(name) <= max_len:
        return name
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"{name[:max_len - 7]}_{hash_suffix}"


def _bar_base_name(lesion_tag, depth_label, comp):
    parts = [p for p in [lesion_tag, depth_label] + list(comp) if p]
    raw   = "_".join(parts).replace(" ", "_")
    return _shorten_name(raw)


def _find_latest(folder, pattern):
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMATICS FOLDER  (sibling of Plots/)
# ──────────────────────────────────────────────────────────────────────────────

def _schematics_folder(plots_folder):
    parent     = os.path.dirname(os.path.abspath(plots_folder))
    schematics = os.path.join(parent, "Schematics")
    if os.path.isdir(schematics):
        return schematics
    print(f"  ⚠️  Schematics/ not found at {schematics} — using Plots/")
    return plots_folder


# ──────────────────────────────────────────────────────────────────────────────
# PER-PLOT FINDERS
# ──────────────────────────────────────────────────────────────────────────────

def _find_schematic(schematics_dir, layer):
    exact = os.path.join(schematics_dir, f"{layer} Schematic.png")
    if os.path.exists(exact):
        return exact
    path = _find_latest(schematics_dir, f"*{layer}*.png")
    if path:
        print(f"    schematic glob [{layer}] → {path}")
        return path
    print(f"    schematic NOT FOUND for {layer}")
    return None


def _find_trace(plots_folder, comp, layer):
    comp_str     = "_vs_".join(comp)
    comp_str_rev = "_vs_".join(reversed(comp))
    candidates = []
    for ext in (".png", ".svg"):
        candidates.append(f"AvgTrace_{comp_str}_{layer}{ext}")
        candidates.append(f"AvgTrace_{comp_str_rev}_{layer}{ext}")
    for fname in candidates:
        path = os.path.join(plots_folder, fname)
        print(f"    trace try {fname} → {'FOUND' if os.path.exists(path) else 'miss'}")
        if os.path.exists(path):
            return path
    for g1, g2 in [(comp[0], comp[1]), (comp[1], comp[0])]:
        for ext in ("png", "svg"):
            pattern = f"*{g1}*{g2}*{layer}*.{ext}"
            path = _find_latest(plots_folder, pattern)
            print(f"    trace glob [{pattern}] → {path}")
            if path:
                return path
    print(f"    trace NOT FOUND for {comp} | {layer}")
    return None


def _find_cdh(plots_folder, comp, lesion_tag, depth_label):
    comp_label = " vs ".join(comp)
    candidates = []
    if lesion_tag:
        candidates += [
            f"{comp_label}_{lesion_tag}_{depth_label}.png",
            f"{comp_label}_{depth_label}_{lesion_tag}.png",
        ]
    candidates.append(f"{comp_label}_{depth_label}.png")
    for fname in candidates:
        path = os.path.join(plots_folder, fname)
        print(f"    CDH try: {fname} → {'FOUND' if os.path.exists(path) else 'miss'}")
        if os.path.exists(path):
            return path
    path = _find_latest(plots_folder, f"*{depth_label}*.png")
    if path and "graph" not in path and "_N." not in path and "Schematic" not in path:
        print(f"    CDH glob fallback → {path}")
        return path
    print(f"    CDH NOT FOUND for {comp_label} | {lesion_tag} | {depth_label}")
    return None


def _find_lesion_cdh(plots_folder, train_group, depth_label):
    fname = f"LesionComp_{train_group}_{depth_label}.png"
    path  = os.path.join(plots_folder, fname)
    return path if os.path.exists(path) else None


def _find_scatter(plots_folder, training_group, lesion_tag):
    safe_group = str(training_group).replace(" ", "_").replace("/", "_")
    if lesion_tag:
        safe_lesion = str(lesion_tag).replace(" ", "_").replace("/", "_")
        candidates = [
            f"rdt_{safe_group}_-_{safe_lesion}_scatter_*.png",
            f"rdt_{safe_lesion}_-_{safe_group}_scatter_*.png",
            f"rdt_*{safe_group}*{safe_lesion}*scatter*.png",
            f"rdt_*{safe_lesion}*{safe_group}*scatter*.png",
        ]
    else:
        candidates = [
            f"rdt_{safe_group}_scatter_*.png",
            f"rdt_*{safe_group}*scatter*.png",
        ]
    for pattern in candidates:
        path = _find_latest(plots_folder, pattern)
        print(f"    scatter [{training_group}|{lesion_tag}]: {pattern} → {path}")
        if path:
            return path
    return None


def _find_cell_bar(plots_folder, lesion_tag, depth_label, comp):
    base = _bar_base_name(lesion_tag, depth_label, comp)
    path = os.path.join(plots_folder, f"{base}_graph.png")
    if os.path.exists(path):
        return path
    return _find_latest(plots_folder, f"*{depth_label}*_graph.png")


def _find_animal_bar(plots_folder, lesion_tag, depth_label, comp):
    base = _bar_base_name(lesion_tag, depth_label, comp)
    path = os.path.join(plots_folder, f"{base}_N.png")
    if os.path.exists(path):
        return path
    return _find_latest(plots_folder, f"*{depth_label}*_N.png")


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _load_image(path):
    if path is None or not os.path.exists(path):
        return None
    if path.lower().endswith(".svg"):
        try:
            import cairosvg, io
            from PIL import Image
            return np.array(Image.open(io.BytesIO(
                cairosvg.svg2png(url=path, dpi=150))).convert("RGBA"))
        except Exception:
            pass
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            import io
            from PIL import Image
            return np.array(Image.open(io.BytesIO(
                renderPM.drawToString(svg2rlg(path), fmt="PNG"))).convert("RGBA"))
        except Exception:
            pass
        return "SVG"
    try:
        return mpimg.imread(path)
    except Exception as e:
        print(f"⚠️  Could not load {path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def _place_image(ax, img, label=None):
    """
    Render image into axes with zero padding.
    Label shown ONLY when no image is available.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if isinstance(img, np.ndarray):
        ax.imshow(img, aspect="auto")
        # No title when a real image is present

    elif isinstance(img, str) and img == "SVG":
        ax.set_facecolor("#F0F0F0")
        ax.text(0.5, 0.5, label or "SVG\n(install cairosvg)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6, color="#888888", style="italic")

    else:
        # None or load error — show placeholder with label
        ax.set_facecolor("#F0F0F0")
        ax.text(0.5, 0.5, label or "NOT FOUND",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="#AAAAAA", style="italic")


def _blank(ax):
    ax.set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FIGURE BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _build_group_figure(comp, lesion_tag, plots_folder, schematics_dir, dpi):
    """
    Replicates the PPTX layout exactly:

    Row 0 (scatter):
        [blank schematic slot] | scatter_G1 | scatter_G2 | ... | scatter_Gn
        Scatters are zero-gap, starting at _SCATTER_START_COL.
        If n_groups > n_data_cols - 1, scatters fill all available slots.

    Row 1 (L2):
        Schematic | Trace | CDH | Cell Bar | Animal Bar

    Row 2 (L3):
        same

    Width ratios: [0.60, 1.19, 1.0, 1.0, 1.0]
    All panels butt together (wspace=0).
    """
    title    = " vs ".join(comp)
    if lesion_tag:
        title += f"  —  {lesion_tag}"

    n_groups    = len(comp)
    n_data_cols = len(_COL_WIDTHS)   # 5

    # ── Collect images ────────────────────────────────────────────────────────
    scatter_imgs = [_load_image(_find_scatter(plots_folder, g, lesion_tag))
                    for g in comp]
    layer_imgs = {}
    for layer in ("L2", "L3"):
        layer_imgs[layer] = [
            _load_image(_find_schematic(schematics_dir, layer)),
            _load_image(_find_trace(plots_folder, comp, layer)),
            _load_image(_find_cdh(plots_folder, comp, lesion_tag, layer)),
            _load_image(_find_cell_bar(plots_folder, lesion_tag, layer, comp)),
            _load_image(_find_animal_bar(plots_folder, lesion_tag, layer, comp)),
        ]
        labels = [f"{layer} Schematic", f"{layer} Avg Trace",
                  f"{layer} CDH", f"{layer} Cells", f"{layer} Animals"]
        layer_imgs[layer] = list(zip(layer_imgs[layer], labels))

    # Debug manifest
    print(f"\n  [Figure manifest] {title}")
    for i, g in enumerate(comp):
        status = "loaded" if isinstance(scatter_imgs[i], np.ndarray) else "NOT FOUND"
        print(f"    scatter[{g}]: {status}")
    for layer in ("L2", "L3"):
        for img, lbl in layer_imgs[layer]:
            status = "loaded" if isinstance(img, np.ndarray) else "NOT FOUND"
            print(f"    {lbl}: {status}")

    # ── Figure sizing ─────────────────────────────────────────────────────────
    # Use inches matching the PPTX proportions.
    # Data cols total width drives figure width; scatter row is slightly shorter.
    col_unit = 2.2          # inches per "1" column unit
    total_w  = sum(_COL_WIDTHS) * col_unit            # ~10.6 inches

    # Row heights proportional to PPTX ratios
    row_h_unit = 2.1        # inches per "1" row unit
    total_h    = sum(_ROW_HEIGHTS) * row_h_unit + 0.4  # title headroom

    fig = plt.figure(figsize=(total_w, total_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Outer GridSpec: 3 rows
    outer = GridSpec(
        3, 1,
        figure=fig,
        height_ratios=_ROW_HEIGHTS,
        hspace=0.14,
        left=0.01, right=0.99,
        top=0.94,  bottom=0.01,
    )

    # ── Row 0: scatter plots ──────────────────────────────────────────────────
    # Scatter plots occupy columns _SCATTER_START_COL .. n_data_cols-1.
    # We create a subgridspec with the same width ratios as the data rows,
    # then hide the schematic slot and fill scatter plots into the rest.
    scatter_gs = GridSpecFromSubplotSpec(
        1, n_data_cols,
        subplot_spec=outer[0],
        width_ratios=_COL_WIDTHS,
        wspace=0.0,
    )

    # Hide the schematic placeholder slot
    ax_blank = fig.add_subplot(scatter_gs[0, 0])
    _blank(ax_blank)

    # Available scatter columns: cols 1..(n_data_cols-1)
    available_scatter_cols = n_data_cols - _SCATTER_START_COL  # 4
    # If more groups than slots, wrap extras into first slots
    for col_i in range(available_scatter_cols):
        ax = fig.add_subplot(scatter_gs[0, _SCATTER_START_COL + col_i])
        if col_i < n_groups:
            _place_image(ax, scatter_imgs[col_i], label=f"RDT: {comp[col_i]}")
        else:
            _blank(ax)

    # ── Rows 1 & 2: data rows ─────────────────────────────────────────────────
    for row_i, layer in enumerate(("L2", "L3"), start=1):
        data_gs = GridSpecFromSubplotSpec(
            1, n_data_cols,
            subplot_spec=outer[row_i],
            width_ratios=_COL_WIDTHS,
            wspace=0.0,
        )
        for col_i, (img, lbl) in enumerate(layer_imgs[layer]):
            ax = fig.add_subplot(data_gs[0, col_i])
            _place_image(ax, img, label=lbl)

    fig.suptitle(title, fontsize=10, fontfamily="Arial",
                 fontstyle="italic", y=0.98)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# LESION SUMMARY FIGURE
# ──────────────────────────────────────────────────────────────────────────────

def _build_lesion_summary_figure(comp, lesion_groups, plots_folder,
                                  schematics_dir, dpi):
    title  = " vs ".join(comp) + "  —  Lesion Comparison Summary"
    layers = ("L2", "L3")
    n_cols = len(comp) * len(layers)

    col_unit = 2.5
    fig = plt.figure(figsize=(col_unit * n_cols, col_unit * 2.1), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, n_cols, figure=fig,
                  hspace=0.10, wspace=0.0,
                  left=0.01, right=0.99,
                  top=0.92,  bottom=0.01)

    col = 0
    for g in comp:
        for layer in layers:
            cdh_img  = _load_image(_find_lesion_cdh(plots_folder, g, layer))
            bar_path = (_find_latest(plots_folder, f"*_{layer}_{g}_graph.png") or
                        _find_latest(plots_folder, f"*{layer}*{g}*graph*.png"))
            bar_img  = _load_image(bar_path)
            _place_image(fig.add_subplot(gs[0, col]), cdh_img,
                         label=f"{g} {layer} CDH (lesions)")
            _place_image(fig.add_subplot(gs[1, col]), bar_img,
                         label=f"{g} {layer} Cells (lesions)")
            col += 1

    fig.suptitle(title, fontsize=10, fontfamily="Arial",
                 fontstyle="italic", y=0.97)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_figure_assembly(
    comparisons,
    plots_folder,
    output_folder,
    lesion_groups=None,
    dpi=300,
):
    """
    Build and save publication figures for every comparison.

    Parameters
    ----------
    comparisons   : list of lists e.g. [['ACC','SAT1']]
    plots_folder  : path to Plots/
    output_folder : path to Figures/ (created if missing)
    lesion_groups : list[str] | None
    dpi           : output DPI

    Returns
    -------
    saved_files : list[str]
    warnings    : list[str]
    """
    os.makedirs(output_folder, exist_ok=True)

    schematics_dir = _schematics_folder(plots_folder)
    saved_files    = []
    all_warnings   = []
    use_lesion     = bool(lesion_groups)

    def _save(fig, base):
        for ext in (".png", ".svg"):
            try:
                fig.savefig(base + ext, dpi=dpi, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                saved_files.append(base + ext)
                print(f"   ✅ {base + ext}")
            except Exception as e:
                msg = f"Could not save {base + ext}: {e}"
                all_warnings.append(msg)
                print(f"⚠️  {msg}")
        plt.close(fig)

    for comp in comparisons:
        comp_str = "_vs_".join(comp)

        if not use_lesion:
            print(f"\n=== Figure: {comp_str} (no lesion) ===")
            try:
                fig = _build_group_figure(comp, None, plots_folder,
                                          schematics_dir, dpi)
                _save(fig, os.path.join(output_folder, f"Figure_{comp_str}"))
            except Exception as e:
                msg = f"Error building figure {comp_str}: {e}"
                all_warnings.append(msg)
                print(f"⚠️  {msg}")

        else:
            for lesion in lesion_groups:
                safe = str(lesion).replace(" ", "_").replace(",", "_")
                print(f"\n=== Figure: {comp_str} | {lesion} ===")
                try:
                    fig = _build_group_figure(comp, str(lesion), plots_folder,
                                              schematics_dir, dpi)
                    _save(fig, os.path.join(output_folder,
                                            f"Figure_{comp_str}_{safe}"))
                except Exception as e:
                    msg = f"Error building figure {comp_str}/{lesion}: {e}"
                    all_warnings.append(msg)
                    print(f"⚠️  {msg}")

            print(f"\n=== Lesion Summary: {comp_str} ===")
            try:
                fig = _build_lesion_summary_figure(comp, lesion_groups,
                                                   plots_folder,
                                                   schematics_dir, dpi)
                _save(fig, os.path.join(output_folder,
                                        f"Figure_{comp_str}_LesionSummary"))
            except Exception as e:
                msg = f"Error building lesion summary {comp_str}: {e}"
                all_warnings.append(msg)
                print(f"⚠️  {msg}")

    return saved_files, all_warnings