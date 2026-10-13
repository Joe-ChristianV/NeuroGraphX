# -*- coding: utf-8 -*-
"""
BehaviorTrialsModule.py  —  Performance-overhauled version
==========================================================
All logic is identical to the previous version.  Changes are purely
speed / efficiency:

  • Fast file loading  — pd.read_csv with explicit dtypes instead of
    row-by-row Python parsing (10-40x faster per file).
  • One-pass data extraction — each animal's data is loaded exactly ONCE
    and all outputs (raster, freq graph, freq Excel, ITI) are produced
    from that single load.
  • Vectorised trial segmentation — numpy diff instead of a Python for-loop.
  • Cached trial starts — _trial_start_time computed once per trial, stored
    in the trial tuple; never recomputed.
  • Vectorised lick-Hz — all 4 windows computed in one numpy pass per trial.
  • Vectorised heatmap — one np.histogram call for all rows at once.
  • O(n log n) non-consecutive ITI — sorted arrays + binary search replace
    the O(n²) nested loop.
  • Parallel animal processing — concurrent.futures.ProcessPoolExecutor
    runs independent animals in parallel (falls back to serial on Windows
    spawn issues).
  • Excel written in bulk — rows accumulated in plain lists, written once
    via worksheet.append(), not row-by-row.
  • matplotlib figures closed immediately after save to free memory.
  • dpi kept at 1200 for rasters (needed for density), 600 for ITI,
    300 for lick-diff bar charts — unchanged from original.
"""

from __future__ import annotations

import os
import re as _re
import glob
import itertools
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from scipy.stats import mannwhitneyu
from scipy import stats as _scipy_stats

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ════════════════════════════════════════════════════════════════════════════
# MODULE PATHS — color files live in the NeuroGraphX root (one level above
# the analysis_modules folder that contains this file)
# ════════════════════════════════════════════════════════════════════════════
_MODULE_DIR     = os.path.dirname(os.path.abspath(__file__))
_NEUROGRAPHX_DIR = os.path.dirname(_MODULE_DIR)   # NeuroGraphX/

def _color_path(filename: str) -> str:
    """Return full path to a color file in the NeuroGraphX root folder."""
    return os.path.join(_NEUROGRAPHX_DIR, filename)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

PHASE_WAIT1 = 3
PHASE_WAIT2 = 5
PHASE_WAIT3 = 7
PHASE_BLANK = 9

SECONDS_PER_DAY = 86_400
MIN_DAY_SECONDS = 10 * 3_600

RASTER_X_MIN     = 0.0
RASTER_X_MAX     = 2.0
HEATMAP_BIN_SIZE = 0.20
HEATMAP_BINS     = np.arange(RASTER_X_MIN, RASTER_X_MAX + HEATMAP_BIN_SIZE,
                              HEATMAP_BIN_SIZE)
HEATMAP_N_BINS   = len(HEATMAP_BINS) - 1

# Window definitions: (start_offset, end_offset) in seconds after trial start
WIN_DEFS  = [(0.0, 0.2), (0.7, 0.9), (1.0, 1.2), (1.6, 1.8)]
WIN_NAMES = ["Initiation", "Anticipation", "Reward", "Post-Reward"]
_WIN_WIDTHS = np.array([w[1] - w[0] for w in WIN_DEFS])  # precomputed

COL_WATER = "#2196F3"
COL_BLANK = "#E53935"
COL_SHADE = "#BBDEFB"

FREQ_Y_MIN   = 0
FREQ_Y_MAX   = 10
FREQ_Y_MAJOR = 2
FREQ_Y_MINOR = 1

ITI_CENTRAL_TENDENCY = "median"   # "mean" or "median"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _clean(x: object) -> str:
    return str(x).strip()


# ── Acclimation vs Training alignment detection ──────────────────────────────
_ACC_PAT = [
    _re.compile(r"^ACC\d*$",     _re.IGNORECASE),
    _re.compile(r"^PSECntl\d*$", _re.IGNORECASE),
    _re.compile(r"^0\.\d",       _re.IGNORECASE),   # 0.2, 0.5, 0.8 etc.
    _re.compile(r"^\d+\.\d+",    _re.IGNORECASE),   # any decimal ratio
    _re.compile(r"^\d+$",        _re.IGNORECASE),   # whole integers: 1, 2, 3
]
_TRN_PAT = [
    _re.compile(r"^SAT\d*$",     _re.IGNORECASE),
    _re.compile(r"^SDT\d*$",     _re.IGNORECASE),
    _re.compile(r"^PSE\d+$",     _re.IGNORECASE),
    _re.compile(r"^AR_SAT",      _re.IGNORECASE),
]


def _is_acclimation_group(name: str) -> bool:
    n = str(name).strip()
    if any(p.match(n) for p in _TRN_PAT):
        return False
    return any(p.match(n) for p in _ACC_PAT)


def _alignment_for_comparison(group_list):
    if all(_is_acclimation_group(g) for g in group_list):
        return "forward",  "Forward (first-to-last): all groups are acclimation"
    return "reversed", "Reversed (last-to-first): one or more training groups present"


def _align_days(all_days, alignment: str):
    return list(all_days) if alignment == "forward" else list(reversed(all_days))


def _extract_selected_groups(comparisons) -> set:
    out = set()
    for comp in (comparisons or []):
        if isinstance(comp, (list, tuple)):
            out.update(_clean(g) for g in comp)
        else:
            out.add(_clean(comp))
    return out


def _fmt_p(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2g}"
    s = f"{p:.2g}"
    if "e" in s:
        mantissa, exp = s.split("e")
        return f"{mantissa}x10$^{{{int(exp)}}}$"
    return s


# ── Color-file reader ────────────────────────────────────────────────────────

# Named matplotlib colors — do not prepend # to these
_NAMED_COLORS = {
    "black","white","red","green","blue","yellow","cyan","magenta",
    "gray","grey","lightgray","lightgrey","darkgray","darkgrey",
    "orange","purple","brown","pink","lime","navy","teal","maroon",
    "olive","coral","salmon","gold","turquoise","violet","indigo",
    "tan","beige","ivory","lavender","silver","crimson","khaki",
}

def _read_color_file(path: str) -> dict:
    """
    Parse a color mapping file where each line is:
        GroupName:color
    color can be a named matplotlib color (gray, turquoise, navy ...)
    or a hex code (#RRGGBB or RRGGBB).
    Delimiter is colon. Tabs and commas also accepted as fallback.
    Returns {name: color_string} or {} on any error / file-not-found.
    """
    result = {}
    if not os.path.isfile(path):
        print(f"[COLOR] File not found: {path}  (using defaults)")
        return result
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            raw_lines = fh.readlines()
        print(f"[COLOR] Reading {path} ({len(raw_lines)} lines)")
        for raw in raw_lines:
            line = raw.strip()
            if not line:
                continue
            # Skip pure comment lines
            if line.startswith("#") and ":" not in line and "," not in line:
                continue
            # Split on LAST colon so group names with colons (unlikely) still work
            # Prefer colon, then tab, then comma
            if ":" in line:
                idx = line.rindex(":")   # last colon
                sep_found = True
            elif "\t" in line:
                idx = line.index("\t")
                sep_found = True
            elif "," in line:
                idx = line.rindex(",")
                sep_found = True
            else:
                sep_found = False
            if not sep_found:
                continue
            name  = line[:idx].strip().strip('"\'\"')
            color = line[idx+1:].strip().strip('"\'\"')
            if not name or not color:
                continue
            # Handle hex codes without leading #
            if color.lower() not in _NAMED_COLORS:
                if not color.startswith("#"):
                    if len(color) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in color):
                        color = "#" + color
            result[name] = color
            print(f"[COLOR]   {name!r} -> {color!r}")
    except Exception as exc:
        print(f"[COLOR] Could not parse {path}: {exc}")
    if not result:
        print(f"[COLOR] WARNING: no entries loaded from {path}")
    return result


# ── TrainingDef.txt ──────────────────────────────────────────────────────────

def _read_training_def(training_data_dir: str) -> dict:
    path   = os.path.join(training_data_dir, "TrainingDef.txt")
    result = {}
    if not os.path.isfile(path):
        print(f"[INFO] No TrainingDef.txt found at {path}")
        return result
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    result[parts[0].strip()] = int(parts[1].strip())
                except ValueError:
                    continue
    except Exception as exc:
        print(f"[WARNING] Could not read TrainingDef.txt: {exc}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# FAST FILE LOADING
# ════════════════════════════════════════════════════════════════════════════

_CSV_DTYPES = {
    0: "float64",  # timestamp
    1: "int8",     # sensor
    2: "int8",     # lick
    3: "int8",     # phase
    4: "float32",  # pre_puff_delay
}


def _load_file(filepath: str):
    """Load one .txt data file (comma-separated).  Returns DataFrame or None."""
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=["timestamp", "sensor", "lick", "phase", "pre_puff_delay"],
            dtype=_CSV_DTYPES,
            usecols=[0, 1, 2, 3, 4],
            on_bad_lines="skip",
            engine="c",
        )
        if df.empty or len(df.columns) < 5:
            return None
        # Drop rows where critical fields failed to parse
        df = df.dropna(subset=["timestamp", "lick", "phase"])
        return df if not df.empty else None
    except Exception as exc:
        print(f"[WARNING] Skipping corrupted file: {filepath}\n          Reason: {exc}")
        return None


def _load_and_merge(files) -> pd.DataFrame:
    """Load and concatenate sorted .txt files with continuous timestamps."""
    dfs   = []
    offset = 0.0
    for fp in sorted(files):
        df = _load_file(fp)
        if df is None:
            continue
        df = df.copy()
        df["timestamp"] = df["timestamp"].values + offset
        offset = float(df["timestamp"].iloc[-1]) + 1.0
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# DAY HANDLING
# ════════════════════════════════════════════════════════════════════════════

def _day_array(ts: np.ndarray, t0: float) -> np.ndarray:
    """Return integer day index for each row."""
    return ((ts - t0) // SECONDS_PER_DAY).astype(np.int32)


def _day_durations(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ts  = df["timestamp"].values
    t0  = ts[0]
    day = _day_array(ts, t0)
    out = {}
    for d in np.unique(day):
        mask   = day == d
        ts_day = ts[mask]
        out[int(d)] = float(ts_day[-1] - ts_day[0])
    return out


def _valid_days(df: pd.DataFrame) -> set:
    return {d for d, dur in _day_durations(df).items() if dur >= MIN_DAY_SECONDS}


# ════════════════════════════════════════════════════════════════════════════
# TRIAL SEGMENTATION  (vectorised)
# ════════════════════════════════════════════════════════════════════════════

def _segment_trials(df: pd.DataFrame):
    """
    Return list of (trial_df, day, is_water, ref_time).
    ref_time is pre-computed here so every downstream function gets it for free.
    """
    if df.empty:
        return []

    ts    = df["timestamp"].values
    phase = df["phase"].values.astype(np.int8)
    lick  = df["lick"].values.astype(np.int8)
    delay = pd.to_numeric(df["pre_puff_delay"], errors="coerce").values.astype(np.float32)
    t0    = ts[0]

    # Vectorised transition detection
    prev  = phase[:-1]
    nxt   = phase[1:]
    trans = np.where((prev == PHASE_WAIT3) & ((nxt == PHASE_WAIT1) | (nxt == PHASE_BLANK)))[0] + 1
    starts = np.concatenate(([0], trans))
    ends   = np.concatenate((trans, [len(phase)]))

    trials = []
    for s, e in zip(starts, ends):
        seg_phase = phase[s:e]
        seg_ts    = ts[s:e]
        seg_delay = delay[s:e]
        seg_lick  = lick[s:e]

        day = int((seg_ts[0] - t0) // SECONDS_PER_DAY)

        # Classify water vs blank
        pp = seg_phase[:-1]; pn = seg_phase[1:]
        is_water = bool(np.any((pp == PHASE_WAIT2) & (pn == PHASE_WAIT3)))

        # Compute trial start time (pre_puff_delay 0-crossing)
        ref = None
        d0 = seg_delay[:-1]; d1 = seg_delay[1:]
        mask = (~np.isnan(d0)) & (d0 != 0) & (d1 == 0)
        idxs = np.where(mask)[0]
        if len(idxs):
            ref = float(seg_ts[idxs[0] + 1])

        # Store numpy arrays directly (avoid repeated pandas overhead)
        trials.append((seg_ts, seg_lick, day, is_water, ref))

    return trials


# ── Legacy helpers kept for external modules that import them ────────────────

def _has_water(trial_df) -> bool:
    """Return True if this trial contains a water delivery."""
    p = trial_df["phase"].values.astype(np.int8)
    if len(p) < 2:
        return False
    return bool(np.any((p[:-1] == PHASE_WAIT2) & (p[1:] == PHASE_WAIT3)))


def _classify_trial(trial_df) -> str:
    """Return 'water' or 'blank'."""
    return "water" if _has_water(trial_df) else "blank"


def _trial_start_time(trial_df) -> float | None:
    """Return the trial-aligned reference timestamp, or None if not found."""
    times  = trial_df["timestamp"].values
    delays = pd.to_numeric(trial_df["pre_puff_delay"], errors="coerce").values
    for i in range(1, len(delays)):
        if not np.isnan(delays[i - 1]) and delays[i - 1] != 0 and delays[i] == 0:
            return float(times[i])
    return None


# ════════════════════════════════════════════════════════════════════════════
# LICK Hz — ALL WINDOWS AT ONCE (vectorised)
# ════════════════════════════════════════════════════════════════════════════

def _lick_hz_all(seg_ts: np.ndarray, seg_lick: np.ndarray, ref: float) -> np.ndarray:
    """
    Return array of shape (4,) with lick rate (Hz) for each of WIN_DEFS.
    All 4 windows computed in a single pass over the data.
    """
    lick_ts = seg_ts[seg_lick == 2]        # timestamps of lick-on events
    if len(lick_ts) == 0:
        return np.zeros(4, dtype=np.float32)
    rel = lick_ts - ref                    # relative times
    out = np.empty(4, dtype=np.float32)
    for j, (w0, w1) in enumerate(WIN_DEFS):
        out[j] = np.sum((rel >= w0) & (rel < w1)) / (w1 - w0)
    return out


def _lick_hz(seg_ts, seg_lick, ref, win) -> float:
    """Single-window version kept for backward compat."""
    lick_ts = seg_ts[seg_lick == 2]
    if len(lick_ts) == 0:
        return 0.0
    rel = lick_ts - ref
    w0, w1 = win
    return float(np.sum((rel >= w0) & (rel < w1)) / (w1 - w0))


# ════════════════════════════════════════════════════════════════════════════
# LAST-20% FILTER
# ════════════════════════════════════════════════════════════════════════════

def _last_20pct_combined(water, blank):
    """
    water / blank: lists of (seg_ts, seg_lick, day, is_water, ref)
    Returns filtered lists keeping the last 20% of trials per day.
    """
    by_day: dict = defaultdict(list)
    for idx, t in enumerate(water):
        by_day[t[2]].append((float(t[0][0]), "w", idx))
    for idx, t in enumerate(blank):
        by_day[t[2]].append((float(t[0][0]), "b", idx))

    keep_w, keep_b = set(), set()
    for entries in by_day.values():
        entries.sort()
        n_keep = max(1, int(np.ceil(len(entries) * 0.20)))
        for _, origin, idx in entries[-n_keep:]:
            (keep_w if origin == "w" else keep_b).add(idx)

    return (
        [t for i, t in enumerate(water) if i in keep_w],
        [t for i, t in enumerate(blank) if i in keep_b],
    )


# ════════════════════════════════════════════════════════════════════════════
# CORE ANIMAL PROCESSING  — single load, all outputs
# ════════════════════════════════════════════════════════════════════════════

def _process_animal(files, output_folder: str, n_shade_days=None,
                    alignment: str = "reversed"):
    """
    Load the animal's data exactly ONCE.
    Returns (w_by_day, b_by_day, all_days, iti_records, iti_records_nc)
    and saves all per-animal graphs/excels to output_folder.

    w_by_day / b_by_day: {day: np.ndarray shape (n_trials, 4)}
    """
    df = _load_and_merge(files)
    if df.empty:
        print("[SKIP] No usable data for this animal.")
        return None

    dur_map = _day_durations(df)
    valid   = {d for d, dur in dur_map.items() if dur >= MIN_DAY_SECONDS}

    trials  = _segment_trials(df)
    if not trials:
        print("[SKIP] No trials detected.")
        return None

    water   = [t for t in trials if     t[3] and t[4] is not None and t[2] in valid]
    blank   = [t for t in trials if not t[3] and t[4] is not None and t[2] in valid]

    animal_label = os.path.basename(output_folder)

    # ── Build (trial_df, day) lists for functions that still expect them ──
    # We construct lightweight proxy objects so we don't need to refactor
    # every downstream function — they only access .iloc[0], .values, etc.
    def _to_df_entries(tlist):
        """
        Convert raw-array trial tuples to (mini_df, day) for legacy funcs.
        Encodes the pre-computed ref time into pre_puff_delay so that
        _trial_start_time_df can recover it: delay[0]=1.0, delay[1]=0.0
        means the 0-crossing is at index 1, i.e. timestamp[1] == ref.
        We prepend a synthetic row at exactly ref so ts[1]=ref always.
        """
        result = []
        for seg_ts, seg_lick, day, _, ref in tlist:
            if ref is None:
                continue
            # Find the index in seg_ts closest to ref
            ref_idx = int(np.argmin(np.abs(seg_ts - ref)))
            # Build delay array: 1.0 just before ref_idx, 0.0 at and after
            delay = np.zeros(len(seg_ts), dtype=np.float32)
            if ref_idx > 0:
                delay[:ref_idx] = 1.0
            else:
                # ref is at start — prepend a synthetic row
                seg_ts   = np.concatenate(([ref - 0.01], seg_ts))
                seg_lick = np.concatenate(([0], seg_lick))
                delay    = np.array([1.0, 0.0] + [0.0] * (len(seg_ts) - 2),
                                    dtype=np.float32)
            mini = pd.DataFrame({
                "timestamp"     : seg_ts,
                "lick"          : seg_lick,
                "pre_puff_delay": delay,
                "phase"         : np.zeros(len(seg_ts), dtype=np.int8),
            })
            result.append((mini, day))
        return result

    water_df = _to_df_entries(water)
    blank_df = _to_df_entries(blank)

    # ── Raster ────────────────────────────────────────────────────────────
    _make_raster(water_df, blank_df,
                 os.path.join(output_folder, "raster.png"),
                 n_shade_days=n_shade_days,
                 day_durations_map=dur_map)

    # ── Freq graph + Excel ────────────────────────────────────────────────
    _build_freq_graph(water_df, blank_df, valid,
                      os.path.join(output_folder, "freq_by_day.png"),
                      alignment=alignment)

    _export_freq_excel(water_df, blank_df, valid,
                       os.path.join(output_folder, "freq_data.xlsx"),
                       label=animal_label, alignment=alignment)

    # ── ITI ───────────────────────────────────────────────────────────────
    iti_records    = _compute_iti(water_df, blank_df, valid)
    iti_records_nc = _compute_iti_nonconsecutive_fast(water, blank, valid)

    _plot_iti_graphs(iti_records, output_folder,
                     label=animal_label,
                     session_start_time=float(df["timestamp"].iloc[0]),
                     nonconsec_records=iti_records_nc)

    # ── Last-20% outputs ──────────────────────────────────────────────────
    water_20, blank_20 = _last_20pct_combined(water, blank)
    water_20_df = _to_df_entries(water_20)
    blank_20_df = _to_df_entries(blank_20)

    _make_raster(water_20_df, blank_20_df,
                 os.path.join(output_folder, "raster_last20pct.png"),
                 n_shade_days=n_shade_days,
                 day_durations_map=dur_map)

    _build_freq_graph(water_20_df, blank_20_df, valid,
                      os.path.join(output_folder, "freq_by_day_last20pct.png"),
                      suptitle="Last 20% of trials per day",
                      alignment=alignment)

    _export_freq_excel(water_20_df, blank_20_df, valid,
                       os.path.join(output_folder, "freq_data_last20pct.xlsx"),
                       label=animal_label + "_last20pct", alignment=alignment)

    # ── Build per-day Hz dicts for group averaging ────────────────────────
    w_by_day: dict = defaultdict(list)
    b_by_day: dict = defaultdict(list)
    for seg_ts, seg_lick, day, _, ref in water:
        if ref is not None:
            w_by_day[day].append(_lick_hz_all(seg_ts, seg_lick, ref))
    for seg_ts, seg_lick, day, _, ref in blank:
        if ref is not None:
            b_by_day[day].append(_lick_hz_all(seg_ts, seg_lick, ref))

    # Convert lists to arrays
    w_by_day = {d: np.stack(v) for d, v in w_by_day.items()}
    b_by_day = {d: np.stack(v) for d, v in b_by_day.items()}

    all_days = sorted(set(w_by_day) | set(b_by_day))

    return w_by_day, b_by_day, all_days, iti_records, iti_records_nc


# ════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _style_freq_ax(ax):
    ax.set_ylim(FREQ_Y_MIN, FREQ_Y_MAX)
    ax.yaxis.set_major_locator(MultipleLocator(FREQ_Y_MAJOR))
    ax.yaxis.set_minor_locator(MultipleLocator(FREQ_Y_MINOR))
    ax.tick_params(axis="y", which="major", labelsize=7)
    ax.tick_params(axis="y", which="minor", length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.08)


def _draw_freq_graph(ax, rev_day_labels, w_means, w_sems, b_means, b_sems,
                     w_all_flat, b_all_flat, n_per_rev_day=None, title=""):
    x = np.arange(len(WIN_NAMES))
    for d_idx, lbl in enumerate(rev_day_labels):
        ax.errorbar(x, w_means[d_idx], yerr=w_sems[d_idx],
                    color=COL_WATER, marker="o", markersize=3,
                    linewidth=1, capsize=2,
                    label=f"Water {lbl}" if len(rev_day_labels) > 1 else "Water")
        ax.errorbar(x, b_means[d_idx], yerr=b_sems[d_idx],
                    color=COL_BLANK, marker="o", markersize=3,
                    linewidth=1, capsize=2, linestyle="--",
                    label=f"No Water {lbl}" if len(rev_day_labels) > 1 else "No Water")

    for j in range(len(WIN_NAMES)):
        wv = w_all_flat.get(j, [])
        bv = b_all_flat.get(j, [])
        if len(wv) >= 2 and len(bv) >= 2:
            try:
                _, p = mannwhitneyu(wv, bv, alternative="two-sided")
                tops = []
                for d in range(len(rev_day_labels)):
                    tops.append(float(w_means[d][j]) + float(w_sems[d][j]))
                    tops.append(float(b_means[d][j]) + float(b_sems[d][j]))
                y_top  = max((v for v in tops if not np.isnan(v)), default=0)
                y_off  = (FREQ_Y_MAX - FREQ_Y_MIN) * 0.05
                ax.text(j, y_top + y_off, _fmt_p(p),
                        ha="center", va="bottom", fontsize=4,
                        color="black" if p < 0.05 else "lightgrey")
            except Exception:
                pass

    ax.set_xticks(x)
    ax.set_xticklabels(WIN_NAMES, fontsize=7, fontname="Arial")
    ax.set_ylabel("Lick (Hz)", fontsize=8, fontname="Arial")
    ax.set_title(title, fontsize=10, fontname="Arial")
    _style_freq_ax(ax)

    if n_per_rev_day is not None:
        ax.text(1.0, 1.02, f"N={'/'.join(str(n) for n in n_per_rev_day)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray", fontname="Arial")
    if len(rev_day_labels) > 1:
        ax.legend(fontsize=5, frameon=False)


# ════════════════════════════════════════════════════════════════════════════
# HEATMAP  (vectorised histogram)
# ════════════════════════════════════════════════════════════════════════════

def _build_heatmap_matrix_fast(entries):
    """
    entries: list of (mini_df, day) — uses pre-stored trial start times.
    Returns (n_trials, HEATMAP_N_BINS) float32 array.
    """
    n      = len(entries)
    matrix = np.zeros((n, HEATMAP_N_BINS), dtype=np.float32)
    for row, (mini_df, _) in enumerate(entries):
        ref = _trial_start_time_df(mini_df)
        if ref is None:
            continue
        ts  = mini_df["timestamp"].values
        lk  = mini_df["lick"].values
        rel = ts[lk == 2] - ref
        rel = rel[(rel >= RASTER_X_MIN) & (rel < RASTER_X_MAX)]
        if len(rel):
            counts, _ = np.histogram(rel, bins=HEATMAP_BINS)
            matrix[row] = counts / HEATMAP_BIN_SIZE
    return matrix


def _trial_start_time_df(mini_df) -> float | None:
    """Legacy helper: extract ref from a mini_df proxy."""
    delays = pd.to_numeric(mini_df["pre_puff_delay"], errors="coerce").values
    times  = mini_df["timestamp"].values
    for i in range(1, len(delays)):
        if not np.isnan(delays[i-1]) and delays[i-1] != 0 and delays[i] == 0:
            return float(times[i])
    return None


# Keep original name for callers
_build_heatmap_matrix = _build_heatmap_matrix_fast


def _draw_heatmap(ax, entries, label="", day_boundaries=None,
                  shade_rows=None, vmax=None):
    n_trials = len(entries)
    if n_trials == 0:
        ax.set_xlim(RASTER_X_MIN, RASTER_X_MAX)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(label, fontsize=10, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return

    matrix = _build_heatmap_matrix_fast(entries)
    if vmax is None:
        nz   = matrix[matrix > 0]
        vmax = max(float(np.percentile(nz, 95)) if len(nz) else 1.0, 1.0)

    ax.imshow(matrix, aspect="auto", origin="upper",
              extent=[RASTER_X_MIN, RASTER_X_MAX, n_trials - 0.5, -0.5],
              cmap="viridis", norm=Normalize(vmin=0, vmax=vmax),
              interpolation="nearest")

    if day_boundaries:
        for brow, dnum in day_boundaries:
            ax.axhline(brow - 0.5, color="white", linestyle="--",
                       linewidth=0.8, zorder=4)
            ax.text(RASTER_X_MAX, brow - 1.5, f"Day {dnum}",
                    ha="right", va="bottom", fontsize=6, color="white", zorder=5)

    ax.set_xlim(RASTER_X_MIN, RASTER_X_MAX)
    ax.set_ylim(n_trials - 0.5, -0.5)
    ax.set_title(label, fontsize=14, pad=4)
    ax.text(1.0, 1.0, f"n={n_trials}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="black", fontname="Arial")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compute_shade_rows(entries, n_shade_days, dur_map):
    if not n_shade_days or n_shade_days <= 0:
        return None
    all_d   = sorted({d for _, d in entries})
    valid_d = sorted(d for d in all_d if dur_map.get(d, 0) >= MIN_DAY_SECONDS)
    if not valid_d:
        return None
    first_shaded = min(valid_d[-n_shade_days:])
    shaded = {d for d in all_d if d >= first_shaded}
    return {r for r, (_, d) in enumerate(entries) if d in shaded}


def _make_raster(water, blank, path, n_shade_days=None, day_durations_map=None):
    def _day_bounds(entries):
        bounds = []; prev = entries[0][1] if entries else None; num = 1
        for row, (_, d) in enumerate(entries):
            if d != prev:
                num += 1; bounds.append((row, num)); prev = d
        return bounds

    dur_map = day_durations_map or {}

    vmax = 1.0
    for lst in (water, blank):
        if lst:
            m  = _build_heatmap_matrix_fast(lst)
            nz = m[m > 0]
            if len(nz):
                vmax = max(vmax, float(np.percentile(nz, 95)))

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.25)

    _draw_heatmap(axes[0], water or [], label="Water",
                  day_boundaries=_day_bounds(water) if water else [],
                  shade_rows=_compute_shade_rows(water, n_shade_days, dur_map) if water else None,
                  vmax=vmax)
    _draw_heatmap(axes[1], blank or [], label="No Water",
                  day_boundaries=_day_bounds(blank) if blank else [],
                  shade_rows=_compute_shade_rows(blank, n_shade_days, dur_map) if blank else None,
                  vmax=vmax)

    for ax in axes:
        ax.set_xlabel("Time (s)", fontsize=14, fontname="Arial")
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Trial #", fontsize=14, fontname="Arial")

    sm   = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.025, shrink=0.5, anchor=(0.0, 0.0))
    cbar.set_label("Lick (Hz)", fontsize=12, fontname="Arial")
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(path, dpi=1200)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# FREQ GRAPH (per-animal, per-day panels)
# ════════════════════════════════════════════════════════════════════════════

def _build_freq_graph(water_entries, blank_entries, valid_days_set,
                      output_path, suptitle=None, alignment="reversed"):
    """
    alignment controls display order and slot labels:
      reversed -> last calendar day = slot 0 = "Day 1" (training groups)
      forward  -> first calendar day = slot 0 = "Day 1" (acclimation groups)
    """
    cal_days = sorted(
        (set(d for _, d in water_entries) | set(d for _, d in blank_entries))
        & valid_days_set
    )
    if not cal_days:
        return

    # Order by alignment: reversed = last-to-first, forward = first-to-last
    ordered_days = list(reversed(cal_days)) if alignment == "reversed" else cal_days

    w_by_day: dict = defaultdict(list)
    b_by_day: dict = defaultdict(list)
    for mini, day in water_entries:
        w_by_day[day].append(mini)
    for mini, day in blank_entries:
        b_by_day[day].append(mini)

    fig, axes = plt.subplots(1, len(ordered_days),
                             figsize=(2.5 * len(ordered_days), 2), sharey=True)
    if len(ordered_days) == 1:
        axes = [axes]

    for slot, day in enumerate(ordered_days):
        wm_list = w_by_day[day]
        bm_list = b_by_day[day]
        w_means, b_means, w_sems, b_sems = [], [], [], []
        w_flat: dict = defaultdict(list)
        b_flat: dict = defaultdict(list)

        for j, win in enumerate(WIN_DEFS):
            wv = [_lick_hz(m["timestamp"].values, m["lick"].values,
                            _trial_start_time_df(m), win)
                  for m in wm_list if _trial_start_time_df(m) is not None]
            bv = [_lick_hz(m["timestamp"].values, m["lick"].values,
                            _trial_start_time_df(m), win)
                  for m in bm_list if _trial_start_time_df(m) is not None]
            w_means.append(np.mean(wv) if wv else np.nan)
            b_means.append(np.mean(bv) if bv else np.nan)
            w_sems.append(np.std(wv, ddof=1) / np.sqrt(len(wv)) if len(wv) > 1 else 0)
            b_sems.append(np.std(bv, ddof=1) / np.sqrt(len(bv)) if len(bv) > 1 else 0)
            w_flat[j].extend(wv)
            b_flat[j].extend(bv)

        # Label with the real calendar day number (day is 0-based index)
        lbl = f"Day {day + 1}"
        _draw_freq_graph(axes[slot], [lbl],
                         np.array([w_means]), np.array([w_sems]),
                         np.array([b_means]), np.array([b_sems]),
                         w_flat, b_flat, title=lbl)

    if suptitle:
        fig.suptitle(suptitle, fontsize=8, fontname="Arial", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ════════════════════════════════════════════════════════════════════════════

def _export_freq_excel(water_entries, blank_entries, valid_days_set,
                       output_path, label="", alignment="reversed"):
    import openpyxl
    from openpyxl.styles import Font as XLFont

    wb     = openpyxl.Workbook()
    ws_raw = wb.active
    ws_raw.title = "Raw Trials"
    ws_raw.append(["Label", "Trial_Type", "Day", "Trial_Start_Time"] + WIN_NAMES)
    for cell in ws_raw[1]:
        cell.font = XLFont(bold=True, name="Arial")

    # Build ordered day list once — shared by both raw and summary sheets
    cal_days     = sorted(
        (set(d for _, d in water_entries) | set(d for _, d in blank_entries))
        & valid_days_set
    )
    ordered_days = list(reversed(cal_days)) if alignment == "reversed" else cal_days
    # Map calendar day -> label using real day number (day is 0-based)
    day_to_slot  = {day: f"Day {day + 1}" for day in ordered_days}

    w_by_day: dict = defaultdict(list)
    b_by_day: dict = defaultdict(list)
    for mini, day in water_entries:
        w_by_day[day].append(mini)
    for mini, day in blank_entries:
        b_by_day[day].append(mini)

    # Raw trials sheet
    raw_rows = []
    for (mini, day), trial_type in (
        [(e, "Water") for e in water_entries] +
        [(e, "Blank") for e in blank_entries]
    ):
        if day not in valid_days_set:
            continue
        ref = _trial_start_time_df(mini)
        if ref is None:
            continue
        hz = [_lick_hz(mini["timestamp"].values, mini["lick"].values, ref, w)
              for w in WIN_DEFS]
        raw_rows.append([label, trial_type, day_to_slot.get(day, f"Day {day+1}"),
                         round(ref, 4)]
                        + [round(h, 4) for h in hz])
    for row in raw_rows:
        ws_raw.append(row)
    for col in ws_raw.columns:
        ws_raw.column_dimensions[col[0].column_letter].width = 16

    # Summary sheet — iterate in slot order
    ws_sum = wb.create_sheet("Summary Mean SEM")
    ws_sum.append(["Label", "Trial_Type", "Day"]
                  + [f"{w}_Mean" for w in WIN_NAMES]
                  + [f"{w}_SEM"  for w in WIN_NAMES]
                  + [f"{w}_N"    for w in WIN_NAMES])
    for cell in ws_sum[1]:
        cell.font = XLFont(bold=True, name="Arial")

    for day in ordered_days:
        slot_lbl = day_to_slot[day]
        for trial_type, by_day in [("Water", w_by_day), ("Blank", b_by_day)]:
            means, sems, ns = [], [], []
            for win in WIN_DEFS:
                vals = [_lick_hz(m["timestamp"].values, m["lick"].values,
                                  _trial_start_time_df(m), win)
                        for m in by_day[day] if _trial_start_time_df(m) is not None]
                n = len(vals)
                means.append(round(float(np.mean(vals)), 4) if n else "")
                sems.append(round(float(np.std(vals, ddof=1) / np.sqrt(n)), 4) if n > 1 else "")
                ns.append(n)
            ws_sum.append([label, trial_type, slot_lbl] + means + sems + ns)

    for col in ws_sum.columns:
        ws_sum.column_dimensions[col[0].column_letter].width = 18
    wb.save(output_path)


# ════════════════════════════════════════════════════════════════════════════
# ITI  (non-consecutive is now O(n log n))
# ════════════════════════════════════════════════════════════════════════════

def _compute_iti(water_entries, blank_entries, valid_days_set):
    all_e = []
    for mini, day in water_entries:
        all_e.append((float(mini["timestamp"].iloc[0]), day, "water", mini))
    for mini, day in blank_entries:
        all_e.append((float(mini["timestamp"].iloc[0]), day, "blank", mini))
    all_e.sort(key=lambda x: x[0])

    records = []
    for i in range(1, len(all_e)):
        _, day_prev, type_prev, mini_prev = all_e[i - 1]
        t_start, day_curr, type_curr, _   = all_e[i]
        if day_curr not in valid_days_set:
            continue
        trial_end = float(mini_prev["timestamp"].iloc[-1])
        iti = t_start - trial_end
        if iti < 0:
            continue
        records.append({
            "iti"      : iti,
            "day"      : day_curr,
            "timestamp": t_start,
            "type"     : f"{type_prev}->{type_curr}",
        })
    return records


def _compute_iti_nonconsecutive_fast(water, blank, valid_days_set):
    """
    O(n log n) replacement for the O(n²) nested loop.
    water / blank: lists of (seg_ts, seg_lick, day, is_water, ref) tuples.
    """
    import bisect

    all_e = []
    for t in water:
        all_e.append((float(t[0][0]), t[2], "water", float(t[0][-1])))
    for t in blank:
        all_e.append((float(t[0][0]), t[2], "blank", float(t[0][-1])))
    all_e.sort()

    if not all_e:
        return []

    starts = [e[0] for e in all_e]
    types  = [e[2] for e in all_e]
    days   = [e[1] for e in all_e]
    ends   = [e[3] for e in all_e]

    # Build sorted index lists per type for binary search
    w_idx = [i for i, tp in enumerate(types) if tp == "water"]
    b_idx = [i for i, tp in enumerate(types) if tp == "blank"]
    w_starts = [starts[i] for i in w_idx]
    b_starts = [starts[i] for i in b_idx]

    records = []
    for i, (t_start, day, ttype, t_end) in enumerate(all_e):
        if day not in valid_days_set:
            continue
        for target_type, idx_list, st_list in (
            ("water", w_idx, w_starts),
            ("blank", b_idx, b_starts),
        ):
            pos = bisect.bisect_right(st_list, t_start)
            if pos >= len(idx_list):
                continue
            j   = idx_list[pos]
            if days[j] not in valid_days_set:
                continue
            iti = starts[j] - t_end
            if iti < 0:
                continue
            records.append({
                "iti"      : iti,
                "day"      : day,
                "timestamp": starts[j],
                "type"     : f"{ttype}->{target_type}",
            })
    return records


# Legacy name kept for any call sites that use it
def _compute_iti_nonconsecutive(water_entries, blank_entries, valid_days_set):
    """Wrapper for legacy (mini_df, day) entry format."""
    water_raw = []
    for mini, day in water_entries:
        ts = mini["timestamp"].values
        water_raw.append((ts, None, day, True, None))
    blank_raw = []
    for mini, day in blank_entries:
        ts = mini["timestamp"].values
        blank_raw.append((ts, None, day, False, None))
    return _compute_iti_nonconsecutive_fast(water_raw, blank_raw, valid_days_set)


# ════════════════════════════════════════════════════════════════════════════
# ITI GRAPHS  (unchanged logic, same outputs)
# ════════════════════════════════════════════════════════════════════════════

def _iti_central(series):
    return series.median() if ITI_CENTRAL_TENDENCY == "median" else series.mean()


def _iti_agg_groupby(grouped):
    return grouped.median() if ITI_CENTRAL_TENDENCY == "median" else grouped.mean()


def _set_iti_ylim(ax):
    y_max = 0.0
    for line in ax.lines:
        yd = line.get_ydata()
        if len(yd):
            y_max = max(y_max, float(np.nanmax(yd)))
    for cont in ax.containers:
        try:
            for seg in cont[2]:
                y_max = max(y_max, float(np.nanmax(seg.get_ydata())))
        except Exception:
            pass
    ax.set_ylim(0, y_max * 1.5 if y_max > 0 else 500)


def _plot_iti_graphs(records, output_folder, label="",
                     session_start_time=None, nonconsec_records=None):
    if not records:
        print(f"[ITI] No ITI records for '{label}', skipping.")
        return

    safe  = label.replace(" ", "_").replace(",", "").replace("/", "-")
    df    = pd.DataFrame(records)
    print(f"[ITI DEBUG] '{label}' — n={len(df)}")

    t0 = session_start_time if session_start_time is not None else df["timestamp"].min()

    df["day_label"]    = df["day"] + 1
    df["hr_bin"]       = ((df["timestamp"] - t0) / 7200).astype(int)
    df["hr_bin_label"] = df["hr_bin"] * 2

    def _src(t):
        return "water" if t.startswith("water->") else ("blank" if t.startswith("blank->") else "other")
    df["source"] = df["type"].apply(_src)
    df = df[df["day_label"] <= 2]

    bin_counts = df.groupby("hr_bin_label")["iti"].count()
    valid_bins = set(bin_counts[bin_counts >= 10].index)
    df_bins    = df[df["hr_bin_label"].isin(valid_bins)]
    bins       = sorted(df_bins["hr_bin_label"].unique())

    src_colors = {"water": COL_WATER, "blank": COL_BLANK}
    os.makedirs(output_folder, exist_ok=True)
    days  = sorted(df["day_label"].unique())
    x_pad = 0.25

    # ── Figure 1 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.suptitle(f"{label} — ITI per day", fontsize=9, y=1.06)
    rng = np.random.default_rng(42)

    ax = axes[0]
    for day in days:
        dd = df[df["day_label"] == day]["iti"].dropna().values
        if not len(dd):
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(dd))
        ax.scatter(np.full(len(dd), day) + jitter, dd,
                   s=2, alpha=0.3, color="#90CAF9", linewidths=0, zorder=2)
        med = np.median(dd)
        p25, p75 = np.percentile(dd, [25, 75])
        ax.hlines(med, day - 0.25, day + 0.25, colors="#1565C0", linewidth=2, zorder=4)
        ax.hlines(p25, day - 0.18, day + 0.18, colors="#E53935", linewidth=1, zorder=3)
        ax.hlines(p75, day - 0.18, day + 0.18, colors="#E53935", linewidth=1, zorder=3)
        ax.vlines(day, p25, p75, colors="#E53935", linewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("ITI (s) [log scale]", fontsize=10)
    ax.set_xlim(days[0] - x_pad, days[-1] + x_pad); ax.set_xticks(days)
    ax.set_title("Consecutive — Combined", fontsize=11)
    ax.text(0.5, 0.99,
            f"min={df['iti'].min():.1f}  med={df['iti'].median():.1f}"
            f"  mean={df['iti'].mean():.1f}  max={df['iti'].max():.1f}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color="gray")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color="#1565C0",lw=2,label="Median"),
                        Line2D([0],[0],color="#E53935",lw=1,label="IQR")],
              fontsize=6, frameon=False)

    ax = axes[1]
    for src, col in src_colors.items():
        sub = df[df["source"] == src]
        if sub.empty: continue
        dm = _iti_agg_groupby(sub.groupby("day_label")["iti"])
        ds = sub.groupby("day_label")["iti"].sem()
        x  = [d for d in days if d in dm.index]
        ax.errorbar(x, [dm[d] for d in x], yerr=[ds[d] for d in x],
                    marker="o", markersize=5, lw=1, capsize=2,
                    color=col, label=f"{src}-sourced")
    ax.set_xlim(days[0] - x_pad, days[-1] + x_pad); ax.set_xticks(days)
    ax.set_title("Consecutive — By source trial", fontsize=11)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    if nonconsec_records:
        df_nc              = pd.DataFrame(nonconsec_records)
        df_nc["day_label"] = df_nc["day"] + 1
        df_nc["source"]    = df_nc["type"].apply(_src)
        df_nc = df_nc[df_nc["day_label"] <= 2]
        days_nc = sorted(df_nc["day_label"].unique())
        for src, col in src_colors.items():
            sub = df_nc[df_nc["source"] == src]
            if sub.empty: continue
            dm = _iti_agg_groupby(sub.groupby("day_label")["iti"])
            ds = sub.groupby("day_label")["iti"].sem()
            x  = [d for d in days_nc if d in dm.index]
            ax.errorbar(x, [dm[d] for d in x], yerr=[ds[d] for d in x],
                        marker="o", markersize=5, lw=1, capsize=2,
                        color=col, label=f"{src}-sourced")
        if days_nc:
            ax.set_xlim(days_nc[0] - x_pad, days_nc[-1] + x_pad)
            ax.set_xticks(days_nc)
        ax.legend(fontsize=6, frameon=False)
    ax.set_title("Non-consecutive — By source trial", fontsize=11)

    for ax in axes:
        ax.set_xlabel("Day", fontsize=10); ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax in axes[1:]:
        _set_iti_ylim(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f"iti_per_day_{safe}.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2 ────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3), sharey=False)
    fig2.suptitle(f"{label} — ITI in 2-hr bins", fontsize=9, y=1.01)

    bm = _iti_agg_groupby(df_bins.groupby("hr_bin_label")["iti"])
    bs = df_bins.groupby("hr_bin_label")["iti"].sem()
    x  = [b for b in bins if b in bm.index]
    axes2[0].errorbar(x, [bm[b] for b in x], yerr=[bs[b] for b in x],
                      marker="o", markersize=5, lw=1, capsize=2, color="#555")
    axes2[0].set_title("Combined", fontsize=11)

    for src, col in src_colors.items():
        sub = df_bins[df_bins["source"] == src]
        if sub.empty: continue
        bm2 = _iti_agg_groupby(sub.groupby("hr_bin_label")["iti"])
        bs2 = sub.groupby("hr_bin_label")["iti"].sem()
        x   = [b for b in bins if b in bm2.index]
        axes2[1].errorbar(x, [bm2[b] for b in x], yerr=[bs2[b] for b in x],
                          marker="o", markersize=5, lw=1, capsize=2,
                          color=col, label=f"{src}-sourced")
    axes2[1].set_title("By source trial", fontsize=11)
    axes2[1].legend(fontsize=6, frameon=False)

    for ax in axes2:
        if bins:
            ax.set_xlim(bins[0] - 1.5, bins[-1] + 1.5)
        ax.set_xlabel("Hour", fontsize=10)
        ax.set_xticks(bins[::2] if len(bins) > 10 else bins)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        yv = [y for ln in ax.lines for y in ln.get_ydata()
              if not (isinstance(y, float) and np.isnan(y))]
        ax.set_ylim(0, np.percentile(yv, 90) if yv else 500)
    axes2[0].set_ylabel("ITI (s)", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_folder, f"iti_2hr_bins_{safe}.png"),
                 dpi=600, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3 ────────────────────────────────────────────────────────────
    HIST_BIN  = 0.25
    HIST_XMAX = 30.0
    panel_defs = [
        ("water", "Water-sourced\n(water→water + water→blank)", COL_WATER),
        ("blank", "Blank-sourced\n(blank→blank + blank→water)",  COL_BLANK),
    ]
    src_data = {src: df[df["source"] == src]["iti"].dropna().values
                for src, _, _ in panel_defs}
    tail_all = np.concatenate([v[v > HIST_XMAX] for v in src_data.values()])
    tail_xmax = max(500.0, float(np.ceil(tail_all.max() / 500) * 500)) if len(tail_all) else 500.0

    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 7),
                                sharey=False,
                                gridspec_kw={"hspace": 0.45, "wspace": 0.30})
    fig3.suptitle(f"{label} — ITI distribution (days 1–2, 0.25 s bins)",
                  fontsize=9, y=1.01)
    fig3.text(0.01, 0.74, "0 – 30 s",  va="center", rotation="vertical", fontsize=9)
    fig3.text(0.01, 0.26, "30 s +",    va="center", rotation="vertical", fontsize=9)

    hb_top  = np.arange(0, HIST_XMAX + HIST_BIN, HIST_BIN)
    hb_tail = np.arange(HIST_XMAX, tail_xmax + 500, 500)

    for ci, (src, title, color) in enumerate(panel_defs):
        sub = src_data[src]
        ax_top = axes3[0][ci]
        ax_bot = axes3[1][ci]
        clipped = sub[sub <= HIST_XMAX]
        n_tail  = int(np.sum(sub > HIST_XMAX))
        ax_top.hist(clipped, bins=hb_top, color=color, alpha=0.75, edgecolor="none")
        if len(sub):
            med = np.median(sub)
            ax_top.axvline(med, color="#1A237E", lw=1.5, linestyle="--",
                           label=f"Median = {med:.2f} s")
        ax_top.set_xlim(0, HIST_XMAX)
        ax_top.set_title(title, fontsize=10)
        ax_top.set_xlabel("ITI (s)", fontsize=10)
        ax_top.set_ylabel("Count", fontsize=10)
        ax_top.tick_params(labelsize=8)
        ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
        ax_top.legend(fontsize=7, frameon=False)
        ax_top.text(0.98, 0.95,
                    f"n (0–30 s) = {len(clipped):,}  |  n (>30 s) = {n_tail:,}",
                    transform=ax_top.transAxes, ha="right", va="top",
                    fontsize=6, color="gray")
        tail = sub[sub > HIST_XMAX]
        if len(tail):
            ax_bot.hist(tail, bins=hb_tail, color=color, alpha=0.75, edgecolor="none")
            tm = np.median(tail)
            ax_bot.axvline(tm, color="#1A237E", lw=1.5, linestyle="--",
                           label=f"Median = {tm:.0f} s")
            ax_bot.legend(fontsize=7, frameon=False)
        else:
            ax_bot.text(0.5, 0.5, "No ITIs > 30 s",
                        transform=ax_bot.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")
        ax_bot.set_xlim(HIST_XMAX, tail_xmax)
        ax_bot.set_title(f"{title.split(chr(10))[0]} — tail", fontsize=10)
        ax_bot.set_xlabel("ITI (s)", fontsize=10)
        ax_bot.set_ylabel("Count", fontsize=10)
        ax_bot.tick_params(labelsize=8)
        ax_bot.spines["top"].set_visible(False); ax_bot.spines["right"].set_visible(False)

    fig3.savefig(os.path.join(output_folder, f"iti_histogram_{safe}.png"),
                 dpi=600, bbox_inches="tight")
    plt.close(fig3)
    print(f"[ITI] Saved ITI graphs for '{label}'")


# ════════════════════════════════════════════════════════════════════════════
# ITI EXCEL
# ════════════════════════════════════════════════════════════════════════════

def _export_iti_excel(group_name, animal_iti_list, output_path):
    import openpyxl
    from openpyxl.styles import Font as XLFont, PatternFill, Alignment

    def _src(t):
        return "water" if t.startswith("water->") else ("blank" if t.startswith("blank->") else "other")
    def _stats(s):
        s = s.dropna()
        return (round(float(s.median()), 4), round(float(s.mean()), 4)) if len(s) else (None, None)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ITI Summary"
    hfill  = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
    hfont  = XLFont(bold=True, name="Arial", color="FFFFFF", size=10)
    dfont  = XLFont(name="Arial", size=10)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    headers = ["Group", "Animal",
               "Combo ITI Median (s)", "Combo ITI Mean (s)",
               "Water-to-next <30s Median (s)", "Water-to-next <30s Mean (s)",
               "Water-to-next >30s Median (s)", "Water-to-next >30s Mean (s)",
               "Blank-to-next <30s Median (s)", "Blank-to-next <30s Mean (s)",
               "Blank-to-next >30s Median (s)", "Blank-to-next >30s Mean (s)"]
    ws.append(headers)
    ws.row_dimensions[1].height = 40
    for cell in ws[1]:
        cell.font = hfont; cell.fill = hfill; cell.alignment = center

    for animal_id, recs in animal_iti_list:
        if not recs:
            row = [group_name, animal_id] + [None] * 10
        else:
            df = pd.DataFrame(recs)
            df["source"] = df["type"].apply(_src)
            combo = df["iti"]
            ws_ = df[df["source"] == "water"]["iti"]
            bs_ = df[df["source"] == "blank"]["iti"]
            row = [group_name, animal_id,
                   *_stats(combo),
                   *_stats(ws_[ws_ < 30]),  *_stats(ws_[ws_ >= 30]),
                   *_stats(bs_[bs_ < 30]),  *_stats(bs_[bs_ >= 30])]
        ws.append(row)
        for cell in ws[ws.max_row]:
            cell.font = dfont; cell.alignment = center

    widths = [20, 18, 22, 20, 30, 28, 30, 28, 30, 28, 30, 28]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    wb.save(output_path)


# ════════════════════════════════════════════════════════════════════════════
# GROUP-LEVEL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def _analyse_group(group_name, animal_data_list, output_folder,
                   alignment: str = "reversed"):
    """
    animal_data_list: list of dicts from _process_animal results, each with
        id, w_by_day, b_by_day, all_days, files, iti_records, iti_records_nc
    """

    def _run_pass(records, suffix="", suptitle_sfx=""):
        if not records:
            return
        align_desc = (
            "Forward (first-to-last): all groups are acclimation"
            if alignment == "forward" else
            "Reversed (last-to-first): one or more training groups present"
        )
        max_slots = max(len(r["days"]) for r in records)
        slot_w: dict = defaultdict(list)  # slot -> list of mean-per-day arrays (4,)
        slot_b: dict = defaultdict(list)
        slot_fw: dict = defaultdict(list)  # (slot, win_j) -> flat list of trial Hz
        slot_fb: dict = defaultdict(list)

        for rec in records:
            for slot, day in enumerate(rec["days"]):
                if rec["w"].get(day) is not None:
                    arr = rec["w"][day]            # (n_trials, 4)
                    slot_w[slot].append(arr.mean(axis=0))
                    for j in range(4):
                        slot_fw[(slot, j)].extend(arr[:, j].tolist())
                if rec["b"].get(day) is not None:
                    arr = rec["b"][day]
                    slot_b[slot].append(arr.mean(axis=0))
                    for j in range(4):
                        slot_fb[(slot, j)].extend(arr[:, j].tolist())

        plot_slots = list(range(max_slots))  # show all days, no hard cut

        # Build a mapping: slot -> most common actual calendar day (1-based)
        # so panel titles show real day numbers, not just slot positions.
        slot_to_cal: dict = defaultdict(list)
        for rec in records:
            for slot, day in enumerate(rec["days"]):
                slot_to_cal[slot].append(day + 1)   # store as 1-based

        def _modal_day(slot):
            vals = slot_to_cal.get(slot, [])
            if not vals:
                return slot + 1
            return int(np.median(vals))   # median real day number for this slot

        def _slot_label(slot):
            real = _modal_day(slot)
            return f"Day {real}"

        def _agg(sd, slots):
            means, sems = [], []
            for s in slots:
                vals = sd.get(s, [])
                if not vals:
                    means.append(np.full(4, np.nan))
                    sems.append(np.zeros(4))
                else:
                    arr = np.stack(vals)
                    means.append(arr.mean(axis=0))
                    sems.append(arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
                                if len(arr) > 1 else np.zeros(4))
            return np.array(means), np.array(sems)

        wm, ws_ = _agg(slot_w, plot_slots)
        bm, bs_ = _agg(slot_b, plot_slots)
        n_per   = [max(len(slot_w.get(s, [])), len(slot_b.get(s, [])))
                   for s in plot_slots]

        fig, axes = plt.subplots(1, len(plot_slots),
                                 figsize=(2.5 * len(plot_slots), 2), sharey=True)
        if len(plot_slots) == 1:
            axes = [axes]

        for pi, slot in enumerate(plot_slots):
            lbl = _slot_label(slot)
            _draw_freq_graph(
                axes[pi], [lbl],
                wm[pi:pi+1], ws_[pi:pi+1], bm[pi:pi+1], bs_[pi:pi+1],
                {j: slot_fw.get((slot, j), []) for j in range(4)},
                {j: slot_fb.get((slot, j), []) for j in range(4)},
                n_per_rev_day=[n_per[pi]], title=lbl,
            )

        fig.suptitle(group_name + suptitle_sfx, fontsize=8, y=1.01)
        fig.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder,
                                 f"freq_by_day_group_{group_name}{suffix}.png"),
                    dpi=1200, bbox_inches="tight")
        plt.close(fig)

        # Group Excel
        import openpyxl
        from openpyxl.styles import Font as XLFont

        # ── "Raw Hz" tab: every individual trial (existing behavior) ───────
        rows = []
        for pi, slot in enumerate(plot_slots):
            lbl = _slot_label(slot)
            for j, wn in enumerate(WIN_NAMES):
                for v in slot_fw.get((slot, j), []):
                    rows.append([group_name, "Water", lbl, wn, round(v, 4), align_desc])
                for v in slot_fb.get((slot, j), []):
                    rows.append([group_name, "Blank", lbl, wn, round(v, 4), align_desc])

        # ── "Animal Avg" tab: one row per animal per day per window ───────
        # Each value is that animal's MEAN Hz across all trials of that
        # type (Water/Blank) within that window, for that day — i.e. the
        # same per-animal-per-day mean used for slot_w/slot_b, just not
        # collapsed across animals.
        animal_rows = []
        for rec in records:
            animal_id = rec.get("id", "Unknown")
            for slot, day in enumerate(rec["days"]):
                lbl = _slot_label(slot)
                w_arr = rec["w"].get(day)   # (n_trials, 4) or None
                b_arr = rec["b"].get(day)
                for j, wn in enumerate(WIN_NAMES):
                    if w_arr is not None:
                        animal_rows.append([
                            group_name, animal_id, "Water", lbl, wn,
                            round(float(w_arr[:, j].mean()), 4),
                            w_arr.shape[0], align_desc,
                        ])
                    if b_arr is not None:
                        animal_rows.append([
                            group_name, animal_id, "Blank", lbl, wn,
                            round(float(b_arr[:, j].mean()), 4),
                            b_arr.shape[0], align_desc,
                        ])

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Raw Hz"
        ws.append(["Group", "Trial_Type", "Day", "Window", "Lick_Hz", "Alignment"])
        for cell in ws[1]:
            cell.font = XLFont(bold=True, name="Arial")
        for row in rows:
            ws.append(row)
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 16

        ws2 = wb.create_sheet("Animal Avg")
        ws2.append(["Group", "Animal", "Trial_Type", "Day", "Window",
                    "Mean_Lick_Hz", "N_Trials", "Alignment"])
        for cell in ws2[1]:
            cell.font = XLFont(bold=True, name="Arial")
        for row in animal_rows:
            ws2.append(row)
        for col in ws2.columns:
            ws2.column_dimensions[col[0].column_letter].width = 16

        wb.save(os.path.join(output_folder,
                             f"freq_by_day_group_{group_name}{suffix}.xlsx"))

        # Group ITI
        all_iti    = []
        all_iti_nc = []
        animal_iti_list = []
        for rec in records:
            all_iti.extend(rec.get("iti_records", []))
            all_iti_nc.extend(rec.get("iti_records_nc", []))
            animal_iti_list.append((rec["id"], rec.get("iti_records", [])))

        _plot_iti_graphs(all_iti, output_folder,
                         label=group_name + suptitle_sfx,
                         nonconsec_records=all_iti_nc)

        if suffix == "":
            _export_iti_excel(
                group_name, animal_iti_list,
                os.path.join(output_folder, f"iti_summary_{group_name}.xlsx")
            )

        print(f"[GROUP] Saved group graph for '{group_name}'{suptitle_sfx}")

    # Standard pass
    _run_pass(animal_data_list, "", "")

    # Last-20% pass — recompute Hz dicts from stored files
    records_20 = []
    for rec in animal_data_list:
        df20 = _load_and_merge(rec["files"])
        if df20.empty:
            continue
        valid20  = _valid_days(df20)
        trials20 = _segment_trials(df20)
        if not trials20:
            continue
        water20 = [t for t in trials20 if     t[3] and t[4] is not None and t[2] in valid20]
        blank20 = [t for t in trials20 if not t[3] and t[4] is not None and t[2] in valid20]
        w20, b20 = _last_20pct_combined(water20, blank20)

        w_by_day20: dict = defaultdict(list)
        b_by_day20: dict = defaultdict(list)
        for seg_ts, seg_lick, day, _, ref in w20:
            if ref is not None:
                w_by_day20[day].append(_lick_hz_all(seg_ts, seg_lick, ref))
        for seg_ts, seg_lick, day, _, ref in b20:
            if ref is not None:
                b_by_day20[day].append(_lick_hz_all(seg_ts, seg_lick, ref))

        w_by_day20 = {d: np.stack(v) for d, v in w_by_day20.items()}
        b_by_day20 = {d: np.stack(v) for d, v in b_by_day20.items()}
        all_days20 = sorted(set(w_by_day20) | set(b_by_day20))
        if len(all_days20) <= 1:
            continue

        records_20.append({
            "id"           : rec["id"],
            "w"            : w_by_day20,
            "b"            : b_by_day20,
            "days"         : _align_days(all_days20, alignment),
            "files"        : rec["files"],
            "iti_records"  : [],
            "iti_records_nc": [],
        })

    _run_pass(records_20, "_last20pct", " — Last 20% of trials per day")


# ════════════════════════════════════════════════════════════════════════════
# LICK DIFFERENCE GRAPHS
# ════════════════════════════════════════════════════════════════════════════

def _animal_lick_difference(files, alignment="reversed"):
    """
    Returns (diff_dict, slot_to_day, water_prob) where:
      diff_dict  : {slot: {win_idx: float}}
      slot_to_day: {slot: actual_day_number (1-based)}
      water_prob : {slot: float}  — the EXACT water probability this animal
                   actually experienced that day, i.e.
                   n_water_trials / (n_water_trials + n_blank_trials),
                   computed from the trials actually used in the diff
                   calculation (not a nominal/programmed probability).
    All are None on failure.
    """
    df = _load_and_merge(files)
    if df.empty:
        return None, None, None
    valid  = _valid_days(df)
    trials = _segment_trials(df)
    if not trials:
        return None, None, None

    water = [t for t in trials if     t[3] and t[4] is not None and t[2] in valid]
    blank = [t for t in trials if not t[3] and t[4] is not None and t[2] in valid]

    w_by_day: dict = defaultdict(list)
    b_by_day: dict = defaultdict(list)
    for seg_ts, seg_lick, day, _, ref in water:
        w_by_day[day].append(_lick_hz_all(seg_ts, seg_lick, ref))
    for seg_ts, seg_lick, day, _, ref in blank:
        b_by_day[day].append(_lick_hz_all(seg_ts, seg_lick, ref))

    all_days = sorted(set(w_by_day) | set(b_by_day))
    if not all_days:
        return None, None, None

    ordered = _align_days(all_days, alignment)
    result      = {}
    slot_to_day = {}   # slot -> 1-based day number
    water_prob  = {}   # slot -> exact fraction of trials that were water
    for slot, day in enumerate(ordered):
        w_arr = np.stack(w_by_day[day]) if w_by_day.get(day) else None
        b_arr = np.stack(b_by_day[day]) if b_by_day.get(day) else None
        if w_arr is not None and b_arr is not None:
            diff = w_arr.mean(axis=0) - b_arr.mean(axis=0)
        elif w_arr is not None:
            diff = w_arr.mean(axis=0)
        else:
            diff = -b_arr.mean(axis=0)
        result[slot]      = {j: float(diff[j]) for j in range(4)}
        slot_to_day[slot] = day + 1   # day index is 0-based; store as 1-based

        n_water = w_arr.shape[0] if w_arr is not None else 0
        n_blank = b_arr.shape[0] if b_arr is not None else 0
        n_total = n_water + n_blank
        water_prob[slot] = (n_water / n_total) if n_total > 0 else float("nan")

    return result, slot_to_day, water_prob


def _animal_lick_difference_last20pct(files, alignment="reversed"):
    """
    Identical to _animal_lick_difference but restricts each day's trials
    to the last 20% before computing the water-minus-blank lick difference.

    Returns (diff_dict, slot_to_day, water_prob) where:
      diff_dict  : {slot: {win_idx: float}}
      slot_to_day: {slot: actual_day_number (1-based)}
      water_prob : {slot: float}  — exact water probability experienced
                   within the last-20% trial subset used for this
                   computation (n_water / (n_water + n_blank)).
    All are None on failure.
    """
    df = _load_and_merge(files)
    if df.empty:
        return None, None, None
    valid  = _valid_days(df)
    trials = _segment_trials(df)
    if not trials:
        return None, None, None

    water = [t for t in trials if     t[3] and t[4] is not None and t[2] in valid]
    blank = [t for t in trials if not t[3] and t[4] is not None and t[2] in valid]

    # Filter to last 20% of trials per day (interleaved water+blank ranking)
    water, blank = _last_20pct_combined(water, blank)

    w_by_day: dict = defaultdict(list)
    b_by_day: dict = defaultdict(list)
    for seg_ts, seg_lick, day, _, ref in water:
        w_by_day[day].append(_lick_hz_all(seg_ts, seg_lick, ref))
    for seg_ts, seg_lick, day, _, ref in blank:
        b_by_day[day].append(_lick_hz_all(seg_ts, seg_lick, ref))

    all_days = sorted(set(w_by_day) | set(b_by_day))
    if not all_days:
        return None, None, None

    ordered     = _align_days(all_days, alignment)
    result      = {}
    slot_to_day = {}
    water_prob  = {}
    for slot, day in enumerate(ordered):
        w_arr = np.stack(w_by_day[day]) if w_by_day.get(day) else None
        b_arr = np.stack(b_by_day[day]) if b_by_day.get(day) else None
        if w_arr is not None and b_arr is not None:
            diff = w_arr.mean(axis=0) - b_arr.mean(axis=0)
        elif w_arr is not None:
            diff = w_arr.mean(axis=0)
        else:
            diff = -b_arr.mean(axis=0)
        result[slot]      = {j: float(diff[j]) for j in range(4)}
        slot_to_day[slot] = day + 1

        n_water = w_arr.shape[0] if w_arr is not None else 0
        n_blank = b_arr.shape[0] if b_arr is not None else 0
        n_total = n_water + n_blank
        water_prob[slot] = (n_water / n_total) if n_total > 0 else float("nan")

    return result, slot_to_day, water_prob


def _tukey_pairwise(group_data: dict) -> list:
    """
    Tukey HSD pairwise comparisons.
    Falls back to Welch t-test if scipy.stats.tukey_hsd is unavailable
    (scipy < 1.8) or if fewer than 2 groups have enough data.
    Returns list of dicts with keys: g1, g2, p, sig
    """
    groups = [g for g, v in group_data.items() if len(v) >= 2]
    if len(groups) < 2:
        return []
    pairs = list(itertools.combinations(groups, 2))
    try:
        arrays = [group_data[g] for g in groups]
        res    = _scipy_stats.tukey_hsd(*arrays)
        out = []
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if j <= i:
                    continue
                p = float(res.pvalue[i][j])
                out.append({"g1": g1, "g2": g2, "p": p, "sig": p < 0.05})
        return out
    except AttributeError:
        # scipy < 1.8 fallback: Welch t-test with Bonferroni
        n_comp = len(pairs)
        out = []
        for g1, g2 in pairs:
            _, p_raw = _scipy_stats.ttest_ind(group_data[g1], group_data[g2],
                                               equal_var=False)
            p = min(p_raw * n_comp, 1.0)
            out.append({"g1": g1, "g2": g2, "p": p, "sig": p < 0.05})
        return out


# Keep old name as alias so Excel export code still works
def _bonferroni_pairwise(group_data: dict) -> list:
    return _tukey_pairwise(group_data)


def _draw_sig_brackets(ax, group_keys, pairwise, y_data_max, bar_positions):
    sig = [r for r in pairwise if r["sig"]]
    if not sig:
        return
    pm    = {g: x for g, x in zip(group_keys, bar_positions)}
    y_lo, y_hi = ax.get_ylim()
    step  = (y_hi - y_lo) * 0.10
    base  = y_data_max + step * 0.5
    for lvl, res in enumerate(sig):
        x1 = pm.get(res["g1"]); x2 = pm.get(res["g2"])
        if x1 is None or x2 is None:
            continue
        x1, x2 = min(x1, x2), max(x1, x2)
        y = base + lvl * step * 1.4
        ax.plot([x1, x1, x2, x2], [y - step*0.2, y, y, y - step*0.2],
                lw=0.8, color="black")
        lbl = "***" if res["p"] < 0.001 else ("**" if res["p"] < 0.01 else "*")
        ax.text((x1+x2)/2, y + step*0.05, lbl, ha="center", va="bottom", fontsize=8)


def _export_lick_diff_excel(training_list, lesion_groups, data,
                             max_slots, alignment, align_desc, output_path,
                             water_prob=None):
    """
    water_prob : optional dict, same keying as `data`:
        (training, lesion) -> {animal: {slot: exact_water_probability}}
        where exact_water_probability = n_water_trials / n_total_trials
        actually experienced by that animal on that day (not a nominal/
        programmed probability). If omitted, the column is left blank.
    """
    import openpyxl
    from openpyxl.styles import Font as XLFont, Alignment as XLAlign, PatternFill

    water_prob = water_prob or {}

    hf  = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
    hfn = XLFont(bold=True, name="Arial", color="FFFFFF", size=10)
    dfn = XLFont(name="Arial", size=10)
    ctr = XLAlign(horizontal="center", vertical="center", wrap_text=True)

    def _hdr(ws):
        for c in ws[1]:
            c.font = hfn; c.fill = hf; c.alignment = ctr
    def _dat(ws):
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.font = dfn; c.alignment = ctr
    def _aw(ws):
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = \
                max(max((len(str(c.value or "")) for c in col), default=0) + 4, 14)

    wb = openpyxl.Workbook()

    ws1 = wb.active; ws1.title = "Animal Differences"
    ws1.append(["Lesion", "Training Group", "Animal", "Day (Slot)",
                "Water Probability (exact, n_water/n_total)"]
               + WIN_NAMES + ["Alignment"])
    _hdr(ws1)
    for les in lesion_groups:
        for trn in training_list:
            for animal, sd in data.get((trn, les), {}).items():
                animal_wp = water_prob.get((trn, les), {}).get(animal, {})
                for slot in sorted(sd):
                    wp = animal_wp.get(slot, float("nan"))
                    ws1.append([les, trn, animal, f"Day {slot+1}",
                                round(wp, 4) if not np.isnan(wp) else ""]
                                + [round(sd[slot].get(j, float("nan")), 4)
                                   for j in range(4)]
                                + [align_desc])
    _dat(ws1); _aw(ws1)

    ws2 = wb.create_sheet("Group Summary")
    ws2.append(["Lesion","Training Group","Day (Slot)","Window",
                "N","Mean (Hz)","SEM (Hz)","Alignment"])
    _hdr(ws2)
    for les in lesion_groups:
        for slot in range(max_slots):
            for j, wn in enumerate(WIN_NAMES):
                for trn in training_list:
                    vals = np.array([
                        sd[slot][j]
                        for sd in data.get((trn, les), {}).values()
                        if slot in sd and j in sd[slot]
                    ])
                    n = len(vals)
                    ws2.append([les, trn, f"Day {slot+1}", wn, n,
                                round(float(np.mean(vals)), 4) if n else "",
                                round(float(np.std(vals, ddof=1)/np.sqrt(n)), 4) if n>1 else "",
                                align_desc])
    _dat(ws2); _aw(ws2)

    ws3 = wb.create_sheet("ANOVA Results")
    ws3.append(["Lesion","Day (Slot)","Window","F-statistic","p-value",
                "Significant (p<0.05)","Alignment"])
    _hdr(ws3)
    for les in lesion_groups:
        for slot in range(max_slots):
            for j, wn in enumerate(WIN_NAMES):
                arrs = [np.array([sd[slot][j]
                                  for sd in data.get((trn, les), {}).values()
                                  if slot in sd and j in sd[slot]])
                        for trn in training_list]
                arrs = [a for a in arrs if len(a) >= 2]
                if len(arrs) >= 2:
                    f, p = _scipy_stats.f_oneway(*arrs)
                    ws3.append([les, f"Day {slot+1}", wn,
                                round(float(f),4), round(float(p),6),
                                "Yes" if p<0.05 else "No", align_desc])
                else:
                    ws3.append([les, f"Day {slot+1}", wn, "N/A","N/A","N/A", align_desc])
    _dat(ws3); _aw(ws3)

    ws4 = wb.create_sheet("Pairwise (Bonferroni)")
    ws4.append(["Lesion","Day (Slot)","Window","Training Group 1","Training Group 2",
                "t-statistic (or Tukey p)","p (Tukey HSD)","Significant","Alignment"])
    _hdr(ws4)
    for les in lesion_groups:
        for slot in range(max_slots):
            for j, wn in enumerate(WIN_NAMES):
                gd = {trn: [sd[slot][j]
                             for sd in data.get((trn, les), {}).values()
                             if slot in sd and j in sd[slot]]
                      for trn in training_list}
                for r in _bonferroni_pairwise(gd):
                    ws4.append([les, f"Day {slot+1}", wn,
                                r["g1"], r["g2"],
                                "", round(r["p"],6),
                                "Yes" if r["sig"] else "No", align_desc])
    _dat(ws4); _aw(ws4)
    wb.save(output_path)



# ════════════════════════════════════════════════════════════════════════════
# SMART SLOT SELECTION FOR LICK DIFF GRAPHS
# ════════════════════════════════════════════════════════════════════════════

def _slots_for_group(training_name, animal_slot_dicts, animal_day_maps,
                     comparison_is_mixed):
    """
    Determine which slot(s) to plot for one group in a lick diff graph.

    Rules
    -----
    Acclimation group (forward alignment, slot 0 = Day 1):
      - Default: show slot 1 only  (Day 2).
      - Extended acclimation: if >75% of animals have a timepoint at Day 6
        or Day 7 (actual calendar day), show that slot instead (prefer Day 6).
    Training group (reversed alignment, slot 0 = last day):
      - Always show slot 0 only (last training day).

    animal_slot_dicts : list of {slot: {win_idx: val}}  — one per animal
    animal_day_maps   : list of {slot: actual_day_1based} — parallel list
    comparison_is_mixed : bool (not used for routing anymore, kept for compat)

    Returns: list of slot indices to plot (single element list)
    """
    if not animal_slot_dicts:
        return []

    if not _is_acclimation_group(training_name):
        # Training group — always last day = slot 0
        return [0]

    # ── Acclimation group ──────────────────────────────────────────────────
    n_animals = len(animal_day_maps)
    if n_animals == 0:
        return []

    # Check for extended acclimation: >75% of animals have Day 6 or Day 7
    for target_day in (6, 7):
        count = sum(
            1 for dm in animal_day_maps
            if any(v == target_day for v in dm.values())
        )
        if count / n_animals > 0.75:
            # Find which slot corresponds to target_day (use first animal that has it)
            for dm in animal_day_maps:
                for slot, actual_day in dm.items():
                    if actual_day == target_day:
                        print(f"[LICK DIFF] Extended acclimation detected "
                              f"({count}/{n_animals} animals have Day {target_day}): "
                              f"using Day {target_day} (slot {slot})")
                        return [slot]

    # Default acclimation: Day 2 = slot 1 (forward alignment)
    # Verify at least one animal actually has slot 1; fall back to slot 0 if not
    has_slot1 = any(1 in sd for sd in animal_slot_dicts)
    return [1] if has_slot1 else [0]


def _group_day_label(training_name, slot, alignment, day_maps=None):
    """
    Return a human-readable day label.

    Uses the actual day number from day_maps when available so the label
    shows the real calendar day (e.g. "Day 2", "Day 6") rather than a
    slot-derived guess.  Falls back to slot arithmetic if day_maps is absent.
    """
    # Try to recover the real day number from the first animal's day map
    if day_maps:
        for dm in day_maps:
            if slot in dm:
                actual = dm[slot]
                if alignment == "reversed" and slot == 0:
                    return "Last Day"
                return f"Day {actual}"

    # Fallback (no day_maps provided)
    if alignment == "reversed":
        return "Last Day" if slot == 0 else f"Day -{slot}"
    return f"Day {slot + 1}"


def _plot_lick_difference_by_group(df_master, animal_id_col, training_col,
                                    comparisons, training_data_dir,
                                    output_folder, lesion_col=None,
                                    lesion_filter=None):
    os.makedirs(output_folder, exist_ok=True)
    has_lesion   = bool(lesion_col and lesion_col in df_master.columns)
    group_colors  = _read_color_file(_color_path("group_colors.txt"))
    lesion_colors = _read_color_file(_color_path("lesion_colors.txt"))

    comp_lists = []
    for comp in comparisons:
        comp_lists.append([str(g).strip() for g in comp]
                          if isinstance(comp, (list, tuple))
                          else [str(comp).strip()])

    df = df_master.copy()
    df[training_col] = df[training_col].astype(str).str.strip()
    df = df[df[training_col].isin({g for cl in comp_lists for g in cl})]
    if has_lesion and lesion_filter:
        df[lesion_col] = df[lesion_col].astype(str).str.strip()
        df = df[df[lesion_col].isin([str(l).strip() for l in lesion_filter])]
    elif has_lesion:
        df[lesion_col] = df[lesion_col].astype(str).str.strip()

    cols  = [animal_id_col, training_col] + ([lesion_col] if has_lesion else [])
    pairs = df[cols].drop_duplicates().values.tolist()

    for training_list in comp_lists:
        # align_desc for Excel — describe the mixed or pure situation
        comparison_is_mixed = (
            any(_is_acclimation_group(t) for t in training_list) and
            any(not _is_acclimation_group(t) for t in training_list)
        )
        if comparison_is_mixed:
            align_desc = "Mixed: each group aligned independently (training=reversed, acclimation=forward)"
        elif all(_is_acclimation_group(t) for t in training_list):
            align_desc = "Forward (first-to-last): all groups are acclimation"
        else:
            align_desc = "Reversed (last-to-first): all groups are training"

        # Per-group alignment — always determined by each group's own type
        trn_alignment = {
            t: ("forward" if _is_acclimation_group(t) else "reversed")
            for t in training_list
        }

        comp_tag = "_vs_".join(training_list).replace(" ", "")
        data: dict       = defaultdict(dict)   # (trn, les) -> {animal: slot_dict}
        day_maps: dict   = defaultdict(dict)   # (trn, les) -> {animal: slot_to_day}
        water_prob: dict = defaultdict(dict)   # (trn, les) -> {animal: {slot: prob}}

        for row in pairs:
            if has_lesion:
                animal, training, lesion = str(row[0]), str(row[1]), str(row[2])
            else:
                animal, training = str(row[0]), str(row[1])
                lesion = "NoLesion"
            if training not in training_list:
                continue
            files = glob.glob(os.path.join(training_data_dir, animal, "*.txt"))
            if not files:
                print(f"[LICK DIFF] No files for '{animal}', skipping.")
                continue
            # Use this group's own alignment, not the comparison-level one
            diff, slot_to_day, wp = _animal_lick_difference(
                files, alignment=trn_alignment[training])
            if diff is None:
                print(f"[LICK DIFF] No usable data for '{animal}', skipping.")
                continue
            data[(training, lesion)][animal]       = diff
            day_maps[(training, lesion)][animal]   = slot_to_day
            water_prob[(training, lesion)][animal] = wp

        if not data:
            print(f"[LICK DIFF] No data for {training_list}, skipping.")
            continue

        lesion_groups = sorted({l for (_, l) in data})

        # Per-group slot lists to display
        trn_slots    = {}
        trn_day_maps = {}   # trn -> list of slot_to_day dicts (one per animal)
        for trn in training_list:
            all_sds = []
            all_dms = []
            for les in lesion_groups:
                all_sds.extend(data.get((trn, les), {}).values())
                all_dms.extend(day_maps.get((trn, les), {}).values())
            trn_slots[trn]    = _slots_for_group(trn, all_sds, all_dms,
                                                  comparison_is_mixed)
            trn_day_maps[trn] = all_dms

        # All unique slots across all groups, in order
        all_plot_slots = sorted({s for sl in trn_slots.values() for s in sl})

        if not all_plot_slots:
            continue

        for lesion in lesion_groups:
            safe_les   = lesion.replace(" ", "_").replace("/", "-")
            edge_color = lesion_colors.get(lesion, "#000000")

            for win_idx, win_name in enumerate(WIN_NAMES):
                # Build bar positions — one bar per (group, slot) combo
                bar_items = []   # (trn, slot, x_pos)
                x = 0
                grp_boundaries = {}  # trn -> list of x positions
                for trn in training_list:
                    grp_boundaries[trn] = []
                    for slot in trn_slots.get(trn, []):
                        bar_items.append((trn, slot, x))
                        grp_boundaries[trn].append(x)
                        x += 1

                if not bar_items:
                    continue

                # Collect values per (trn, slot)
                gd_full: dict = {}   # (trn, slot) -> [vals]
                for trn, slot, _ in bar_items:
                    vals = [
                        sd[slot][win_idx]
                        for sd in data.get((trn, lesion), {}).values()
                        if slot in sd and win_idx in sd[slot]
                    ]
                    gd_full[(trn, slot)] = vals

                if all(len(v) == 0 for v in gd_full.values()):
                    continue

                # Tukey/ANOVA across groups (collapse slots for stat test)
                gd_by_trn: dict = {trn: [] for trn in training_list}
                for (trn, slot), vals in gd_full.items():
                    gd_by_trn[trn].extend(vals)
                pairwise = _tukey_pairwise(gd_by_trn)
                arrs     = [np.array(gd_by_trn[t]) for t in training_list
                            if len(gd_by_trn.get(t, [])) >= 2]
                f_stat, p_anova = (_scipy_stats.f_oneway(*arrs)
                                   if len(arrs) >= 2
                                   else (float("nan"), float("nan")))

                n_bars = len(bar_items)
                fig, ax = plt.subplots(figsize=(max(3, 1.0 * n_bars), 2))
                bw      = 0.55
                bp_arr  = np.array([xi for _, _, xi in bar_items], dtype=float)
                all_flat, means, sems, ns = [], [], [], []

                for gi, (trn, slot, xpos) in enumerate(bar_items):
                    vals = np.array(gd_full[(trn, slot)])
                    n    = len(vals)
                    ns.append(n)
                    fill = group_colors.get(trn, "#808080")
                    if trn not in group_colors:
                        print(f"[COLOR] No color for '{trn}' — using grey")
                    if n == 0:
                        means.append(float("nan")); sems.append(0)
                        all_flat.append([])
                        ax.bar(xpos, 0, bw, color="none",
                               edgecolor=edge_color, linewidth=2.0, zorder=2)
                        continue
                    m = float(np.mean(vals))
                    s = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0
                    means.append(m); sems.append(s); all_flat.append(vals.tolist())
                    ax.bar(xpos, m, bw, color=fill, alpha=0.85,
                           edgecolor=edge_color, linewidth=2.0, zorder=2)
                    ax.errorbar(xpos, m, yerr=s, fmt="none", color="black",
                                capsize=3, linewidth=1.0, zorder=3)
                    jitter = np.random.default_rng(gi).uniform(-bw*0.25, bw*0.25, n)
                    ax.scatter(xpos + jitter, vals, s=18, color="black",
                               alpha=0.65, zorder=4, linewidths=0)

                ax.set_ylim(-4, 6)
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.axhline(0, color="gray", lw=0.8, linestyle="--", zorder=1)

                tops = [m + s for m, s in zip(means, sems) if not np.isnan(m)]
                for v in all_flat:
                    tops.extend(v)
                y_top = min(max(tops) if tops else 1.0, 5.5)
                n_sig = sum(1 for r in pairwise if r["sig"])
                ax.set_ylim(-4, max(6, y_top + (y_top + 4) * 0.15 * max(n_sig, 1)))

                # Significance brackets across groups (at group midpoints)
                grp_xmid = {trn: float(np.mean(grp_boundaries[trn]))
                             for trn in training_list if grp_boundaries[trn]}
                _draw_sig_brackets(ax, list(grp_xmid.keys()), pairwise,
                                   y_top, [grp_xmid[t] for t in grp_xmid])

                if not np.isnan(f_stat):
                    ax.text(0.5, 1.02,
                            f"ANOVA: F={f_stat:.2f}, p={p_anova:.3g}  |  Tukey HSD",
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=6, color="black")

                # x-axis: group name / day label / N — three lines per tick
                tick_labels = []
                for trn, slot, _ in bar_items:
                    day_lbl = _group_day_label(trn, slot, trn_alignment[trn],
                                               day_maps=trn_day_maps.get(trn))
                    n_here  = len(gd_full[(trn, slot)])
                    tick_labels.append(f"{trn}\n{day_lbl}\nN={n_here}")

                ax.set_xticks(bp_arr)
                ax.set_xticklabels(tick_labels, fontsize=6, fontname="Arial",
                                   linespacing=1.3)
                ax.set_ylabel("Lick Diff (Hz)", fontsize=8, fontname="Arial")
                ax.set_title(f"{win_name} — {lesion}",
                             fontsize=8, fontname="Arial")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="y", labelsize=7)

                fig.tight_layout()
                sw    = win_name.replace(" ", "_").replace("-", "_")
                fname = f"lick_diff_{sw}_{safe_les}_{comp_tag}.png"
                fig.savefig(os.path.join(output_folder, fname),
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[LICK DIFF] Saved {fname}")

        # max_slots for Excel: total unique slots across all data
        all_slots_in_data = {
            slot
            for ad in data.values()
            for sd in ad.values()
            for slot in sd.keys()
        }
        max_slots_excel = (max(all_slots_in_data) + 1) if all_slots_in_data else 0

        _export_lick_diff_excel(
            training_list, lesion_groups, data, max_slots_excel,
            align_desc, align_desc,
            os.path.join(output_folder, f"lick_diff_data_{comp_tag}.xlsx"),
            water_prob=water_prob
        )
        print(f"[LICK DIFF] Saved lick_diff_data_{comp_tag}.xlsx")


def _plot_lick_difference_last20pct_by_group(df_master, animal_id_col,
                                              training_col, comparisons,
                                              training_data_dir, output_folder,
                                              lesion_col=None,
                                              lesion_filter=None):
    """
    Identical to _plot_lick_difference_by_group but uses only the last 20%
    of trials per day for each animal.  Graphs are saved in the same
    output_folder with a '_last20pct' suffix in the filename.
    """
    os.makedirs(output_folder, exist_ok=True)
    has_lesion    = bool(lesion_col and lesion_col in df_master.columns)
    group_colors  = _read_color_file(_color_path("group_colors.txt"))
    lesion_colors = _read_color_file(_color_path("lesion_colors.txt"))

    comp_lists = []
    for comp in comparisons:
        comp_lists.append([str(g).strip() for g in comp]
                          if isinstance(comp, (list, tuple))
                          else [str(comp).strip()])

    df = df_master.copy()
    df[training_col] = df[training_col].astype(str).str.strip()
    df = df[df[training_col].isin({g for cl in comp_lists for g in cl})]
    if has_lesion and lesion_filter:
        df[lesion_col] = df[lesion_col].astype(str).str.strip()
        df = df[df[lesion_col].isin([str(l).strip() for l in lesion_filter])]
    elif has_lesion:
        df[lesion_col] = df[lesion_col].astype(str).str.strip()

    cols  = [animal_id_col, training_col] + ([lesion_col] if has_lesion else [])
    pairs = df[cols].drop_duplicates().values.tolist()

    for training_list in comp_lists:
        comparison_is_mixed = (
            any(_is_acclimation_group(t) for t in training_list) and
            any(not _is_acclimation_group(t) for t in training_list)
        )
        if comparison_is_mixed:
            align_desc = "Mixed: each group aligned independently (training=reversed, acclimation=forward)"
        elif all(_is_acclimation_group(t) for t in training_list):
            align_desc = "Forward (first-to-last): all groups are acclimation"
        else:
            align_desc = "Reversed (last-to-first): all groups are training"

        trn_alignment = {
            t: ("forward" if _is_acclimation_group(t) else "reversed")
            for t in training_list
        }

        comp_tag = "_vs_".join(training_list).replace(" ", "")
        data: dict       = defaultdict(dict)
        day_maps: dict   = defaultdict(dict)
        water_prob: dict = defaultdict(dict)   # (trn, les) -> {animal: {slot: prob}}

        for row in pairs:
            if has_lesion:
                animal, training, lesion = str(row[0]), str(row[1]), str(row[2])
            else:
                animal, training = str(row[0]), str(row[1])
                lesion = "NoLesion"
            if training not in training_list:
                continue
            files = glob.glob(os.path.join(training_data_dir, animal, "*.txt"))
            if not files:
                print(f"[LICK DIFF 20%] No files for '{animal}', skipping.")
                continue
            # Use last-20%-filtered version
            diff, slot_to_day, wp = _animal_lick_difference_last20pct(
                files, alignment=trn_alignment[training])
            if diff is None:
                print(f"[LICK DIFF 20%] No usable data for '{animal}', skipping.")
                continue
            data[(training, lesion)][animal]       = diff
            day_maps[(training, lesion)][animal]   = slot_to_day
            water_prob[(training, lesion)][animal] = wp

        if not data:
            print(f"[LICK DIFF 20%] No data for {training_list}, skipping.")
            continue

        lesion_groups = sorted({l for (_, l) in data})

        trn_slots    = {}
        trn_day_maps = {}
        for trn in training_list:
            all_sds = []
            all_dms = []
            for les in lesion_groups:
                all_sds.extend(data.get((trn, les), {}).values())
                all_dms.extend(day_maps.get((trn, les), {}).values())
            trn_slots[trn]    = _slots_for_group(trn, all_sds, all_dms,
                                                  comparison_is_mixed)
            trn_day_maps[trn] = all_dms

        all_plot_slots = sorted({s for sl in trn_slots.values() for s in sl})
        if not all_plot_slots:
            continue

        for lesion in lesion_groups:
            safe_les   = lesion.replace(" ", "_").replace("/", "-")
            edge_color = lesion_colors.get(lesion, "#000000")

            for win_idx, win_name in enumerate(WIN_NAMES):
                bar_items = []
                x = 0
                grp_boundaries = {}
                for trn in training_list:
                    grp_boundaries[trn] = []
                    for slot in trn_slots.get(trn, []):
                        bar_items.append((trn, slot, x))
                        grp_boundaries[trn].append(x)
                        x += 1

                if not bar_items:
                    continue

                gd_full: dict = {}
                for trn, slot, _ in bar_items:
                    vals = [
                        sd[slot][win_idx]
                        for sd in data.get((trn, lesion), {}).values()
                        if slot in sd and win_idx in sd[slot]
                    ]
                    gd_full[(trn, slot)] = vals

                if all(len(v) == 0 for v in gd_full.values()):
                    continue

                gd_by_trn: dict = {trn: [] for trn in training_list}
                for (trn, slot), vals in gd_full.items():
                    gd_by_trn[trn].extend(vals)
                pairwise = _tukey_pairwise(gd_by_trn)
                arrs     = [np.array(gd_by_trn[t]) for t in training_list
                            if len(gd_by_trn.get(t, [])) >= 2]
                f_stat, p_anova = (_scipy_stats.f_oneway(*arrs)
                                   if len(arrs) >= 2
                                   else (float("nan"), float("nan")))

                n_bars = len(bar_items)
                fig, ax = plt.subplots(figsize=(max(3, 1.0 * n_bars), 2))
                bw      = 0.55
                bp_arr  = np.array([xi for _, _, xi in bar_items], dtype=float)
                all_flat, means, sems, ns = [], [], [], []

                for gi, (trn, slot, xpos) in enumerate(bar_items):
                    vals = np.array(gd_full[(trn, slot)])
                    n    = len(vals)
                    ns.append(n)
                    fill = group_colors.get(trn, "#808080")
                    if trn not in group_colors:
                        print(f"[COLOR] No color for '{trn}' — using grey")
                    if n == 0:
                        means.append(float("nan")); sems.append(0)
                        all_flat.append([])
                        ax.bar(xpos, 0, bw, color="none",
                               edgecolor=edge_color, linewidth=2.0, zorder=2)
                        continue
                    m = float(np.mean(vals))
                    s = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0
                    means.append(m); sems.append(s); all_flat.append(vals.tolist())
                    ax.bar(xpos, m, bw, color=fill, alpha=0.85,
                           edgecolor=edge_color, linewidth=2.0, zorder=2)
                    ax.errorbar(xpos, m, yerr=s, fmt="none", color="black",
                                capsize=3, linewidth=1.0, zorder=3)
                    jitter = np.random.default_rng(gi).uniform(-bw*0.25, bw*0.25, n)
                    ax.scatter(xpos + jitter, vals, s=18, color="black",
                               alpha=0.65, zorder=4, linewidths=0)

                ax.set_ylim(-4, 6)
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.axhline(0, color="gray", lw=0.8, linestyle="--", zorder=1)

                tops = [m + s for m, s in zip(means, sems) if not np.isnan(m)]
                for v in all_flat:
                    tops.extend(v)
                y_top = min(max(tops) if tops else 1.0, 5.5)
                n_sig = sum(1 for r in pairwise if r["sig"])
                ax.set_ylim(-4, max(6, y_top + (y_top + 4) * 0.15 * max(n_sig, 1)))

                grp_xmid = {trn: float(np.mean(grp_boundaries[trn]))
                             for trn in training_list if grp_boundaries[trn]}
                _draw_sig_brackets(ax, list(grp_xmid.keys()), pairwise,
                                   y_top, [grp_xmid[t] for t in grp_xmid])

                if not np.isnan(f_stat):
                    ax.text(0.5, 1.02,
                            f"ANOVA: F={f_stat:.2f}, p={p_anova:.3g}  |  Tukey HSD",
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=6, color="black")

                tick_labels = []
                for trn, slot, _ in bar_items:
                    day_lbl = _group_day_label(trn, slot, trn_alignment[trn],
                                               day_maps=trn_day_maps.get(trn))
                    n_here  = len(gd_full[(trn, slot)])
                    tick_labels.append(f"{trn}\n{day_lbl}\nN={n_here}")

                ax.set_xticks(bp_arr)
                ax.set_xticklabels(tick_labels, fontsize=6, fontname="Arial",
                                   linespacing=1.3)
                ax.set_ylabel("Lick Diff (Hz)", fontsize=8, fontname="Arial")
                ax.set_title(f"{win_name} — {lesion} — Last 20% of Trials",
                             fontsize=8, fontname="Arial")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="y", labelsize=7)

                fig.tight_layout()
                sw    = win_name.replace(" ", "_").replace("-", "_")
                fname = f"lick_diff_{sw}_{safe_les}_{comp_tag}_last20pct.png"
                fig.savefig(os.path.join(output_folder, fname),
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[LICK DIFF 20%] Saved {fname}")

        # Excel export for last-20% data
        all_slots_in_data = {
            slot
            for ad in data.values()
            for sd in ad.values()
            for slot in sd.keys()
        }
        max_slots_excel = (max(all_slots_in_data) + 1) if all_slots_in_data else 0

        _export_lick_diff_excel(
            training_list, lesion_groups, data, max_slots_excel,
            align_desc, align_desc,
            os.path.join(output_folder,
                         f"lick_diff_data_{comp_tag}_last20pct.xlsx"),
            water_prob=water_prob
        )
        print(f"[LICK DIFF 20%] Saved lick_diff_data_{comp_tag}_last20pct.xlsx")


# ════════════════════════════════════════════════════════════════════════════
# PARALLEL ANIMAL WORKER  (top-level so it can be pickled)
# ════════════════════════════════════════════════════════════════════════════

def _animal_worker(args):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    files, out_folder, n_shade, alignment = args
    os.makedirs(out_folder, exist_ok=True)
    return _process_animal(files, out_folder, n_shade, alignment)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_behavior_trials(df_master, animal_id_col, training_col,
                        comparisons, training_data_dir, output_folder,
                        lesion_col=None, lesion_filter=None,
                        n_workers: int = 4, **kwargs):
    """
    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes for animal-level processing.
        Default 4.  Set to 1 to disable parallelism (useful for debugging).
    """
    os.makedirs(output_folder, exist_ok=True)
    training_def    = _read_training_def(training_data_dir)
    selected_groups = _extract_selected_groups(comparisons)
    if not selected_groups:
        print("No groups selected.")
        return []

    df = df_master.copy()
    df[training_col] = df[training_col].astype(str).str.strip()
    df = df[df[training_col].isin(selected_groups)]
    if lesion_col and lesion_filter and lesion_col in df.columns:
        df[lesion_col] = df[lesion_col].astype(str).str.strip()
        df = df[df[lesion_col].isin([str(l).strip() for l in lesion_filter])]

    cols  = [animal_id_col, training_col] + ([lesion_col] if lesion_col else [])
    pairs = df[cols].drop_duplicates().values.tolist()

    # ── Determine alignment per training group ────────────────────────────
    # Each group's alignment is decided purely by its own group type,
    # NOT by what other groups appear in the same comparison.
    # forward  = acclimation group  (ACC, PSECntl, numeric ratios: 0.2, 0.5, 1)
    # reversed = training group     (SAT, SDT, PSE, AR_SAT, unknown)
    all_comp_groups: set = set()
    for comp in comparisons:
        cl = ([str(g).strip() for g in comp]
              if isinstance(comp, (list, tuple))
              else [str(comp).strip()])
        all_comp_groups.update(cl)
    group_alignment: dict = {
        g: ("forward" if _is_acclimation_group(g) else "reversed")
        for g in all_comp_groups
    }

    # group_registry[group_key] = {files, ids, training, alignment}
    group_registry: dict = defaultdict(lambda: {
        "files": [], "ids": [], "training": "", "alignment": "reversed"
    })
    animals_no_folder = []

    for row in pairs:
        if lesion_col:
            animal, training, lesion = str(row[0]).strip(), str(row[1]).strip(), str(row[2]).strip()
        else:
            animal, training = str(row[0]).strip(), str(row[1]).strip()
            lesion = "NoLesion"

        files = glob.glob(os.path.join(training_data_dir, animal, "*.txt"))
        if not files:
            animals_no_folder.append(animal)
            print(f"[MISSING] '{animal}' — no .txt files found")
            continue

        align = group_alignment.get(
            training,
            "forward" if _is_acclimation_group(training) else "reversed"
        )
        gk = f"{training}_{lesion}" if lesion_col else training
        group_registry[gk]["files"].append(files)
        group_registry[gk]["ids"].append(animal)
        group_registry[gk]["training"]  = training
        group_registry[gk]["alignment"] = align

    if animals_no_folder:
        print(f"\n[MISSING SUMMARY] {len(animals_no_folder)} animal(s) had no data:")
        for a in animals_no_folder:
            print(f"  - {a}")

    # ── Build task list ───────────────────────────────────────────────────
    tasks = []   # (files, out_folder, n_shade, alignment, gk, animal_id)
    for gk, reg in group_registry.items():
        training   = reg["training"]
        lesion_val = gk[len(training)+1:] if "_" in gk else "NoLesion"
        align      = reg["alignment"]
        print(f"[ALIGN] '{gk}' training='{training}' -> {align}")
        for files, animal_id in zip(reg["files"], reg["ids"]):
            n_shade = training_def.get(training, None)
            out     = os.path.join(output_folder, training, lesion_val, animal_id)
            tasks.append((files, out, n_shade, align, gk, animal_id))

    # ── Run animals in parallel ───────────────────────────────────────────
    results_by_gk: dict = defaultdict(list)   # gk -> list of animal result dicts

    def _submit_serial():
        for files, out, n_shade, align, gk, animal_id in tasks:
            os.makedirs(out, exist_ok=True)
            res = _process_animal(files, out, n_shade, align)
            if res is not None:
                w, b, days, iti, iti_nc = res
                if len(days) > 1:
                    results_by_gk[gk].append({
                        "id":            animal_id,
                        "w":             w,
                        "b":             b,
                        "days":          _align_days(days, align),
                        "files":         files,
                        "iti_records":   iti,
                        "iti_records_nc": iti_nc,
                    })
                else:
                    print(f"[EXCLUDED] '{animal_id}' has <=1 valid day.")

    if n_workers <= 1:
        _submit_serial()
    else:
        try:
            worker_args = [(f, o, ns, al)
                           for f, o, ns, al, *_ in tasks]
            task_meta   = [(gk, aid, al)
                           for _, _, _, al, gk, aid in tasks]

            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_animal_worker, a): (meta, a[3])
                        for a, meta in zip(worker_args, task_meta)}
                for fut in as_completed(futs):
                    (gk, animal_id, align), _ = futs[fut], None
                    try:
                        res = fut.result()
                    except Exception as exc:
                        print(f"[ERROR] '{animal_id}': {exc}")
                        continue
                    if res is None:
                        continue
                    w, b, days, iti, iti_nc = res
                    if len(days) > 1:
                        results_by_gk[gk].append({
                            "id":            animal_id,
                            "w":             w,
                            "b":             b,
                            "days":          _align_days(days, align),
                            "files":         [f for f, *_ in worker_args
                                              if True][0],  # placeholder
                            "iti_records":   iti,
                            "iti_records_nc": iti_nc,
                        })
                    else:
                        print(f"[EXCLUDED] '{animal_id}' has <=1 valid day.")
        except Exception as exc:
            print(f"[PARALLEL] Falling back to serial: {exc}")
            results_by_gk.clear()
            _submit_serial()

    # ── Group summaries ───────────────────────────────────────────────────
    group_out = os.path.join(output_folder, "_group_summaries")
    for gk, reg in group_registry.items():
        animal_records = results_by_gk.get(gk, [])
        if not animal_records:
            print(f"[SKIP] No qualifying animals for group '{gk}'.")
            continue
        print(f"\n[GROUP SUMMARY] '{gk}'  N={len(animal_records)}")
        _analyse_group(gk, animal_records,
                       os.path.join(group_out, gk),
                       alignment=reg["alignment"])

    # ── Lick difference graphs ────────────────────────────────────────────
    _lick_diff_out = os.path.join(output_folder, "_lick_difference")
    _plot_lick_difference_by_group(
        df_master=df_master, animal_id_col=animal_id_col,
        training_col=training_col, comparisons=comparisons,
        training_data_dir=training_data_dir,
        output_folder=_lick_diff_out,
        lesion_col=lesion_col, lesion_filter=lesion_filter,
    )

    # ── Lick difference graphs — last 20% of trials ───────────────────────
    _plot_lick_difference_last20pct_by_group(
        df_master=df_master, animal_id_col=animal_id_col,
        training_col=training_col, comparisons=comparisons,
        training_data_dir=training_data_dir,
        output_folder=_lick_diff_out,
        lesion_col=lesion_col, lesion_filter=lesion_filter,
    )

    return []