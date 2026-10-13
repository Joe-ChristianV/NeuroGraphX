# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:32:20 2026

@author: JoeCh
"""

README

BehaviorTrialsModule
# BehaviorTrialsModule — Methods Summary

## Data Loading & Day Segmentation
Raw `.txt` files (comma-separated: timestamp, sensor, lick, phase, pre_puff_delay) are loaded per animal and concatenated with continuous timestamps. Days are defined by 86,400-second windows from the first timestamp; a day is only considered valid if it spans ≥10 hours of recording. The 0-based calendar day index is derived as `(timestamp - t0) // 86400`.

## Trial Segmentation
Trials are detected by phase transitions: a new trial begins wherever phase transitions out of WAIT3 (7) into either WAIT1 (3) or BLANK (9). Each trial is classified as **water** (contains a WAIT2→WAIT3 transition) or **blank** (no water delivery). The trial reference time ("trial start") is the timestamp where `pre_puff_delay` crosses from non-zero to zero.

## Lick Rate Calculation
For each trial, lick events (lick == 2) are collected and binned into four windows relative to trial start:
- **Initiation**: 0.0–0.2 s
- **Anticipation**: 0.8–1.0 s
- **Reward**: 1.0–1.2 s
- **Post-Reward**: 1.6–1.8 s

Rate is lick count divided by window width (Hz).

## Alignment
Animals are aligned differently depending on group type, because training groups have a fixed end but variable start:
- **Acclimation groups** (ACC, PSECntl, numeric ratios like 0.2/0.5/0.8): days ordered first→last (forward). Panels labelled by real calendar day number.
- **Training groups** (SAT, SDT, PSE, AR_SAT): days ordered last→first (reversed). This anchors all animals at their final training day regardless of how many days they ran. Panels still labelled with real calendar day numbers so you can see which actual day each panel represents.

## Per-Animal Outputs
For each animal: a raster heatmap (lick density over time, 1200 dpi), a frequency-by-day line graph (mean ± SEM water vs. no-water per window, per day), and an ITI analysis. A "last 20% of trials per day" version of the raster and freq graph is also produced.

## Group Summaries
Animals within a group are aggregated by **slot** (position in their aligned day order). For each slot, the mean-per-animal is computed first, then the group mean ± SEM is taken across animals. Panel labels use the median real calendar day across animals at that slot. Mann-Whitney U tests compare water vs. blank within each window.

## Lick Difference Graphs
Per animal, the lick difference is `mean(water Hz) - mean(blank Hz)` per window per day. For the bar chart:
- **Acclimation groups**: always plot Day 2. Exception — if >75% of animals have a Day 6 timepoint, Day 6 is used instead (Day 7 as secondary fallback).
- **Training groups**: always plot the last day (slot 0 in reversed alignment).

One-way ANOVA with Tukey HSD post-hoc is run across groups. Brackets mark significant pairs (p < 0.05).

## Group Color & Training Definition Files
`group_colors.txt` in the NeuroGraphX root maps group names to hex or named colors (colon-delimited). `TrainingDef.txt` in the data directory maps group names to an integer shading count used for raster highlighting (comma-delimited).