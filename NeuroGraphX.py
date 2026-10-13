# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 21:36:44 2025

@author: JoeCh w/ AI Assistance
"""

# NeuroGraphX.py
# Version: 0.0.25 (patched)
# Integrated with analysis_modules.BarGraphModule

import sys
import os
import pandas as pd

from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QComboBox, QLineEdit, QListWidget, QAbstractItemView,
    QMessageBox, QHBoxLayout, QVBoxLayout, QInputDialog      # <-- add QInputDialog
)
from PyQt5.QtCore import Qt

# ===================== ANALYSIS MODULE IMPORTS =====================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

run_bargraph_analysis = None
CD_Analysis = None
run_rdt_analysis = None
find_optimal_cutoff = None
run_trace_averager = None
run_figure_assembly = None
run_behavior_trials = None
run_correlation  = None
run_correlation2 = None

_RDT_IMPORT_ERROR = None

# ---- Core imports ----
try:
    from analysis_modules import run_bargraph_analysis, CD_Analysis
except Exception as e:
    print(f"[IMPORT ERROR] BarGraph/CD_Analysis: {e}")

# ---- Depth correlation ----
try:
    from analysis_modules import run_rdt_analysis, find_optimal_cutoff
except Exception as e:
    _RDT_IMPORT_ERROR = str(e)
    print(f"[IMPORT ERROR] DepthCorrelationModule: {e}")

# ---- Trace averager ----
try:
    from analysis_modules import run_trace_averager
except Exception as e:
    print(f"[IMPORT ERROR] Raw_Trace_Module: {e}")

# ---- Figure assembly ----
try:
    from analysis_modules import run_figure_assembly
except Exception as e:
    print(f"[IMPORT ERROR] FigureAssemblyModule: {e}")

# ---- Behavior trials ----
try:
    from analysis_modules import run_behavior_trials
except Exception as e:
    print(f"[IMPORT ERROR] BehaviorTrialsModule: {e}")

# ---- Correlation ----
try:
    from analysis_modules import run_correlation
except Exception as e:
    print(f"[IMPORT ERROR] CorrelationModule: {e}")

# ---- Correlation2 ----
try:
    from analysis_modules import run_correlation2
except Exception as e:
    print(f"[IMPORT ERROR] Correlations2Module: {e}")
    
class NeuroGraphX(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroGraphX v0.0.25 (patched)")
        self.setGeometry(100, 100, 1300, 700)
        self.setFixedSize(1300, 700)

        # ---------------- THEME ----------------
        self.theme = {
            "bg_color": "#2E3440",
            "panel_color": "#3B4252",
            "font_color": "#D8DEE9",
            "button_color": "#5E81AC",
            "button_text": "#ECEFF4",
            "dropdown_color": "#4C566A",
            "dropdown_text": "#ECEFF4",
            "lineedit_bg": "#434C5E",
            "lineedit_text": "#ECEFF4",
            "table_bg": "#FFFFFF",
            "table_text": "#000000",
            "highlight": "#A3BE8C"
        }
        self.setStyleSheet(f"background-color: {self.theme['bg_color']}; color: {self.theme['font_color']};")

        # ---------------- DATA ----------------
        self.df = None
        self.comparisons = []
        self.current_depth_value = None
        self.loaded_file_path = None

        # ---------------- LEFT PANEL ----------------
        self.status_label = QLabel("Controls", self)
        self.status_label.setGeometry(20, 20, 200, 25)

        self.load_button = QPushButton("Load File", self)
        self.load_button.setGeometry(20, 50, 160, 30)
        self.load_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.load_button.clicked.connect(self.load_file)

        self.run_module_button = QPushButton("Run Bar Graph Module", self)
        self.run_module_button.setGeometry(20, 90, 160, 30)
        self.run_module_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.run_module_button.clicked.connect(self.launch_bargraph_module)

        self.clear_selected_button = QPushButton("Clear Comparisons", self)
        self.clear_selected_button.setGeometry(20, 130, 160, 30)
        self.clear_selected_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.clear_selected_button.clicked.connect(self.clear_selected_comparisons)

        # ---------------- TOP DROPDOWNS ----------------
        self.measurement_label = QLabel("Measurement", self)
        self.measurement_label.setGeometry(220, 20, 170, 20)
        self.measurement_combo = QComboBox(self)
        self.measurement_combo.setGeometry(220, 40, 170, 25)

        self.filename_label = QLabel("Data Filename", self)
        self.filename_label.setGeometry(220, 70, 170, 20)
        self.filename_combo = QComboBox(self)
        self.filename_combo.setGeometry(220, 90, 170, 25)

        self.depth_label = QLabel("Depth", self)
        self.depth_label.setGeometry(410, 20, 170, 20)
        self.depth_combo = QComboBox(self)
        self.depth_combo.setGeometry(410, 40, 170, 25)
        self.depth_combo.addItem("None")
        self.depth_combo.currentTextChanged.connect(self.on_depth_column_changed)

        self.depth_lineedit = QLineEdit(self)
        self.depth_lineedit.setGeometry(410, 70, 130, 25)
        self.depth_lineedit.setPlaceholderText("Enter number (threshold)")
        self.depth_lineedit.returnPressed.connect(self.store_depth_value)
        self.depth_lineedit.setEnabled(False)

        self.depth_button = QPushButton("OK", self)
        self.depth_button.setGeometry(545, 70, 40, 25)
        self.depth_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.depth_button.clicked.connect(self.store_depth_value)
        self.depth_button.setEnabled(False)

        self.depth_display = QLabel("", self)
        self.depth_display.setGeometry(410, 100, 170, 15)
        small_font = QFont()
        small_font.setPointSize(8)
        self.depth_display.setFont(small_font)
        self.depth_display.setStyleSheet("color: #AAA;")

        self.training_label = QLabel("Training", self)
        self.training_label.setGeometry(600, 20, 170, 20)
        self.training_combo = QComboBox(self)
        self.training_combo.setGeometry(600, 40, 170, 25)
        self.training_combo.currentTextChanged.connect(self.on_training_column_changed)

        self.lesion_label = QLabel("Lesion", self)
        self.lesion_label.setGeometry(600, 70, 170, 20)
        self.lesion_combo = QComboBox(self)
        self.lesion_combo.setGeometry(600, 90, 170, 25)
        self.lesion_combo.addItem("None")

        self.animal_label = QLabel("Animal ID", self)
        self.animal_label.setGeometry(790, 20, 170, 20)
        self.animal_combo = QComboBox(self)
        self.animal_combo.setGeometry(790, 40, 170, 25)

        self.water_ml_label = QLabel("Water (mL)", self)
        self.water_ml_label.setGeometry(790, 70, 170, 20)
        self.water_ml_combo = QComboBox(self)
        self.water_ml_combo.setGeometry(790, 90, 170, 25)
        self.water_ml_combo.addItem("None")

        # ---------------- COMPARISONS ----------------
        self.saved_label = QLabel("Saved Comparisons", self)
        self.saved_label.setGeometry(220, 140, 360, 20)
        self.saved_comparisons_list = QListWidget(self)
        self.saved_comparisons_list.setGeometry(220, 160, 360, 145)
        self.saved_comparisons_list.setSelectionMode(QAbstractItemView.MultiSelection)

        self.avail_label = QLabel("Available Groups", self)
        self.avail_label.setGeometry(600, 140, 360, 20)
        self.comparison_list = QListWidget(self)
        self.comparison_list.setGeometry(600, 160, 360, 120)
        self.comparison_list.setSelectionMode(QAbstractItemView.MultiSelection)

        self.add_comparison_button = QPushButton("Add Comparison", self)
        self.add_comparison_button.setGeometry(600, 285, 360, 25)
        self.add_comparison_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.add_comparison_button.clicked.connect(self.add_comparison)

        self.comparison_list.setEnabled(False)
        self.add_comparison_button.setEnabled(False)

        # ---------------- CD ANALYSIS / EXAMPLE TRACE ----------------
        self.cd_button = QPushButton("CD Analysis", self)
        self.cd_button.setGeometry(20, 180, 160, 30)
        self.cd_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.cd_button.clicked.connect(self.cd_analysis)

        self.cd_entry = QLineEdit(self)
        self.cd_entry.setGeometry(20, 220, 110, 25)
        self.cd_entry.setPlaceholderText("Enter number")
        self.cd_entry.returnPressed.connect(self.cd_analysis)  # Trigger CD directly

        self.cd_enter_button = QPushButton("Enter", self)
        self.cd_enter_button.setGeometry(135, 220, 45, 25)
        self.cd_enter_button.clicked.connect(self.cd_analysis)

        # -------- Example Trace --------
        self.raw_trace_button = QPushButton("Example Trace", self)
        self.raw_trace_button.setGeometry(20, 260, 160, 30)
        self.raw_trace_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.raw_trace_button.clicked.connect(self.raw_trace)
        
        # -------- DEPTH CORRELATION --------
        self.depth_corr_button = QPushButton("Depth Correlation", self)
        self.depth_corr_button.setGeometry(20, 300, 160, 30)
        self.depth_corr_button.setStyleSheet(f"background-color: {self.theme['button_color']}; color: {self.theme['button_text']};")
        self.depth_corr_button.clicked.connect(self.depth_correlation)
        
        # -------- Figure Maker -----------
        self.make_figures_button = QPushButton("Make Figures", self)
        self.make_figures_button.setGeometry(20, 350, 160, 50)
        self.make_figures_button.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        self.make_figures_button.clicked.connect(self.make_figures)
             
        # ---------------- Right Pannel ----------------
        self.behavior_label = QLabel("Behavior", self)
        self.behavior_label.setGeometry(1000, 20, 160, 20)
        
        self.Behavior_Trials_button = QPushButton("Trials", self)
        self.Behavior_Trials_button.setGeometry(1000, 40, 160, 20)
        self.Behavior_Trials_button.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        self.Behavior_Trials_button.clicked.connect(self.behavior_trials)

        self.Behavior_Corr_button = QPushButton("Correlations", self)
        self.Behavior_Corr_button.setGeometry(1000, 60, 160, 20)
        self.Behavior_Corr_button.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        self.Behavior_Corr_button.clicked.connect(self.behavior_correlations)

        self.Behavior_Corr2_button = QPushButton("Correlations 2", self)
        self.Behavior_Corr2_button.setGeometry(1000, 82, 160, 20)
        self.Behavior_Corr2_button.setStyleSheet(
            f"QPushButton {{ background-color: {self.theme['button_color']}; color: {self.theme['button_text']}; }}"
            f"QToolTip {{ color: #5b86d4; background-colo r: #070708; }}"
        )
        self.Behavior_Corr2_button.setToolTip(
            "<b>Correlations 2</b> — Simplified lick-diff correlations<br><br>"
            "Produces 4-panel scatter plots (one panel per time window:<br>"
            "<i>Initiation, Anticipation, Reward, Post-Reward</i>) for:<br><br>"
            "&nbsp;&nbsp;• <b>Lick Diff vs Reward Size</b><br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;(water − blank lick Hz vs µL/trial)<br>"
            "&nbsp;&nbsp;• <b>Lick Diff vs Expected Value</b><br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;(water − blank lick Hz vs mL × P(Water))<br>"
            "&nbsp;&nbsp;• <b>Lick Diff vs qEPSC</b> — coloured by comparison group<br>"
            "&nbsp;&nbsp;• <b>Lick Diff vs qEPSC</b> — all animals together<br><br>"
            "All graphs are split by <b>lesion</b> and <b>cortical layer</b>."
        )

        self.Behavior_Corr2_button.clicked.connect(self.behavior_correlations2)                 
        
        # ---------------- PREVIEW TABLE ----------------
        self.table = QTableWidget(self)
        self.table.setGeometry(220, 320, 740, 350)
        self.table.setStyleSheet("background-color: white; color: black;")

        # ---------------- PLOT CONTAINER ----------------
        self.plot_container = QWidget(self)
        self.plot_container.setGeometry(220, 320, 740, 350)
        self.plot_container.hide()
        self.plot_layout = QHBoxLayout(self.plot_container)
        # create initial canvases
        self.plot_left = FigureCanvas(Figure())
        self.plot_right = FigureCanvas(Figure())
        self.plot_layout.addWidget(self.plot_left)
        self.plot_layout.addWidget(self.plot_right)
        
        # ── NAVIGATOR WIDGETS (depth correlation graph cycling) ────────────
        self._rdt_results   = []   # list of result dicts from run_rdt_analysis
        self._rdt_page_idx  = 0    # which result we're currently viewing
        self._rdt_fig_type  = 0    # 0=scatter+bar, 1=tree, 2=bar only
        
        self.nav_widget = QWidget(self)
        self.nav_widget.setGeometry(220, 675, 740, 25)
        self.nav_widget.hide()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        self.nav_prev_btn = QPushButton("◀ Prev", self.nav_widget)
        self.nav_prev_btn.clicked.connect(self._nav_prev)
        self.nav_prev_btn.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        
        self.nav_fig_btn = QPushButton("Scatter ↔ Tree ↔ Bar", self.nav_widget)
        self.nav_fig_btn.clicked.connect(self._nav_cycle_fig)
        self.nav_fig_btn.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        
        self.nav_next_btn = QPushButton("Next ▶", self.nav_widget)
        self.nav_next_btn.clicked.connect(self._nav_next)
        self.nav_next_btn.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        
        self.nav_label = QLabel("", self.nav_widget)
        self.nav_label.setStyleSheet(f"color: {self.theme['font_color']};")
        
        self.nav_back_btn = QPushButton("Back to Table", self.nav_widget)
        self.nav_back_btn.clicked.connect(self._nav_back_to_table)
        self.nav_back_btn.setStyleSheet(
            f"background-color: {self.theme['button_color']}; "
            f"color: {self.theme['button_text']};")
        
        nav_layout.addWidget(self.nav_prev_btn)
        nav_layout.addWidget(self.nav_fig_btn)
        nav_layout.addWidget(self.nav_next_btn)
        nav_layout.addWidget(self.nav_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self.nav_back_btn)
         
    # ---------------- FILE LOADING ----------------
    def load_file(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.csv *.xlsx *.xls)")
        if not path:
            return
        self.loaded_file_path = path
        try:
            if path.lower().endswith(".csv"):
                self.df = pd.read_csv(path)
            else:
                self.df = pd.read_excel(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not load file:\n{e}")
            return

        filename = os.path.basename(path)
        self.status_label.setText(f"Loaded: {filename}")

        cols = list(map(str, self.df.columns))
        # Populate combos (use "Select" for most, "None" for depth to preserve existing behavior)
        for combo in [self.measurement_combo, self.filename_combo, self.training_combo, self.animal_combo, self.lesion_combo]:
            combo.clear()
            combo.addItem("Select")
            combo.addItems(cols)

        self.water_ml_combo.clear()
        self.water_ml_combo.addItem("None")
        self.water_ml_combo.addItems(cols)

        self.depth_combo.blockSignals(True)
        self.depth_combo.clear()
        self.depth_combo.addItem("None")
        self.depth_combo.addItems(cols)
        self.depth_combo.blockSignals(False)
        self.depth_lineedit.setEnabled(False)
        self.depth_button.setEnabled(False)
        self.depth_display.setText("")
        self.current_depth_value = None

        self.comparison_list.clear()
        self.comparison_list.setEnabled(False)
        self.add_comparison_button.setEnabled(False)
        self.saved_comparisons_list.clear()
        self.comparisons = []

        self.show_preview()
        self.auto_select_columns()

    # ---------------- PREVIEW ----------------
    def show_preview(self, max_rows=20):
        if self.df is None:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        self.plot_container.hide()
        self.table.show()
        cols = list(map(str, self.df.columns))
        rows = min(len(self.df), max_rows)
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(rows)
        self.table.setHorizontalHeaderLabels(cols)
        for i in range(rows):
            for j, c in enumerate(cols):
                self.table.setItem(i, j, QTableWidgetItem(str(self.df.iat[i, j])))
    
    # ---------------- AUTO COLUMN DETECTION ----------------
    def auto_select_columns(self):
        if self.df is None:
            return
    
        cols = list(map(str, self.df.columns))
    
        def find_best_match(keywords):
            scores = {}
            for col in cols:
                col_lower = col.lower()
                score = sum(1 for k in keywords if k in col_lower)
                scores[col] = score
            best = max(scores, key=scores.get)
            return best if scores[best] > 0 else None
    
        measurement_keywords = ["qEPSC", "ipsc", "amplitude", "epsc", "measurement", "current"]
        filename_keywords = ["file", "filename", "recording", "trace"]
        training_keywords = ["group", "training", "condition", "treatment"]
        depth_keywords = ["depth", "distance"]
        lesion_keywords = ["lesion", "damage"]
        animal_keywords = ["animal", "mouse", "id", "subject"]
        water_ml_keywords = ["water", "ml", "volume"]
    
        measurement_col = find_best_match(measurement_keywords)
        filename_col = find_best_match(filename_keywords)
        training_col = find_best_match(training_keywords)
        depth_col = find_best_match(depth_keywords)
        lesion_col = find_best_match(lesion_keywords)
        animal_col = find_best_match(animal_keywords)
        water_ml_col = find_best_match(water_ml_keywords)
    
        def set_combo(combo, value):
            if value and value in [combo.itemText(i) for i in range(combo.count())]:
                combo.setCurrentText(value)
    
        set_combo(self.measurement_combo, measurement_col)
        set_combo(self.filename_combo, filename_col)
        set_combo(self.training_combo, training_col)
        set_combo(self.depth_combo, depth_col)
        set_combo(self.lesion_combo, lesion_col)
        set_combo(self.animal_combo, animal_col)
        set_combo(self.water_ml_combo, water_ml_col)
    
        if training_col:
            self.on_training_column_changed(training_col)                
    
    # ---------------- DEPTH ----------------
    def on_depth_column_changed(self, text):
        enabled = text not in ["None", "", "Select"]
        self.depth_lineedit.setEnabled(enabled)
        self.depth_button.setEnabled(enabled)
        if not enabled:
            self.depth_display.setText("")
            self.depth_lineedit.clear()
            self.current_depth_value = None

    def store_depth_value(self):
        txt = self.depth_lineedit.text().strip()
        if txt == "":
            QMessageBox.information(self, "Depth", "Enter a number.")
            return
        try:
            val = float(txt)
        except ValueError:
            QMessageBox.warning(self, "Depth", "Depth must be a number.")
            return
        self.current_depth_value = val
        self.depth_display.setText(f"Depth entered: {val}")

    # ---------------- TRAINING / COMPARISONS ----------------
    def on_training_column_changed(self, text):
        if self.df is None or text in ["Select", ""] or text not in self.df.columns:
            self.comparison_list.clear()
            self.comparison_list.setEnabled(False)
            self.add_comparison_button.setEnabled(False)
            return
        unique = sorted(self.df[text].dropna().astype(str).unique())
        self.comparison_list.clear()
        self.comparison_list.addItems(unique)
        self.comparison_list.setEnabled(True)
        self.add_comparison_button.setEnabled(True)

    def add_comparison(self):
        selected = [it.text() for it in self.comparison_list.selectedItems()]
        if len(selected) < 1 or len(selected) > 5:
            QMessageBox.warning(self, "Invalid selection", "Select 1-5 groups.")
            return
        # normalize stored comparison representation to list of strings
        if selected in self.comparisons:
            QMessageBox.information(self, "Duplicate", "This comparison exists.")
            self.comparison_list.clearSelection()
            return
        self.comparisons.append(selected)
        self.saved_comparisons_list.addItem(" vs ".join(selected))
        self.comparison_list.clearSelection()

    def clear_selected_comparisons(self):
        selected_items = self.saved_comparisons_list.selectedItems()
        for item in selected_items:
            text_list = item.text().split(" vs ")
            # remove from comparisons list by value if present
            for c in self.comparisons:
                if c == text_list:
                    self.comparisons.remove(c)
                    break
            self.saved_comparisons_list.takeItem(self.saved_comparisons_list.row(item))

    # ---------------- MODULE LAUNCH ----------------
    def launch_bargraph_module(self):
        if run_bargraph_analysis is None:
            QMessageBox.critical(self, "Module missing", "analysis_modules.run_bargraph_analysis not found.")
            return
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return

        measurement = self.measurement_combo.currentText()
        training = self.training_combo.currentText()
        lesion_col = self.lesion_combo.currentText() if self.lesion_combo.currentText() not in ["Select", "None"] else None
        depth_col = self.depth_combo.currentText() if self.depth_combo.currentText() not in ["None", "Select", ""] else None
        depth_threshold = self.current_depth_value
        animal_id_col = self.animal_combo.currentText() if self.animal_combo.currentText() not in ["Select", ""] else None

        # validations
        if measurement in ["Select", "", None]:
            QMessageBox.warning(self, "Missing selection", "Select a Measurement column.")
            return
        if training in ["Select", "", None]:
            QMessageBox.warning(self, "Missing selection", "Select a Training column.")
            return
        if not self.comparisons:
            QMessageBox.warning(self, "No comparisons", "Add one or more saved comparisons to analyze.")
            return
        if depth_col and depth_threshold is None:
            QMessageBox.warning(self, "Missing depth threshold", "Depth column selected — enter a numeric depth threshold.")
            return

        base_dir = os.path.dirname(__file__)
        group_colors_path = os.path.join(base_dir, "group_colors.txt")
        lesion_colors_path = os.path.join(base_dir, "lesion_colors.txt")
        output_dir = base_dir

        try:
            results, summary_df = run_bargraph_analysis(
                df=self.df,
                measurement_col=measurement,
                training_col=training,
                lesion_col=lesion_col,
                depth_col=depth_col,
                depth_threshold=depth_threshold,
                animal_id_col=animal_id_col,
                comparisons=self.comparisons,
                group_colors_path=group_colors_path,
                lesion_colors_path=lesion_colors_path,
                output_dir=output_dir
            )
        except Exception as e:
            QMessageBox.critical(self, "Analysis error", f"Error running bar graph module:\n{e}")
            return

        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            csv_path = os.path.join(output_dir, f"bargraph_summary_{ts}.csv")
            try:
                summary_df.to_csv(csv_path, index=False)
                QMessageBox.information(self, "Analysis complete", f"Analysis finished. Summary saved to:\n{csv_path}")
            except Exception:
                QMessageBox.information(self, "Analysis complete", "Analysis finished. Summary returned (could not save duplicate CSV).")
        else:
            QMessageBox.information(self, "Analysis complete", "Analysis finished. No summary data returned.")

    # ---------------- CD ANALYSIS ----------------
    def cd_analysis(self):
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return

        main_file = getattr(self, "loaded_file_path", None)
        if main_file is None:
            QMessageBox.warning(self, "No file", "Load a dataset first.")
            return

        # Sample size
        try:
            sample_size = int(self.cd_entry.text().strip())
            if sample_size <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Enter a valid positive number for sample size.")
            return

        measurement = self.measurement_combo.currentText()
        if measurement in ["Select", "", None]:
            QMessageBox.warning(self, "Missing selection", "Select a Measurement column for CD analysis.")
            return

        depth_col = self.depth_combo.currentText() if self.depth_combo.currentText() not in ["None", "Select", ""] else None
        depth_threshold = self.current_depth_value if depth_col else None
        if depth_col and depth_threshold is None:
            QMessageBox.warning(self, "Missing depth threshold", "Depth column selected — enter a numeric depth threshold.")
            return

        lesion_col = self.lesion_combo.currentText() if self.lesion_combo.currentText() not in ["Select", "None"] else None
        training_col = self.training_combo.currentText() if self.training_combo.currentText() not in ["Select", ""] else None
        filename_col = self.filename_combo.currentText() if self.filename_combo.currentText() not in ["Select", ""] else None
        if filename_col is None:
            QMessageBox.warning(self, "Missing selection", "Select a Filename column.")
            return

        cd_folder = os.path.join(os.path.dirname(__file__), "Raw_Data", "CD_Data")
        os.makedirs(cd_folder, exist_ok=True)

        # Load group colors
        try:
            from analysis_modules.BarGraphModule import load_colors
            base_dir = os.path.dirname(__file__)
            group_colors_path = os.path.join(base_dir, "group_colors.txt")
            group_colors = load_colors(group_colors_path)
        except Exception:
            group_colors = {}

        # Run CD analysis
        try:
            figs, _ = CD_Analysis.run_cd_analysis(
                df_group=self.df,
                comparisons=self.comparisons,
                main_file=main_file,
                measurement_col=measurement,
                cd_folder=cd_folder,
                depth_col=depth_col,
                depth_threshold=depth_threshold,
                training_col=training_col,
                filename_col=filename_col,
                lesion_col=lesion_col,
                sample_size=sample_size,
                group_colors=group_colors,
            )
        except Exception as e:
            # show error and reset view
            QMessageBox.critical(self, "CD Analysis Error", f"Error running CD Analysis:\n{e}")
            self.plot_container.hide()
            self.table.show()
            return

        # Save plots to disk
        output_folder = os.path.join(os.path.dirname(__file__), "Plots")
        os.makedirs(output_folder, exist_ok=True)
        for name, fig in figs.items():
            safe_name = name.replace(" ", "_").replace("/", "_")
            try:
                fig.savefig(os.path.join(output_folder, f"{safe_name}.png"), dpi=600, transparent=True)
            except Exception:
                # continue saving others if one fails
                pass

        # show plot container
        self.plot_container.hide()
        self.table.show()
        
        QMessageBox.information(self, "CD Analysis Complete", f"Plots saved to:\n{output_folder}")

    # ---------------- Raw Trace ANALYSIS ----------------
    def raw_trace(self):
        """
        Average raw electrophysiology traces per comparison group and save
        SVG + CSV plots to the Plots/ folder.
 
        Reads .txt trace files from:
            <script_dir>/Raw_Data/Raw_Traces/
 
        Saves outputs to:
            <script_dir>/Plots/
 
        Uses the Depth column + threshold already configured in the GUI to
        split cells into L2 / L3 layers (if a depth column is selected).
        Uses the saved Comparisons list exactly as built for the bar-graph module.
        """
 
        # ── Guard: module available ──────────────────────────────────────────
        if run_trace_averager is None:
            QMessageBox.critical(
                self, "Module missing",
                "Could not import Raw_Trace_Module.\n\n"
                "Make sure Raw_Trace_Module.py is in the same folder as NeuroGraphX.py."
            )
            return
 
        # ── Guard: data loaded ───────────────────────────────────────────────
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a dataset file first.")
            return
 
        # ── Guard: comparisons defined ───────────────────────────────────────
        if not self.comparisons:
            QMessageBox.warning(
                self, "No comparisons",
                "Add at least one comparison in the 'Saved Comparisons' list."
            )
            return
 
        # ── Resolve columns ──────────────────────────────────────────────────
        training_col = self.training_combo.currentText()
        if training_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection",
                                "Select a Training column.")
            return
 
        filename_col = self.filename_combo.currentText()
        if filename_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection",
                                "Select a Data Filename column.")
            return
 
        depth_col = self.depth_combo.currentText()
        depth_col = None if depth_col in ("None", "Select", "") else depth_col
 
        depth_threshold = self.current_depth_value   # float or None
 
        # If a depth column is selected, a threshold is required so the module
        # knows where to split L2 vs L3.
        if depth_col and depth_threshold is None:
            QMessageBox.warning(
                self, "Missing depth threshold",
                "A Depth column is selected but no threshold has been entered.\n"
                "Enter a numeric value in the Depth field and press OK first."
            )
            return
 
        lesion_col = self.lesion_combo.currentText()
        lesion_col = None if lesion_col in ("Select", "None") else lesion_col
 
        # ── Build paths ──────────────────────────────────────────────────────
        base_dir      = os.path.dirname(os.path.abspath(__file__))
        traces_folder = os.path.join(base_dir, "Raw_Data", "Raw_Traces")
        output_folder = os.path.join(base_dir, "Plots")
 
        # ── Load group colours (reuse existing helper) ───────────────────────
        try:
            from analysis_modules.BarGraphModule import load_colors
            group_colors = load_colors(os.path.join(base_dir, "group_colors.txt"))
        except Exception:
            group_colors = {}
 
        # ── Run ──────────────────────────────────────────────────────────────
        try:
            saved_files, warnings = run_trace_averager(
                df_group       = self.df,
                comparisons    = self.comparisons,
                split_value    = depth_threshold if depth_threshold is not None else 0,
                traces_folder  = traces_folder,
                output_folder  = output_folder,
                training_column = training_col,
                depth_column   = depth_col if depth_col else "",
                filename_column = filename_col,
                lesion_column  = lesion_col,
                group_colors   = group_colors,
                dpi            = 600,
            )
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Folder not found", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Trace Averager Error",
                                 f"Error running Example Trace:\n{e}")
            return
 
        # ── Report results ───────────────────────────────────────────────────
        n_saved = len(saved_files)
        msg     = f"Done — {n_saved} file(s) saved to:\n{output_folder}"
 
        if warnings:
            msg += f"\n\n⚠️  {len(warnings)} warning(s):\n"
            # Show up to 6 warnings to avoid an enormous dialog
            msg += "\n".join(f"  • {w}" for w in warnings[:6])
            if len(warnings) > 6:
                msg += f"\n  … and {len(warnings) - 6} more (see console)."
 
        QMessageBox.information(self, "Example Trace Complete", msg)

    # ---------------- Depth Correlation ANALYSIS ----------------
    def depth_correlation(self):
        if run_rdt_analysis is None:
            QMessageBox.critical(self, "Module missing",
                                 f"Could not import DepthCorrelationModule.\n\nReason: {_RDT_IMPORT_ERROR}")
            return
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return
        if not self.comparisons:
            QMessageBox.warning(self, "No comparisons",
                                "Add at least one comparison before running depth correlation.")
            return
    
        # ── 1. Resolve columns ───────────────────────────────────────────────
        running_col = self.depth_combo.currentText()
        if running_col in ("None", "Select", ""):
            QMessageBox.warning(self, "Missing selection", "Select a Depth column.")
            return
    
        outcome_col = self.measurement_combo.currentText()
        if outcome_col in ("Select", ""):
            QMessageBox.warning(self, "Missing selection", "Select a Measurement column.")
            return
    
        training_col = self.training_combo.currentText()
        training_col = None if training_col in ("Select", "") else training_col
    
        lesion_col = self.lesion_combo.currentText()                 # Bug 1 fix
        lesion_col = None if lesion_col in ("Select", "None") else lesion_col
    
        # ── 2. Optional depth threshold filter ──────────────────────────────
        df_work = self.df.copy()
        if self.current_depth_value is not None:
            df_work = df_work[
                df_work[running_col] <= self.current_depth_value
            ].reset_index(drop=True)
            if df_work.empty:
                QMessageBox.warning(self, "Empty subset",
                                    "No rows remain after applying depth threshold.")
                return
    
        # ── 3. Load colors ───────────────────────────────────────────────────
        base_dir = os.path.dirname(__file__)
        try:
            from analysis_modules.BarGraphModule import load_colors
            group_colors  = load_colors(os.path.join(base_dir, "group_colors.txt"))
            lesion_colors = load_colors(os.path.join(base_dir, "lesion_colors.txt"))  # Bug 2 fix
        except Exception:
            group_colors  = {}
            lesion_colors = {}
    
        # ── 4. Find optimal cutoff ───────────────────────────────────────────
        try:
            optimal_cutoff = find_optimal_cutoff(df_work, running_col, outcome_col)
        except Exception as e:
            QMessageBox.critical(self, "RDT error", f"Could not compute optimal cutoff:\n{e}")
            return
    
        # ── 5. Cutoff selection dialog ───────────────────────────────────────
        method, ok = QInputDialog.getItem(
            self, "Cutoff selection",
            f"RDT optimal cutoff: {optimal_cutoff:.2f} µm\n\nWhich cutoff should be used?",
            ["Use RDT cutoff", "Enter manually"], 0, False)
        if not ok:
            return
    
        use_rdt      = (method == "Use RDT cutoff")
        manual_cutoff = None
        if not use_rdt:
            manual_cutoff, ok = QInputDialog.getDouble(
                self, "Manual cutoff", "Enter cutoff value (µm):",
                value=optimal_cutoff, decimals=2)
            if not ok:
                return
    
        # ── 6. Layer labels ──────────────────────────────────────────────────
        label_below, ok1 = QInputDialog.getText(
            self, "Label below cutoff", "Label for the group BELOW the cutoff:", text="L2")
        if not ok1:
            label_below = "L2"
    
        label_above, ok2 = QInputDialog.getText(
            self, "Label above cutoff", "Label for the group ABOVE the cutoff:", text="L3")
        if not ok2:
            label_above = "L3"
    
        # ── 7. Run ───────────────────────────────────────────────────────────
        try:
            results = run_rdt_analysis(
                df=df_work,
                running_col=running_col,
                outcome_col=outcome_col,
                training_col=training_col,
                lesion_col=lesion_col,
                comparisons=self.comparisons,
                group_colors=group_colors,
                lesion_colors=lesion_colors,
                cutoff=manual_cutoff,           # Bug 3 fix — was manual_val
                use_rdt_cutoff=use_rdt,
                label_below=label_below,
                label_above=label_above,
            )
        except Exception as e:
            QMessageBox.critical(self, "Analysis error", f"Error running depth correlation:\n{e}")
            return
    
        if not results:
            QMessageBox.warning(self, "No results",
                                "No group/lesion combinations had enough data (≥5 rows).")
            return
    
        # ── 8. Save all figures ──────────────────────────────────────────────
        output_dir = os.path.join(base_dir, "Plots")
        os.makedirs(output_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    
        for r in results:                       # Bug 3 fix — results is a list
            safe = r["label"].replace(" ", "_").replace("/", "_").replace("–", "-")
            for fig_key, suffix in [("scatter_fig", "scatter"),
                                     ("tree_fig",    "tree"),
                                     ("bar_fig",     "bar")]:
                try:
                    r[fig_key].savefig(
                        os.path.join(output_dir, f"rdt_{safe}_{suffix}_{ts}.png"),
                        dpi=600, bbox_inches="tight", transparent=True)
                except Exception:
                    pass
    
        # ── 9. Set up navigator and show first result ────────────────────────
        self._rdt_results  = results
        self._rdt_page_idx = 0
        self._rdt_fig_type = 0
    
        self.table.hide()
        self.plot_container.show()
        self.nav_widget.show()
        self._show_rdt_page()
    
        QMessageBox.information(
            self, "Depth Correlation",
            f"Done — {len(results)} group/lesion combination(s) analysed.\n"
            f"Plots saved to:\n{output_dir}\n\n"
            f"Use the navigator below the graph to cycle through results.")

    # ---------------- Figure MAker ----------------
    def make_figures(self):
        """
        Assemble publication figures from plots already saved in Plots/.
        One figure per comparison (no-lesion) or one per lesion + a summary
        figure (lesion mode).  Saves PNG + SVG to Figures/ folder.
        """
        if run_figure_assembly is None:
            QMessageBox.critical(
                self, "Module missing",
                "Could not import FigureAssemblyModule.\n\n"
                "Make sure FigureAssemblyModule.py is in the same folder "
                "as NeuroGraphX.py."
            )
            return
 
        if not self.comparisons:
            QMessageBox.warning(
                self, "No comparisons",
                "Add at least one comparison before making figures."
            )
            return
 
        base_dir      = os.path.dirname(os.path.abspath(__file__))
        plots_folder  = os.path.join(base_dir, "Plots")
        output_folder = os.path.join(base_dir, "Figures")
 
        if not os.path.isdir(plots_folder):
            QMessageBox.warning(
                self, "Plots folder missing",
                f"Could not find the Plots folder:\n{plots_folder}\n\n"
                "Run your analyses first to generate the plot files."
            )
            return
 
        # Determine lesion groups from the lesion combo + loaded data
        lesion_col = self.lesion_combo.currentText()
        lesion_col = None if lesion_col in ("Select", "None") else lesion_col
 
        lesion_groups = None
        if lesion_col and self.df is not None and lesion_col in self.df.columns:
            lesion_groups = sorted(
                self.df[lesion_col].dropna().astype(str).unique().tolist()
            )
 
        # Confirm with the user what is about to be built
        n_comp   = len(self.comparisons)
        mode_str = (f"Lesion mode — {len(lesion_groups)} lesion group(s) "
                    f"+ 1 summary figure per comparison"
                    if lesion_groups
                    else "No-lesion mode — 1 figure per comparison")
 
        reply = QMessageBox.question(
            self, "Make Figures",
            f"Build publication figures?\n\n"
            f"  Comparisons : {n_comp}\n"
            f"  Mode        : {mode_str}\n"
            f"  Source      : {plots_folder}\n"
            f"  Output      : {output_folder}\n\n"
            f"Missing plots will appear as grey placeholders.",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            return
 
        try:
            saved_files, warnings = run_figure_assembly(
                comparisons   = self.comparisons,
                plots_folder  = plots_folder,
                output_folder = output_folder,
                lesion_groups = lesion_groups,
                dpi           = 600,
            )
        except Exception as e:
            QMessageBox.critical(self, "Figure Assembly Error",
                                 f"Error assembling figures:\n{e}")
            return
 
        n_saved = len(saved_files)
        msg     = f"Done — {n_saved} file(s) saved to:\n{output_folder}"
 
        if warnings:
            msg += f"\n\n⚠️  {len(warnings)} warning(s):\n"
            msg += "\n".join(f"  • {w}" for w in warnings[:6])
            if len(warnings) > 6:
                msg += f"\n  … and {len(warnings) - 6} more (see console)."
 
        QMessageBox.information(self, "Make Figures Complete", msg)

    # ---------------- BEHAVIOR TRIALS ----------------
    def behavior_trials(self):
        if run_behavior_trials is None:
            QMessageBox.critical(
                self, "Module missing",
                "Could not import BehaviorTrialsModule.\n\n"
                "Make sure BehaviorTrialsModule.py is in the analysis_modules folder."
            )
            return
    
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return
    
        if not self.comparisons:
            QMessageBox.warning(
                self, "No comparisons",
                "Add at least one comparison before running Behavior Trials."
            )
            return
    
        training_col = self.training_combo.currentText()
        if training_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select a Training column.")
            return
    
        animal_id_col = self.animal_combo.currentText()
        if animal_id_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select an Animal ID column.")
            return
    
        lesion_col = self.lesion_combo.currentText()
        lesion_col = None if lesion_col in ("Select", "None") else lesion_col
    
        # Build lesion filter list from loaded data if lesion col is active
        lesion_filter = None
        if lesion_col and self.df is not None and lesion_col in self.df.columns:
            lesion_filter = sorted(
                self.df[lesion_col].dropna().astype(str).unique().tolist()
            )
    
        base_dir          = os.path.dirname(os.path.abspath(__file__))
        training_data_dir = os.path.join(base_dir, "Raw_Data", "Training_Data")
        output_folder     = os.path.join(base_dir, "Plots")
    
        if not os.path.isdir(training_data_dir):
            QMessageBox.warning(
                self, "Folder missing",
                f"Training_Data folder not found:\n{training_data_dir}\n\n"
                "Create Raw_Data/Training_Data/ and place animal subfolders inside it."
            )
            return
    
        # Confirm before running — can be slow for many animals
        flat_groups = sorted({g.split("-")[-1] for comp in self.comparisons for g in comp})
        reply = QMessageBox.question(
            self, "Behavior Trials",
            f"Run behavioral trial analysis?\n\n"
            f"  Groups     : {', '.join(flat_groups)}\n"
            f"  Lesion col : {lesion_col or 'None'}\n"
            f"  Data dir   : {training_data_dir}\n"
            f"  Output     : {output_folder}\n\n"
            f"This may take a moment for large datasets.",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            return
    
        try:
            all_stats = run_behavior_trials(
                df_master         = self.df,
                animal_id_col     = animal_id_col,
                training_col      = training_col,
                comparisons       = self.comparisons,
                training_data_dir = training_data_dir,
                output_folder     = output_folder,
                lesion_col        = lesion_col,
                lesion_filter     = lesion_filter,
                dpi               = 600,
            )
        except Exception as e:
            QMessageBox.critical(self, "Behavior Trials Error",
                                 f"Error running Behavior Trials:\n{e}")
            return
    
        n = len(all_stats)
        if n == 0:
            QMessageBox.warning(
                self, "No results",
                "No animals were found in Training_Data matching the selected "
                "groups.\n\nCheck that animal ID subfolders exist and match the "
                "Animal ID column values in your spreadsheet."
            )
            return
    
        QMessageBox.information(
            self, "Behavior Trials Complete",
            f"Done — {n} animal(s) processed.\n\n"
            f"Rasters saved to subfolders in:\n{output_folder}\n"
            f"Summary Excel: {output_folder}/BehaviorTrials_Summary.xlsx"
        )

    # ---------------- Behavioral Correlations -------------------
    def behavior_correlations(self):
        if run_correlation is None:
            QMessageBox.critical(
                self, "Module missing",
                "Could not import CorrelationModule.\n\n"
                "Make sure CorrelationModule.py is in the analysis_modules folder."
            )
            return
 
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return
 
        if not self.comparisons:
            QMessageBox.warning(
                self, "No comparisons",
                "Add at least one comparison before running Behavior Correlations."
            )
            return
 
        # ── Resolve columns (same pattern as every other module) ─────────────
        training_col = self.training_combo.currentText()
        if training_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select a Training column.")
            return
 
        animal_id_col = self.animal_combo.currentText()
        if animal_id_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select an Animal ID column.")
            return
 
        measurement_col = self.measurement_combo.currentText()
        if measurement_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select a Measurement column.")
            return
 
        lesion_col = self.lesion_combo.currentText()
        lesion_col = None if lesion_col in ("Select", "None") else lesion_col
 
        depth_col = self.depth_combo.currentText()
        depth_col = None if depth_col in ("None", "Select", "") else depth_col
 
        depth_threshold = self.current_depth_value  # float or None, set by user via OK button
        if depth_col and depth_threshold is None:
            QMessageBox.warning(
                self, "Missing depth threshold",
                "A Depth column is selected but no threshold has been entered.\n"
                "Enter a numeric value in the Depth field and press OK first."
            )
            return
 
        # lesion_filter: all lesion values present in data (same as behavior_trials)
        lesion_filter = None
        if lesion_col and lesion_col in self.df.columns:
            lesion_filter = sorted(
                self.df[lesion_col].dropna().astype(str).unique().tolist()
            )
 
        # ── Paths ─────────────────────────────────────────────────────────────
        base_dir          = os.path.dirname(os.path.abspath(__file__))
        training_data_dir = os.path.join(base_dir, "Raw_Data", "Training_Data")
        output_folder     = os.path.join(base_dir, "Plots")
 
        if not os.path.isdir(training_data_dir):
            QMessageBox.warning(
                self, "Folder missing",
                f"Training_Data folder not found:\n{training_data_dir}\n\n"
                "Create Raw_Data/Training_Data/ and place animal subfolders inside it."
            )
            return
 
        # ── Confirm ───────────────────────────────────────────────────────────
        depth_info = (f"{depth_col} < {depth_threshold} → L2 / ≥ {depth_threshold} → L3"
                      if depth_col else "None")
        reply = QMessageBox.question(
            self, "Behavior Correlations",
            f"Correlate lick behavior against bar graph measurements?\n\n"
            f"  Training col  : {training_col}\n"
            f"  Animal ID col : {animal_id_col}\n"
            f"  Measure col   : {measurement_col}\n"
            f"  Lesion col    : {lesion_col or 'None'}\n"
            f"  Depth split   : {depth_info}\n"
            f"  Data dir      : {training_data_dir}\n"
            f"  Output        : {output_folder}/_correlations/\n\n"
            f"This reads each animal's last valid training day.",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            return
 
        # ── Run ───────────────────────────────────────────────────────────────
        water_ml_col = self.water_ml_combo.currentText()
        water_ml_col = None if water_ml_col in ("None", "Select", "") else water_ml_col

        try:
            run_correlation(
                df_master         = self.df,
                animal_id_col     = animal_id_col,
                training_col      = training_col,
                bar_measure_cols  = [measurement_col],   # the GUI's selected measurement
                comparisons       = self.comparisons,
                training_data_dir = training_data_dir,
                output_folder     = output_folder,
                lesion_col        = lesion_col,
                lesion_filter     = lesion_filter,
                depth_col         = depth_col,
                depth_threshold   = depth_threshold,
                water_ml_col      = water_ml_col,
            )
        except Exception as e:
            QMessageBox.critical(self, "Correlation Error",
                                 f"Error running Behavior Correlations:\n{e}")
            return
 
        QMessageBox.information(
            self, "Behavior Correlations Complete",
            f"Done.\n\nCorrelation figures saved to:\n"
            f"{os.path.join(output_folder, '_correlations')}"
        )

    # ---------------- Behavioral Correlations 2 -------------------
    def behavior_correlations2(self):
        if run_correlation2 is None:
            QMessageBox.critical(
                self, "Module missing",
                "Could not import Correlations2Module.\n\n"
                "Make sure Correlations2Module.py is in the analysis_modules folder."
            )
            return

        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a file first.")
            return

        if not self.comparisons:
            QMessageBox.warning(
                self, "No comparisons",
                "Add at least one comparison before running Correlations 2."
            )
            return

        training_col = self.training_combo.currentText()
        if training_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select a Training column.")
            return

        animal_id_col = self.animal_combo.currentText()
        if animal_id_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select an Animal ID column.")
            return

        measurement_col = self.measurement_combo.currentText()
        if measurement_col in ("Select", "", None):
            QMessageBox.warning(self, "Missing selection", "Select a Measurement column.")
            return

        lesion_col = self.lesion_combo.currentText()
        lesion_col = None if lesion_col in ("Select", "None") else lesion_col

        depth_col = self.depth_combo.currentText()
        depth_col = None if depth_col in ("None", "Select", "") else depth_col

        depth_threshold = self.current_depth_value
        if depth_col and depth_threshold is None:
            QMessageBox.warning(
                self, "Missing depth threshold",
                "A Depth column is selected but no threshold has been entered.\n"
                "Enter a numeric value in the Depth field and press OK first."
            )
            return

        lesion_filter = None
        if lesion_col and lesion_col in self.df.columns:
            lesion_filter = sorted(
                self.df[lesion_col].dropna().astype(str).unique().tolist()
            )

        base_dir          = os.path.dirname(os.path.abspath(__file__))
        training_data_dir = os.path.join(base_dir, "Raw_Data", "Training_Data")
        output_folder     = os.path.join(base_dir, "Plots")

        if not os.path.isdir(training_data_dir):
            QMessageBox.warning(
                self, "Folder missing",
                f"Training_Data folder not found:\n{training_data_dir}\n\n"
                "Create Raw_Data/Training_Data/ and place animal subfolders inside it."
            )
            return

        # Load group & lesion colours
        try:
            from analysis_modules.BarGraphModule import load_colors
            group_colors  = load_colors(os.path.join(base_dir, "group_colors.txt"))
            lesion_colors = load_colors(os.path.join(base_dir, "lesion_colors.txt"))
        except Exception:
            group_colors  = {}
            lesion_colors = {}

        water_ml_col = self.water_ml_combo.currentText()
        water_ml_col = None if water_ml_col in ("None", "Select", "") else water_ml_col

        depth_info = (
            f"{depth_col} < {depth_threshold} \u2192 L2 / \u2265 {depth_threshold} \u2192 L3"
            if depth_col else "None"
        )
        reply = QMessageBox.question(
            self, "Correlations 2",
            f"Run simplified lick-diff correlation figures?\n\n"
            f"  Training col  : {training_col}\n"
            f"  Animal ID col : {animal_id_col}\n"
            f"  Measure col   : {measurement_col}\n"
            f"  Lesion col    : {lesion_col or 'None'}\n"
            f"  Depth split   : {depth_info}\n"
            f"  Water mL col  : {water_ml_col or 'None'}\n"
            f"  Output        : {output_folder}/_corr2_*/\n\n"
            f"Produces lick-diff vs reward size, expected value, and qEPSC.",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply != QMessageBox.Ok:
            return

        try:
            run_correlation2(
                df_master         = self.df,
                animal_id_col     = animal_id_col,
                training_col      = training_col,
                bar_measure_cols  = [measurement_col],
                comparisons       = self.comparisons,
                training_data_dir = training_data_dir,
                output_folder     = output_folder,
                lesion_col        = lesion_col,
                lesion_filter     = lesion_filter,
                depth_col         = depth_col,
                depth_threshold   = depth_threshold,
                water_ml_col      = water_ml_col,
                group_colors      = group_colors,
                lesion_colors     = lesion_colors,
            )
        except Exception as e:
            QMessageBox.critical(self, "Correlations 2 Error",
                                 f"Error running Correlations 2:\n{e}")
            return

        QMessageBox.information(
            self, "Correlations 2 Complete",
            f"Done.\n\nFigures saved to sub-folders under:\n"
            f"{output_folder}\n\n"
            f"  _corr2_reward_size/\n"
            f"  _corr2_expected_value/\n"
            f"  _corr2_qepsc_by_comparison/\n"
            f"  _corr2_qepsc_all/"
        )
    def _replace_left_canvas_with_figure(self, fig):
        try:
            # remove existing widget
            self.plot_layout.removeWidget(self.plot_left)
            self.plot_left.setParent(None)
        except Exception:
            pass
        # create new canvas from figure
        self.plot_left = FigureCanvas(fig)
        self.plot_layout.insertWidget(0, self.plot_left)
        try:
            self.plot_left.draw()
        except Exception:
            pass

    def _replace_right_canvas_with_figure(self, fig):
        try:
            # remove existing widget
            self.plot_layout.removeWidget(self.plot_right)
            self.plot_right.setParent(None)
        except Exception:
            pass
        # create new canvas from figure
        self.plot_right = FigureCanvas(fig)
        self.plot_layout.addWidget(self.plot_right)
        try:
            self.plot_right.draw()
        except Exception:
            pass

    def _show_rdt_page(self):
        """Render the current (result, fig_type) into the plot container."""
        if not self._rdt_results:
            return
        r = self._rdt_results[self._rdt_page_idx]
 
        if self._rdt_fig_type == 1:       # tree left, scatter right
            self._replace_left_canvas_with_figure(r["tree_fig"])
            self._replace_right_canvas_with_figure(r["scatter_fig"])
        elif self._rdt_fig_type == 2:     # bar left, scatter right
            self._replace_left_canvas_with_figure(r["bar_fig"])
            self._replace_right_canvas_with_figure(r["scatter_fig"])
        else:                              # default: scatter left, bar right
            self._replace_left_canvas_with_figure(r["scatter_fig"])
            self._replace_right_canvas_with_figure(r["bar_fig"])
 
        self.plot_container.update()
 
        total = len(self._rdt_results)
        fig_names = ["Scatter + Bar", "Tree", "Bar + Scatter"]
        self.nav_label.setText(
            f"  {self._rdt_page_idx + 1}/{total}  –  {r['label']}  "
            f"[{fig_names[self._rdt_fig_type]}]  "
            f"cutoff={r['chosen_cutoff']:.2f} µm  "
            f"MW p={r['stats']['mw_p']:.4f}")
 
    def _nav_prev(self):
        if not self._rdt_results:
            return
        self._rdt_page_idx = (self._rdt_page_idx - 1) % len(self._rdt_results)
        self._rdt_fig_type = 0
        self._show_rdt_page()
 
    def _nav_next(self):
        if not self._rdt_results:
            return
        self._rdt_page_idx = (self._rdt_page_idx + 1) % len(self._rdt_results)
        self._rdt_fig_type = 0
        self._show_rdt_page()
 
    def _nav_cycle_fig(self):
        """Cycle scatter+bar → tree → bar+scatter within the current result."""
        if not self._rdt_results:
            return
        self._rdt_fig_type = (self._rdt_fig_type + 1) % 3
        self._show_rdt_page()
 
    def _nav_back_to_table(self):
        self.plot_container.hide()
        self.nav_widget.hide()
        self.table.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuroGraphX()
    win.show()
    sys.exit(app.exec_())