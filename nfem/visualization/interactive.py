"""This module contains an interactive user interface.

Author: Thomas Oberbichler
"""

from PyQt5.QtWidgets import QFrame, QGridLayout, QTextEdit, QMessageBox, QCheckBox, QGroupBox, QApplication, QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QComboBox, QStackedWidget, QLabel, QDoubleSpinBox, QSpinBox
from PyQt5.QtGui import QFontDatabase
from PyQt5 import Qt, QtGui
from PyQt5 import QtCore

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy.linalg as la

import sys

from .plot import plot_model, plot_load_displacement_curve, plot_bounding_cube
from .plot import animate_model
from ..assembler import Assembler

def _create_int_spinbox(value=0, step=1, minimum=-100, maximum=100):
    widget = QSpinBox()
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)
    widget.setValue(value)
    widget.setSingleStep(step)
    return widget

def _create_double_spinbox(value=0, step=0.1, minimum=None, maximum=None):
    widget = QDoubleSpinBox()
    widget.setMinimum(minimum if minimum else -Qt.qInf())
    widget.setMaximum(maximum if maximum else Qt.qInf())
    widget.setValue(value)
    widget.setSingleStep(step)
    return widget

def _create_free_dof_combobox(model, assembler=None, value=None):
    if assembler is None:
        assembler = Assembler(model)
    widget = QComboBox()
    for free_dof in assembler.free_dofs:
        widget.addItem(_dof_to_string(free_dof, assembler))

    if value is not None:
        index = assembler.index_of_dof(value)
        widget.setCurrentIndex(index)
    return widget

def _dof_to_string(dof, assembler):
    index = assembler.index_of_dof(dof)
    return 'Dof #{}: {} {}'.format(index+1, dof[0], dof[1])

def _string_to_dof(string, assembler):
    dof_number = int(string[ 5:string.index(':')-1])
    return assembler.dof_at_index(dof_number-1)

class Stream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class InteractiveWindow(QWidget):
    def __init__(self, model, dof):
        super(InteractiveWindow, self).__init__()

        self.branches = [model]
        self.dof = dof

        self.tolerance = -5
        self.max_iterations = 100

        self.solve_det_k = True
        self.solve_eigenvalue = False

        self.animation_window = None

        # --- setup window

        self.resize(1000, 400)
        self.setWindowTitle('NFEM Teaching Tool')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # --- sidebar

        sidebar = self._create_sidebar()
        layout.addWidget(sidebar, 1, 1, 2, 1)

        # --- plot_canvas

        figure = Figure(dpi=80)
        self.figure = figure

        plot_canvas = FigureCanvasQTAgg(figure)
        plot_canvas.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot_canvas, 1, 2, 1, 1)
        self.plot_canvas = plot_canvas

        plot_3d = figure.add_subplot(1, 2, 1, projection='3d')
        self.plot_3d = plot_3d

        plot_2d = figure.add_subplot(1, 2, 2)
        self.plot_2d = plot_2d

        self.redraw()
        
        # --- log

        widget = QTextEdit()
        widget.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        widget.setReadOnly(True)
        widget.setFixedHeight(200)
        widget.setFrameStyle(QFrame.HLine)
        layout.addWidget(widget, 2, 2, 1, 1)
        self.logTextEdit = widget
        
        sys.stdout = Stream(textWritten=self.write_log)

    def _create_sidebar(self):
        sidebar = QWidget(self)
        sidebar.setFixedWidth(250)
        sidebar.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        sidebar.setLayout(layout)

        # --- prediction

        widget = QGroupBox('Prediction', self)
        layout.addWidget(widget)

        group_layout = QVBoxLayout()
        widget.setLayout(group_layout)

        node_id, dof_type = self.dof

        widget = QComboBox()
        widget.addItem('Set load factor (λ)')
        widget.addItem('Increment load factor (λ)')
        widget.addItem('Set {} at node {}'.format(dof_type, node_id))
        widget.addItem('Increment {} at node {}'.format(dof_type, node_id))
        widget.addItem('Set prediction')
        widget.addItem('Set direction')
        group_layout.addWidget(widget)
        self._predictor_combobox = widget
        
        widget = QStackedWidget()
        widget.addWidget(_LoadPredictorWidget(self))
        widget.addWidget(_LoadIncrementPredictorWidget(self))
        widget.addWidget(_DisplacementPredictorWidget(self, self.dof))
        widget.addWidget(_DisplacementIncrementPredictorWidget(self, self.dof))
        widget.addWidget(_ExplicitPredictorWidget(self, self.dof))
        widget.addWidget(_DirectionPredictorWidget(self, self.dof))
        group_layout.addWidget(widget)
        self._predictor_stack = widget

        self._predictor_combobox.currentIndexChanged.connect(self._predictor_stack.setCurrentIndex)

        # --- strategy

        widget = QGroupBox('Constraint', self)
        layout.addWidget(widget)

        group_layout = QVBoxLayout()
        widget.setLayout(group_layout)

        widget = QComboBox()
        widget.addItem('Linear', 'linear')
        widget.addItem('Load control', 'load-control')
        widget.addItem('Displacement control', 'displacement-control')
        widget.addItem('Arc-length control', 'arc-length-control')
        group_layout.addWidget(widget)
        self._strategy_combobox = widget

        # --- solution

        widget = QGroupBox('Solution', self)
        layout.addWidget(widget)

        group_layout = QVBoxLayout()
        widget.setLayout(group_layout)

        widget = QGroupBox('Newton-Raphson', widget)
        group_layout.addWidget(widget)

        nr_group_layout = QVBoxLayout()
        widget.setLayout(nr_group_layout)

        widget = QLabel('Tolerance:')
        nr_group_layout.addWidget(widget)

        widget = QWidget()
        nr_group_layout.addWidget(widget)

        tolerance_layout = QHBoxLayout()
        tolerance_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(tolerance_layout)

        widget = QLabel('10^')
        tolerance_layout.addWidget(widget)

        widget = _create_int_spinbox(value=self.tolerance)
        widget.valueChanged.connect(self._set_tolerance)
        tolerance_layout.addWidget(widget, 1)

        widget = QLabel('Maximum iterations:')
        nr_group_layout.addWidget(widget)

        widget = _create_int_spinbox(value=self.max_iterations, minimum=0, maximum=1000)
        widget.valueChanged.connect(self._set_max_iterations)
        nr_group_layout.addWidget(widget)

        # --- ---  other solution options

        widget = QCheckBox('Solve Det(K)')
        widget.setChecked(self.solve_det_k)
        widget.stateChanged.connect(self._set_solve_det_K)
        group_layout.addWidget(widget)

        widget = QCheckBox('Solve attendant eigenvalue analysis')        
        widget.setChecked(self.solve_eigenvalue)
        widget.stateChanged.connect(self._set_solve_eigenvalue_analysis)
        group_layout.addWidget(widget)

        #  --- --- action
     
        button = QPushButton('Solve')
        button.clicked.connect(self.solve_button_click)
        group_layout.addWidget(button)

        # --- space

        layout.addStretch(1)

        # --- path control

        widget = QGroupBox('Path control', self)
        layout.addWidget(widget)

        group_layout = QVBoxLayout()
        widget.setLayout(group_layout)

        button = QPushButton('Go back')
        button.clicked.connect(self.go_back_button_click)
        group_layout.addWidget(button)

        button = QPushButton('Reset path')
        button.clicked.connect(self.reset_branch_button_click)
        group_layout.addWidget(button)

        button = QPushButton('New path')
        button.clicked.connect(self.new_branch_button_click)
        group_layout.addWidget(button)

        button = QPushButton('Reset all')
        button.clicked.connect(self.reset_button_click)
        group_layout.addWidget(button)

        button = QPushButton('Animation')
        button.clicked.connect(self._animation_button_click)
        layout.addWidget(button)

        return sidebar

    def _set_tolerance(self, value):
        self.tolerance = value

    def _set_max_iterations(self, value):
        self.max_iterations = value

    def _set_solve_det_K(self, value):
        self.solve_det_k = value

    def _set_solve_eigenvalue_analysis(self, value):
        self.solve_eigenvalue = value

    @property
    def model(self):
        return self.branches[-1]

    @model.setter
    def model(self, value):
        self.branches[-1] = value

    def solve_button_click(self):
        try:
            model = self.model.get_duplicate()

            dof = self.dof

            self._predictor_stack.currentWidget().predict(model)

            selected_strategy = self._strategy_combobox.currentData()

            tolerance = 10**self.tolerance
            max_iterations = self.max_iterations

            if selected_strategy == 'linear':
                model.perform_linear_solution_step()
            elif selected_strategy == 'load-control':
                model.perform_non_linear_solution_step(
                    strategy=selected_strategy,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    solve_det_k=self.solve_det_k,
                    solve_attendant_eigenvalue=self.solve_eigenvalue,
                )
            elif selected_strategy == 'displacement-control':
                model.perform_non_linear_solution_step(
                    strategy=selected_strategy,
                    dof=dof,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    solve_det_k=self.solve_det_k,
                    solve_attendant_eigenvalue=self.solve_eigenvalue,
                )
            elif selected_strategy == 'arc-length-control':
                model.perform_non_linear_solution_step(
                    strategy=selected_strategy,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    solve_det_k=self.solve_det_k,
                    solve_attendant_eigenvalue=self.solve_eigenvalue,
                )
            else:
                raise RuntimeError('Invalid solution strategy:', selected_strategy)

        except Exception as e:
            QMessageBox(QMessageBox.Critical, 'Error', str(e), QMessageBox.Ok, self).show()
            return

        self.model = model

        self.redraw()

    def new_branch_button_click(self):
        new_model = self.model.get_duplicate()

        new_model._previous_model= self.model.get_previous_model()

        self.branches.append(new_model)

        self.redraw()

    def go_back_button_click(self):
        if self.model.get_previous_model() is None:
            return

        self.model = self.model.get_previous_model()

        self.redraw()

    def reset_branch_button_click(self):
        if self.model.get_previous_model() is None:
            return

        self.model = self.model.get_initial_model()

        self.redraw()

    def reset_button_click(self):
        model = self.model.get_initial_model()

        self.branches = [model]

        self.redraw()

    def _animation_button_click(self):
        if self.animation_window is not None:
            self.animation_window.close()
        model = self.model
        self.animation_window = _AnimationWindow(self, model)      

    def redraw(self):
        model = self.model
        node_id, dof_type = self.dof

        plot_3d = self.plot_3d
        plot_2d = self.plot_2d

        plot_3d.clear()
        plot_3d.grid()

        plot_bounding_cube(plot_3d, model)

        plot_model(plot_3d, model, 'gray', True)
        plot_model(plot_3d, model, 'red', False)

        plot_2d.clear()
        plot_2d.set(xlabel='{} at node {}'.format(dof_type, node_id), ylabel='Load factor ($\lambda$)', title='Load-displacement diagram')
        plot_2d.set_facecolor('white')
        plot_2d.yaxis.tick_right()
        plot_2d.yaxis.set_label_position('right')
        plot_2d.grid()

        for model in self.branches:
            plot_load_displacement_curve(plot_2d, model, self.dof)

        self.plot_canvas.draw()

    def closeEvent(self, event):
        if self.animation_window is not None:
            self.animation_window.close()
        super(InteractiveWindow, self).closeEvent(event) 

    def write_log(self, text):
        cursor = self.logTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.logTextEdit.setTextCursor(cursor)
        self.logTextEdit.ensureCursorVisible()


class _LoadPredictorWidget(QWidget):
    def __init__(self, parent):
        super(_LoadPredictorWidget, self).__init__(parent)

        self._lam = 0.0

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        widget = _create_double_spinbox(self._lam)
        widget.valueChanged.connect(self._set_lam)
        layout.addWidget(widget)

        layout.addStretch(1)

    def _set_lam(self, value):
        self._lam = value

    def predict(self, model):
        model.lam = self._lam

class _LoadIncrementPredictorWidget(QWidget):
    def __init__(self, parent):
        super(_LoadIncrementPredictorWidget, self).__init__(parent)

        self._delta_lam = 0.1

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        widget = _create_double_spinbox(self._delta_lam)
        widget.valueChanged.connect(self._set_delta_lam)
        layout.addWidget(widget)

        layout.addStretch(1)

    def _set_delta_lam(self, value):
        self._delta_lam = value

    def predict(self, model):
        model.lam += self._delta_lam

class _DisplacementPredictorWidget(QWidget):
    def __init__(self, parent, dof):
        super(_DisplacementPredictorWidget, self).__init__(parent)

        self._dof = dof
        self._d = 0.0

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        widget = _create_double_spinbox(self._d)
        widget.valueChanged.connect(self._set_d)
        layout.addWidget(widget)

        layout.addStretch(1)

    def _set_d(self, value):
        self._d = value

    def predict(self, model):
        model.set_dof_state(self._dof, self._d)

class _DisplacementIncrementPredictorWidget(QWidget):
    def __init__(self, parent, dof):
        super(_DisplacementIncrementPredictorWidget, self).__init__(parent)

        self._dof = dof
        self._delta_d = -0.1

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        widget = _create_double_spinbox(self._delta_d)
        widget.valueChanged.connect(self._set_delta_d)
        layout.addWidget(widget)

        layout.addStretch(1)

    def _set_delta_d(self, value):
        self._delta_d = value

    def predict(self, model):
        current_d = model.get_dof_state(self._dof)
        model.set_dof_state(self._dof, current_d + self._delta_d)

class _ExplicitPredictorWidget(QWidget):
    def __init__(self, parent, dof):
        super(_ExplicitPredictorWidget, self).__init__(parent)

        self._dof = dof
        self._lam = 0.0
        self._d = 0.0

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        widget = QLabel('Set load factor (λ):')
        layout.addWidget(widget)

        widget = _create_double_spinbox(self._lam)
        widget.valueChanged.connect(self._set_lam)
        layout.addWidget(widget)

        widget = QLabel('Set {} at {}:'.format(dof[1], dof[0]))
        layout.addWidget(widget)

        widget = _create_double_spinbox(self._d)
        widget.valueChanged.connect(self._set_d)
        layout.addWidget(widget)

        layout.addStretch(1)

    def _set_lam(self, value):
        self._lam = value

    def _set_d(self, value):
        self._d = value

    def predict(self, model):
        model.lam = self._lam
        model.set_dof_state(self._dof, self._d)

class _DirectionPredictorWidget(QWidget):
    def __init__(self, parent, dof):
        super(_DirectionPredictorWidget, self).__init__(parent)

        self._dof = dof
        self._delta_lam = 0.1
        self._delta_d = -0.1
        self._scale = True
        self._length = 0.1

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        widget = QLabel('Increment load factor (λ):')
        layout.addWidget(widget)

        widget = _create_double_spinbox(self._delta_lam)
        widget.valueChanged.connect(self._set_delta_lam)
        layout.addWidget(widget)
        
        widget = QLabel('Increment {} at {}:'.format(dof[1], dof[0]))
        layout.addWidget(widget)

        widget = _create_double_spinbox(self._delta_d)
        widget.valueChanged.connect(self._set_delta_d)
        layout.addWidget(widget)

        widget = QCheckBox('Set length:')
        widget.setChecked(self._scale)
        widget.stateChanged.connect(self._set_scale)
        layout.addWidget(widget)

        widget = _create_double_spinbox(self._length)
        widget.valueChanged.connect(self._set_length)
        layout.addWidget(widget)

        layout.addStretch(1)

    def _set_scale(self, value):
        self._scale = value

    def _set_delta_lam(self, value):
        self._delta_lam = value

    def _set_delta_d(self, value):
        self._delta_d = value

    def _set_length(self, value):
        self._length = value

    def predict(self, model):
        current_d = model.get_dof_state(self._dof)

        delta_lam = self._delta_lam
        delta_d = self._delta_d

        if self._scale:
            factor = self._length / la.norm([delta_lam, delta_d])
            delta_lam *= factor
            delta_d *= factor

        model.lam += delta_lam
        model.set_dof_state(self._dof, current_d + delta_d)

class _AnimationWindow(QWidget):
    def __init__(self, parent, model):
        super(_AnimationWindow, self).__init__()

        self.setWindowTitle('Animation')

        layout = QVBoxLayout() 

        figure = Figure(dpi=80)
        figure = figure
        
        animation_canvas = FigureCanvasQTAgg(figure)
        animation_canvas.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(animation_canvas)  

        ax_3d = figure.add_subplot(1, 1, 1, projection='3d')
        ax_3d = ax_3d

        self.setLayout(layout)

        # store the animation
        self.a = animate_model(figure, ax_3d, model)

        self.show()

def interact(model, dof):
    app = QApplication([])

    window = InteractiveWindow(model, dof=dof)
    window.show()

    app.exec_()

    return window.model
