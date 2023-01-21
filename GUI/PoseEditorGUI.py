import sys
import os
import math
import glob
import json
import shutil

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from Button import arrangeButton, updatePose

class PoseEditorGUI(QMainWindow):
    def __init__(self, confidence_rate=0.95, linewidth=3.0):
        super().__init__()
        self.window = QWidget()
        self.confidence_rate = confidence_rate
        self.linewidth = linewidth
        
        # Initialize
        self.MARKER_SIZE = 5
        self.PICKER_SIZE = 3
        self.xlim = 512
        self.ylim = 384
        # self.createActions()
        # self.createMenus()
        self.initUI()
        self.initFigure()
        self.isSelectSaveDir = False
        self.isOpenJsonDir = False
        self.isOpenImageDir = False
        self.PoseMemory = []
        self.FigureCanvas.mpl_connect('motion_notify_event', self.motion)
        self.FigureCanvas.mpl_connect('pick_event', self.onpick)
        self.FigureCanvas.mpl_connect('button_release_event', self.release)
        self.window.show()

    def initUI(self):
        self.FigureWidget = QWidget(self)
        self.FigureLayout = QVBoxLayout(self.FigureWidget)
        self.FigureLayout.setContentsMargins(0, 0, 0, 0)
        self.OPImageFigure = QLabel(self)
        print(__file__)
        pixmap = QPixmap(os.path.dirname(os.path.abspath(__file__))+'/images/keypoints_pose_25.png')
        pixmap = pixmap.scaled(258, 420, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.OPImageFigure.setPixmap(pixmap)
        self.OPImageFigure.setScaledContents(True)
        self.FileList = QListWidget(self)
        self.FileList.itemSelectionChanged.connect(self.changeJsonFile)
        self.FileList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.FileList.customContextMenuRequested.connect(self.openFileListMenu)
        self.button_grid = arrangeButton(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.FileList)
        splitter.addWidget(self.FigureWidget)
        splitter.addWidget(self.OPImageFigure)
        hlayout = QHBoxLayout()
        hlayout.addWidget(splitter)
        vlayout = QVBoxLayout()
        vlayout.addLayout(self.button_grid)
        vlayout.addLayout(hlayout)
        self.window.setLayout(vlayout)
        self.window.setWindowTitle('OpenPoseEditor')

    def initFigure(self):
        self.Figure, self.Axes = plt.subplots()
        self.Figure = plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        self.FigureCanvas = FigureCanvas(self.Figure)
        self.FigureLayout.addWidget(self.FigureCanvas)
        self.Axes = self.Figure.add_subplot(1,1,1)
    
    def initLine(self):
        self.neck_line,         = self.Axes.plot([self.pose[0*3],  self.pose[1*3]],  [self.pose[0*3+1],  self.pose[1*3+1]],  linewidth=self.linewidth, color='crimson') 
        self.R_shoulder_line,   = self.Axes.plot([self.pose[1*3],  self.pose[2*3]],  [self.pose[1*3+1],  self.pose[2*3+1]],  linewidth=self.linewidth, color='darkorange') 
        self.R_elbow_line,      = self.Axes.plot([self.pose[2*3],  self.pose[3*3]],  [self.pose[2*3+1],  self.pose[3*3+1]],  linewidth=self.linewidth, color='goldenrod')
        self.R_hand_line,       = self.Axes.plot([self.pose[3*3],  self.pose[4*3]],  [self.pose[3*3+1],  self.pose[4*3+1]],  linewidth=self.linewidth, color='yellow')
        self.L_shoulder_line,   = self.Axes.plot([self.pose[1*3],  self.pose[5*3]],  [self.pose[1*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
        self.L_elbow_line,      = self.Axes.plot([self.pose[5*3],  self.pose[6*3]],  [self.pose[5*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lawngreen')
        self.L_hand_line,       = self.Axes.plot([self.pose[7*3],  self.pose[6*3]],  [self.pose[7*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lime')
        self.body_line,         = self.Axes.plot([self.pose[1*3],  self.pose[8*3]],  [self.pose[1*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='red')
        self.R_hip_line,        = self.Axes.plot([self.pose[8*3],  self.pose[9*3]],  [self.pose[8*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='limegreen')
        self.R_knee_line,       = self.Axes.plot([self.pose[9*3],  self.pose[10*3]], [self.pose[9*3+1],  self.pose[10*3+1]], linewidth=self.linewidth, color='mediumturquoise')
        self.R_ankle_line,      = self.Axes.plot([self.pose[10*3], self.pose[11*3]], [self.pose[10*3+1], self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
        self.L_hip_line,        = self.Axes.plot([self.pose[8*3],  self.pose[12*3]], [self.pose[8*3+1],  self.pose[12*3+1]], linewidth=self.linewidth, color='deepskyblue')
        self.L_knee_line,       = self.Axes.plot([self.pose[12*3], self.pose[13*3]], [self.pose[12*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
        self.L_ankle_line,      = self.Axes.plot([self.pose[13*3], self.pose[14*3]], [self.pose[13*3+1], self.pose[14*3+1]], linewidth=self.linewidth, color='blue')
        self.R_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[15*3]], [self.pose[0*3+1],  self.pose[15*3+1]], linewidth=self.linewidth, color='fuchsia')
        self.L_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[16*3]], [self.pose[0*3+1],  self.pose[16*3+1]], linewidth=self.linewidth, color='purple')
        self.R_ear_line,        = self.Axes.plot([self.pose[15*3], self.pose[17*3]], [self.pose[15*3+1], self.pose[17*3+1]], linewidth=self.linewidth, color='magenta')
        self.L_ear_line,        = self.Axes.plot([self.pose[16*3], self.pose[18*3]], [self.pose[16*3+1], self.pose[18*3+1]], linewidth=self.linewidth, color='indigo')
        self.L_b_toe_line,      = self.Axes.plot([self.pose[14*3], self.pose[19*3]], [self.pose[14*3+1], self.pose[19*3+1]], linewidth=self.linewidth, color='blue')
        self.L_s_toe_line,      = self.Axes.plot([self.pose[19*3], self.pose[20*3]], [self.pose[19*3+1], self.pose[20*3+1]], linewidth=self.linewidth, color='blue')
        self.L_heel_line,       = self.Axes.plot([self.pose[14*3], self.pose[21*3]], [self.pose[14*3+1], self.pose[21*3+1]], linewidth=self.linewidth, color='blue')
        self.R_b_toe_line,      = self.Axes.plot([self.pose[11*3], self.pose[22*3]], [self.pose[11*3+1], self.pose[22*3+1]], linewidth=self.linewidth, color='cyan')
        self.R_s_toe_line,      = self.Axes.plot([self.pose[22*3], self.pose[23*3]], [self.pose[22*3+1], self.pose[23*3+1]], linewidth=self.linewidth, color='cyan')
        self.R_heel_line,       = self.Axes.plot([self.pose[11*3], self.pose[24*3]], [self.pose[11*3+1], self.pose[24*3+1]], linewidth=self.linewidth, color='cyan')
    
    def initPoint(self):
        self.nose        = self.Axes.plot(self.pose[0*3],  self.pose[0*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='crimson')
        self.neck        = self.Axes.plot(self.pose[1*3],  self.pose[1*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='red')
        self.Rshoulder   = self.Axes.plot(self.pose[2*3],  self.pose[2*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='darkorange')
        self.Relbow      = self.Axes.plot(self.pose[3*3],  self.pose[3*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='goldenrod')
        self.Rhand       = self.Axes.plot(self.pose[4*3],  self.pose[4*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='yellow')
        self.Lshoulder   = self.Axes.plot(self.pose[5*3],  self.pose[5*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='yellowgreen')
        self.Lelbow      = self.Axes.plot(self.pose[6*3],  self.pose[6*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='lawngreen')
        self.Lhand       = self.Axes.plot(self.pose[7*3],  self.pose[7*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='lime')
        self.Midhip      = self.Axes.plot(self.pose[8*3],  self.pose[8*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='orangered')
        self.Rhip        = self.Axes.plot(self.pose[9*3],  self.pose[9*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='limegreen')
        self.Rknee       = self.Axes.plot(self.pose[10*3], self.pose[10*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='mediumturquoise')
        self.Rankle      = self.Axes.plot(self.pose[11*3], self.pose[11*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.Lhip        = self.Axes.plot(self.pose[12*3], self.pose[12*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='deepskyblue')
        self.Lknee       = self.Axes.plot(self.pose[13*3], self.pose[13*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='dodgerblue')
        self.Lankle      = self.Axes.plot(self.pose[14*3], self.pose[14*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.Reye        = self.Axes.plot(self.pose[15*3], self.pose[15*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='fuchsia')
        self.Leye        = self.Axes.plot(self.pose[16*3], self.pose[16*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='purple')
        self.Rear        = self.Axes.plot(self.pose[17*3], self.pose[17*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='magenta')
        self.Lear        = self.Axes.plot(self.pose[18*3], self.pose[18*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='indigo')
        self.LBigtoe     = self.Axes.plot(self.pose[19*3], self.pose[19*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.LSmalltoe   = self.Axes.plot(self.pose[20*3], self.pose[20*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.Lheel       = self.Axes.plot(self.pose[21*3], self.pose[21*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.RBigtoe     = self.Axes.plot(self.pose[22*3], self.pose[22*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.RSmalltoe   = self.Axes.plot(self.pose[23*3], self.pose[23*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.Rheel       = self.Axes.plot(self.pose[24*3], self.pose[24*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')

    def motion(self, event):
        global gco
        x = event.xdata
        y = event.ydata
        try:
            gco.set_data(x,y)
        except Exception:
            return
        # Nose
        if "Line2D(_line24)"  in str(gco): 
            self.pose[0*3], self.pose[0*3+1] = x,y
            self.neck_line.remove()
            self.R_eye_line.remove()
            self.L_eye_line.remove()
            self.neck_line,         = self.Axes.plot([self.pose[0*3],  self.pose[1*3]],  [self.pose[0*3+1],  self.pose[1*3+1]],  linewidth=self.linewidth, color='crimson') 
            self.R_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[15*3]], [self.pose[0*3+1],  self.pose[15*3+1]], linewidth=self.linewidth, color='fuchsia')
            self.L_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[16*3]], [self.pose[0*3+1],  self.pose[16*3+1]], linewidth=self.linewidth, color='purple')
        # Neck
        elif "Line2D(_line25)"  in str(gco): 
            self.pose[1*3], self.pose[1*3+1] = x,y
            self.neck_line.remove()
            self.L_shoulder_line.remove()
            self.R_shoulder_line.remove()
            self.body_line.remove()
            self.neck_line,         = self.Axes.plot([self.pose[0*3],  self.pose[1*3]],  [self.pose[0*3+1],  self.pose[1*3+1]],  linewidth=self.linewidth, color='crimson') 
            self.R_shoulder_line,   = self.Axes.plot([self.pose[1*3],  self.pose[2*3]],  [self.pose[1*3+1],  self.pose[2*3+1]],  linewidth=self.linewidth, color='darkorange') 
            self.L_shoulder_line,   = self.Axes.plot([self.pose[1*3],  self.pose[5*3]],  [self.pose[1*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
            self.body_line,         = self.Axes.plot([self.pose[1*3],  self.pose[8*3]],  [self.pose[1*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='red')
        # R Shoulder
        elif "Line2D(_line26)" in str(gco): 
            self.pose[2*3], self.pose[2*3+1] = x,y
            self.R_shoulder_line.remove()
            self.R_elbow_line.remove()
            self.R_shoulder_line,   = self.Axes.plot([self.pose[1*3],  self.pose[2*3]],  [self.pose[1*3+1],  self.pose[2*3+1]],  linewidth=self.linewidth, color='darkorange') 
            self.R_elbow_line,      = self.Axes.plot([self.pose[2*3],  self.pose[3*3]],  [self.pose[2*3+1],  self.pose[3*3+1]],  linewidth=self.linewidth, color='goldenrod')
        # R Elbow
        elif "Line2D(_line27)" in str(gco): 
            self.pose[3*3], self.pose[3*3+1] = x,y
            self.R_elbow_line.remove()
            self.R_hand_line.remove()
            self.R_elbow_line,      = self.Axes.plot([self.pose[2*3],  self.pose[3*3]],  [self.pose[2*3+1],  self.pose[3*3+1]],  linewidth=self.linewidth, color='goldenrod')
            self.R_hand_line,       = self.Axes.plot([self.pose[3*3],  self.pose[4*3]],  [self.pose[3*3+1],  self.pose[4*3+1]],  linewidth=self.linewidth, color='yellow')
        # R Wrist
        elif "Line2D(_line28)" in str(gco): 
            self.pose[4*3], self.pose[4*3+1] = x,y
            self.R_hand_line.remove()
            self.R_hand_line,       = self.Axes.plot([self.pose[3*3],  self.pose[4*3]],  [self.pose[3*3+1],  self.pose[4*3+1]],  linewidth=self.linewidth, color='yellow')
        # L Shoulder
        elif "Line2D(_line29)"  in str(gco):
            self.pose[5*3], self.pose[5*3+1] = x,y
            self.L_shoulder_line.remove()
            self.L_elbow_line.remove()
            self.L_shoulder_line,   = self.Axes.plot([self.pose[1*3],  self.pose[5*3]],  [self.pose[1*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
            self.L_elbow_line,      = self.Axes.plot([self.pose[5*3],  self.pose[6*3]],  [self.pose[5*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lawngreen')
        # L Elbow
        elif "Line2D(_line30)" in str(gco):
            self.pose[6*3], self.pose[6*3+1] = x,y
            self.L_elbow_line.remove()
            self.L_hand_line.remove()
            self.L_elbow_line,      = self.Axes.plot([self.pose[5*3],  self.pose[6*3]],  [self.pose[5*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lawngreen')
            self.L_hand_line,       = self.Axes.plot([self.pose[7*3],  self.pose[6*3]],  [self.pose[7*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lime')
        # L Wrist
        elif "Line2D(_line31)" in str(gco):
            self.pose[7*3], self.pose[7*3+1] = x,y
            self.L_hand_line.remove()
            self.L_hand_line,       = self.Axes.plot([self.pose[7*3],  self.pose[6*3]],  [self.pose[7*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lime')
        # Mid Hip
        elif "Line2D(_line32)" in str(gco):
            self.pose[8*3], self.pose[8*3+1] = x,y
            self.body_line.remove()
            self.L_hip_line.remove()
            self.R_hip_line.remove()
            self.body_line,         = self.Axes.plot([self.pose[1*3],  self.pose[8*3]],  [self.pose[1*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='red')
            self.R_hip_line,        = self.Axes.plot([self.pose[8*3],  self.pose[9*3]],  [self.pose[8*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='limegreen')
            self.L_hip_line,        = self.Axes.plot([self.pose[8*3],  self.pose[12*3]], [self.pose[8*3+1],  self.pose[12*3+1]], linewidth=self.linewidth, color='deepskyblue')
        # Right Hip
        elif "Line2D(_line33)" in str(gco):
            self.pose[9*3], self.pose[9*3+1] = x,y
            self.R_hip_line.remove()
            self.R_knee_line.remove()
            self.R_hip_line,        = self.Axes.plot([self.pose[8*3],  self.pose[9*3]],  [self.pose[8*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='limegreen')
            self.R_knee_line,       = self.Axes.plot([self.pose[9*3],  self.pose[10*3]], [self.pose[9*3+1],  self.pose[10*3+1]], linewidth=self.linewidth, color='mediumturquoise')
        # Right Knee
        elif "Line2D(_line34)" in str(gco):
            self.pose[10*3], self.pose[10*3+1] = x,y
            self.R_knee_line.remove()
            self.R_ankle_line.remove()
            self.R_knee_line,       = self.Axes.plot([self.pose[9*3],  self.pose[10*3]], [self.pose[9*3+1],  self.pose[10*3+1]], linewidth=self.linewidth, color='mediumturquoise')
            self.R_ankle_line,      = self.Axes.plot([self.pose[10*3], self.pose[11*3]], [self.pose[10*3+1], self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
        # Right Ankle
        elif "Line2D(_line35)" in str(gco):
            self.pose[11*3], self.pose[11*3+1] = x,y
            self.R_ankle_line.remove()
            self.R_b_toe_line.remove()
            self.R_heel_line.remove()
            self.R_ankle_line,      = self.Axes.plot([self.pose[10*3], self.pose[11*3]], [self.pose[10*3+1], self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
            self.R_b_toe_line,      = self.Axes.plot([self.pose[11*3], self.pose[22*3]], [self.pose[11*3+1], self.pose[22*3+1]], linewidth=self.linewidth, color='cyan')
            self.R_heel_line,       = self.Axes.plot([self.pose[11*3], self.pose[24*3]], [self.pose[11*3+1], self.pose[24*3+1]], linewidth=self.linewidth, color='cyan')
        # Left Hip
        elif "Line2D(_line36)" in str(gco):
            self.pose[12*3], self.pose[12*3+1] = x,y
            self.L_hip_line.remove()
            self.L_knee_line.remove()
            self.L_hip_line,        = self.Axes.plot([self.pose[8*3],  self.pose[12*3]], [self.pose[8*3+1],  self.pose[12*3+1]], linewidth=self.linewidth, color='deepskyblue')
            self.L_knee_line,       = self.Axes.plot([self.pose[12*3], self.pose[13*3]], [self.pose[12*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
        # Left Knee
        elif "Line2D(_line37)" in str(gco):
            self.pose[13*3], self.pose[13*3+1] = x,y
            self.L_knee_line.remove()
            self.L_ankle_line.remove()
            self.L_knee_line,       = self.Axes.plot([self.pose[12*3], self.pose[13*3]], [self.pose[12*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
            self.L_ankle_line,      = self.Axes.plot([self.pose[13*3], self.pose[14*3]], [self.pose[13*3+1], self.pose[14*3+1]], linewidth=self.linewidth, color='blue')
        # Left Ankle
        elif "Line2D(_line38)" in str(gco):
            self.pose[14*3], self.pose[14*3+1] = x,y
            self.L_ankle_line.remove()
            self.L_b_toe_line.remove()
            self.L_heel_line.remove()
            self.L_ankle_line,      = self.Axes.plot([self.pose[13*3], self.pose[14*3]], [self.pose[13*3+1], self.pose[14*3+1]], linewidth=self.linewidth, color='blue')
            self.L_b_toe_line,      = self.Axes.plot([self.pose[14*3], self.pose[19*3]], [self.pose[14*3+1], self.pose[19*3+1]], linewidth=self.linewidth, color='blue')
            self.L_heel_line,       = self.Axes.plot([self.pose[14*3], self.pose[21*3]], [self.pose[14*3+1], self.pose[21*3+1]], linewidth=self.linewidth, color='blue')
        # Right Eye
        elif "Line2D(_line39)" in str(gco):
            self.pose[15*3], self.pose[15*3+1] = x,y
            self.R_eye_line.remove()
            self.R_ear_line.remove()
            self.R_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[15*3]], [self.pose[0*3+1],  self.pose[15*3+1]], linewidth=self.linewidth, color='fuchsia')
            self.R_ear_line,        = self.Axes.plot([self.pose[15*3], self.pose[17*3]], [self.pose[15*3+1], self.pose[17*3+1]], linewidth=self.linewidth, color='magenta')
        # Left Eye
        elif "Line2D(_line40)" in str(gco):
            self.pose[16*3], self.pose[16*3+1] = x,y
            self.L_eye_line.remove()
            self.L_ear_line.remove()
            self.L_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[16*3]], [self.pose[0*3+1],  self.pose[16*3+1]], linewidth=self.linewidth, color='purple')
            self.L_ear_line,        = self.Axes.plot([self.pose[16*3], self.pose[18*3]], [self.pose[16*3+1], self.pose[18*3+1]], linewidth=self.linewidth, color='indigo')
        # Right Ear
        elif "Line2D(_line41)" in str(gco):
            self.pose[17*3], self.pose[17*3+1] = x,y
            self.R_ear_line.remove()
            self.R_ear_line,        = self.Axes.plot([self.pose[15*3], self.pose[17*3]], [self.pose[15*3+1], self.pose[17*3+1]], linewidth=self.linewidth, color='magenta')
        # Left Ear
        elif "Line2D(_line42)" in str(gco):
            self.pose[18*3], self.pose[18*3+1] = x,y
            self.L_ear_line.remove()
            self.L_ear_line,        = self.Axes.plot([self.pose[16*3], self.pose[18*3]], [self.pose[16*3+1], self.pose[18*3+1]], linewidth=self.linewidth, color='indigo')
        # Left Big Toe
        elif "Line2D(_line43)" in str(gco):
            self.pose[19*3], self.pose[19*3+1] = x,y
            self.L_b_toe_line.remove()            
            self.L_s_toe_line.remove()     
            self.L_b_toe_line,      = self.Axes.plot([self.pose[14*3], self.pose[19*3]], [self.pose[14*3+1], self.pose[19*3+1]], linewidth=self.linewidth, color='blue')
            self.L_s_toe_line,      = self.Axes.plot([self.pose[19*3], self.pose[20*3]], [self.pose[19*3+1], self.pose[20*3+1]], linewidth=self.linewidth, color='blue')
        # Left Small Toe
        elif "Line2D(_line44)" in str(gco):
            self.pose[20*3], self.pose[20*3+1] = x,y
            self.L_s_toe_line.remove()      
            self.L_s_toe_line,      = self.Axes.plot([self.pose[19*3], self.pose[20*3]], [self.pose[19*3+1], self.pose[20*3+1]], linewidth=self.linewidth, color='blue')
        # Left Heel Toe
        elif "Line2D(_line45)" in str(gco):
            self.pose[21*3], self.pose[21*3+1] = x,y
            self.L_heel_line.remove()            
            self.L_heel_line,       = self.Axes.plot([self.pose[14*3], self.pose[21*3]], [self.pose[14*3+1], self.pose[21*3+1]], linewidth=self.linewidth, color='blue')
        # Right Big Toe
        elif "Line2D(_line46)" in str(gco):
            self.pose[22*3], self.pose[22*3+1] = x,y
            self.R_b_toe_line.remove()            
            self.R_s_toe_line.remove() 
            self.R_b_toe_line,      = self.Axes.plot([self.pose[11*3], self.pose[22*3]], [self.pose[11*3+1], self.pose[22*3+1]], linewidth=self.linewidth, color='cyan')
            self.R_s_toe_line,      = self.Axes.plot([self.pose[22*3], self.pose[23*3]], [self.pose[22*3+1], self.pose[23*3+1]], linewidth=self.linewidth, color='cyan')
        # Right Small Toe
        elif "Line2D(_line47)" in str(gco):
            self.pose[23*3], self.pose[23*3+1] = x,y
            self.R_s_toe_line.remove()            
            self.R_s_toe_line,      = self.Axes.plot([self.pose[22*3], self.pose[23*3]], [self.pose[22*3+1], self.pose[23*3+1]], linewidth=self.linewidth, color='cyan')
        # Right Heel Toe
        elif "Line2D(_line48)" in str(gco):
            self.pose[24*3], self.pose[24*3+1] = x,y
            self.R_heel_line.remove()            
            self.R_heel_line,       = self.Axes.plot([self.pose[11*3], self.pose[24*3]], [self.pose[11*3+1], self.pose[24*3+1]], linewidth=self.linewidth, color='cyan')
        
        plt.xlim(0, self.xlim)
        plt.ylim(0, self.ylim)
        # plt.draw()
        self.Figure.canvas.draw_idle()

    def onpick(self, event):
        global gco
        gco = event.artist
        ind = event.ind[0]
        self.PoseMemory.append(tuple(self.pose))
        self.pose = list(self.pose)

    def release(self, event):
        global gco
        gco = None
        self.save()

    def save(self):

        def get_distance(x1, y1, x2, y2):
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        save_path = self.save_dir + self.FileList.selectedItems()[0].text()
        # Invert Pose
        self.pose = tuple(self.pose)
        save_pose = list(self.pose)
        for i in range(len(save_pose)):
            if i%3==0:
                save_pose[i] = round(save_pose[i], 3)
            elif i%3==1:
                save_pose[i] -= self.ylim
                save_pose[i] *= -1
                save_pose[i] = round(save_pose[i], 3)

        # modified joint --> update confidence rate
        for i in range(0, len(save_pose), 3):
            distance = get_distance(save_pose[i], save_pose[i+1], self.original_pose[i], self.original_pose[i+1])
            if distance > 0.01:
                save_pose[i+2] = self.confidence_rate
        self.pose = list(self.pose)
        # Save
        self.json_file['people'][0]['pose_keypoints_2d'] = save_pose
        with open(save_path, 'w') as f:
            json.dump(self.json_file ,f)

    def changeJsonFile(self):
    
        file_name = self.FileList.selectedItems()[0].text()
        index = self.FileList.selectedIndexes()[0].row()
        with open(self.save_dir + file_name, 'r') as f:
            self.json_file = json.load(f)
        with open(self.json_dir + file_name, 'r') as f:
            original_json_file = json.load(f)
        pose = self.json_file['people'][0]['pose_keypoints_2d']
        self.original_pose = tuple(original_json_file['people'][0]['pose_keypoints_2d'])
        # Invert Pose
        for i in range(len(pose)):
            if i%3==1:
                pose[i] *= -1
                pose[i] += self.ylim
        self.pose = pose
        self.PoseMemory = [self.pose]
        # Load Image
        self.Axes.clear()
        try:
            self.image = cv2.imread(self.imageFiles[index], 1)
        except AttributeError:
            QMessageBox.warning(self, "Couldn't find image", 'Open Image Directory first')
            return
        self.image = cv2.flip(self.image, 0)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.Axes_image = self.Axes.imshow(self.image)
        self.initLine()
        self.initPoint()
        plt.xlim(0, self.xlim)
        plt.ylim(0, self.ylim)
        self.Figure.canvas.draw_idle()

    def openFileListMenu(self):
        def copy(self):
            self.copy_pose = tuple(self.pose)
            self.pose = list(self.pose)
        
        def paste(self):
            if hasattr(self, 'copy_pose') is False:
                return
            self.pose = list(self.copy_pose)
            self.PoseMemory.append(self.pose)
            updatePose(self, self.pose)
        
        indexes = self.FileList.selectedIndexes()
        if len(indexes) == 0: return
        index = indexes[0].row()
        menu = QMenu('Menu', self)
        menu.addAction(QAction('Copy', self, triggered=lambda :copy(self)))
        menu.addAction(QAction('Paste', self, triggered=lambda :paste(self)))
        menu.exec_(QCursor.pos())


def main():
    QApp = QtWidgets.QApplication(sys.argv)
    ex = PoseEditorGUI()
    sys.exit(QApp.exec_())

if __name__=='__main__':
    main()