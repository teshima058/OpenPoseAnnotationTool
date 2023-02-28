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
from matplotlib.backends.qt_compat import QtWidgets
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
        self.xlim = 640 #512
        self.ylim = 640 #384
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
        pixmap = QPixmap(os.path.dirname(os.path.abspath(__file__))+'/images/Kanoya_pose.png')
        pixmap = pixmap.scaled(258, 420, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.OPImageFigure.setPixmap(pixmap)
        self.OPImageFigure.setScaledContents(True)
        self.FileList = QListWidget(self)                         
        self.FileList.itemSelectionChanged.connect(self.changeJsonFile)         #annotation変更時
        self.FileList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.FileList.customContextMenuRequested.connect(self.openFileListMenu) #右クリック時
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
        self.neck_line,         = self.Axes.plot([self.pose[0*3],  self.pose[17*3]],  [self.pose[0*3+1],  self.pose[17*3+1]],  linewidth=self.linewidth, color='black') 
        self.R_shoulder_line,   = self.Axes.plot([self.pose[17*3],  self.pose[6*3]],  [self.pose[17*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='salmon') 
        self.R_elbow_line,      = self.Axes.plot([self.pose[6*3],  self.pose[8*3]],  [self.pose[6*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='orangered')
        self.R_hand_line,       = self.Axes.plot([self.pose[8*3],  self.pose[10*3]],  [self.pose[8*3+1],  self.pose[10*3+1]],  linewidth=self.linewidth, color='red')
        self.L_shoulder_line,   = self.Axes.plot([self.pose[17*3],  self.pose[5*3]],  [self.pose[17*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
        self.L_elbow_line,      = self.Axes.plot([self.pose[5*3],  self.pose[7*3]],  [self.pose[5*3+1],  self.pose[7*3+1]],  linewidth=self.linewidth, color='lime')
        self.L_hand_line,       = self.Axes.plot([self.pose[7*3],  self.pose[9*3]],  [self.pose[7*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='darkgreen')
        self.R_hip_line,        = self.Axes.plot([self.pose[17*3],  self.pose[12*3]],  [self.pose[17*3+1],  self.pose[12*3+1]],  linewidth=self.linewidth, color='fuchsia')
        self.R_knee_line,       = self.Axes.plot([self.pose[12*3],  self.pose[14*3]], [self.pose[12*3+1],  self.pose[14*3+1]], linewidth=self.linewidth, color='pink')
        self.R_ankle_line,      = self.Axes.plot([self.pose[14*3], self.pose[16*3]], [self.pose[14*3+1], self.pose[16*3+1]], linewidth=self.linewidth, color='purple')
        self.L_hip_line,        = self.Axes.plot([self.pose[17*3],  self.pose[11*3]], [self.pose[17*3+1],  self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
        self.L_knee_line,       = self.Axes.plot([self.pose[11*3], self.pose[13*3]], [self.pose[11*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
        self.L_ankle_line,      = self.Axes.plot([self.pose[13*3], self.pose[15*3]], [self.pose[13*3+1], self.pose[15*3+1]], linewidth=self.linewidth, color='blue')
        self.R_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[2*3]], [self.pose[0*3+1],  self.pose[2*3+1]], linewidth=self.linewidth, color='gold')
        self.L_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[1*3]], [self.pose[0*3+1],  self.pose[1*3+1]], linewidth=self.linewidth, color='orange')
        self.R_ear_line,        = self.Axes.plot([self.pose[2*3], self.pose[4*3]], [self.pose[2*3+1], self.pose[4*3+1]], linewidth=self.linewidth, color='yellow')
        self.L_ear_line,        = self.Axes.plot([self.pose[1*3], self.pose[3*3]], [self.pose[1*3+1], self.pose[3*3+1]], linewidth=self.linewidth, color='darkorange')
    
    def initPoint(self):
        self.nose        = self.Axes.plot(self.pose[0*3],  self.pose[0*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='black')
        self.Leye        = self.Axes.plot(self.pose[1*3],  self.pose[1*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='orange')
        self.Reye        = self.Axes.plot(self.pose[2*3],  self.pose[2*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='gold')
        self.Lear        = self.Axes.plot(self.pose[3*3],  self.pose[3*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='darkorange')
        self.Rear        = self.Axes.plot(self.pose[4*3],  self.pose[4*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='yellow')
        self.Lshoulder   = self.Axes.plot(self.pose[5*3],  self.pose[5*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='yellowgreen')
        self.Rshoulder   = self.Axes.plot(self.pose[6*3],  self.pose[6*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='salmon')
        self.Lelbow      = self.Axes.plot(self.pose[7*3],  self.pose[7*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='lime')
        self.Relbow      = self.Axes.plot(self.pose[8*3],  self.pose[8*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='orangered')
        self.Lhand       = self.Axes.plot(self.pose[9*3],  self.pose[9*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='darkgreen')
        self.Rhand       = self.Axes.plot(self.pose[10*3], self.pose[10*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='red')
        self.Lhip        = self.Axes.plot(self.pose[11*3], self.pose[11*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.Rhip        = self.Axes.plot(self.pose[12*3], self.pose[12*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='fuchsia')
        self.Lknee       = self.Axes.plot(self.pose[13*3], self.pose[13*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='dodgerblue')
        self.Rknee       = self.Axes.plot(self.pose[14*3], self.pose[14*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='pink')
        self.Lankle      = self.Axes.plot(self.pose[15*3], self.pose[15*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.Rankle      = self.Axes.plot(self.pose[16*3], self.pose[16*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='purple')
        self.neck        = self.Axes.plot(self.pose[17*3], self.pose[17*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='magenta')


    def motion(self, event):
        global gco
        x = event.xdata
        y = event.ydata
        try:
            gco.set_data(x,y)
        except Exception:
            return
        print("zzz")
        print(str(gco))
        print("aaa")
        # Nose
        #if "Line2D(_line17)"  in str(gco): 
        if "Line2D(_child18)"  in str(gco): 
            print("bbb")
            self.pose[0*3], self.pose[0*3+1] = x,y
            self.neck_line.remove()
            self.R_eye_line.remove()
            self.L_eye_line.remove()
            self.neck_line,         = self.Axes.plot([self.pose[0*3],  self.pose[17*3]],  [self.pose[0*3+1],  self.pose[17*3+1]],  linewidth=self.linewidth, color='black') 
            self.R_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[2*3]], [self.pose[0*3+1],  self.pose[2*3+1]], linewidth=self.linewidth, color='gold')
            self.L_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[1*3]], [self.pose[0*3+1],  self.pose[1*3+1]], linewidth=self.linewidth, color='orange')
        # Neck
        #elif "Line2D(_line34)"  in str(gco): 
        elif "Line2D(_child35)"  in str(gco): 
            print("ccc")
            self.pose[17*3], self.pose[17*3+1] = x,y
            self.neck_line.remove()
            self.L_shoulder_line.remove()
            self.R_shoulder_line.remove()
            self.R_hip_line.remove()
            self.L_hip_line.remove()
            self.neck_line,         = self.Axes.plot([self.pose[0*3],  self.pose[17*3]],  [self.pose[0*3+1],  self.pose[17*3+1]],  linewidth=self.linewidth, color='black') 
            self.R_shoulder_line,   = self.Axes.plot([self.pose[17*3],  self.pose[6*3]],  [self.pose[17*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='salmon') 
            self.L_shoulder_line,   = self.Axes.plot([self.pose[17*3],  self.pose[5*3]],  [self.pose[17*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
            self.R_hip_line,        = self.Axes.plot([self.pose[17*3],  self.pose[12*3]],  [self.pose[17*3+1],  self.pose[12*3+1]],  linewidth=self.linewidth, color='fuchsia')
            self.L_hip_line,        = self.Axes.plot([self.pose[17*3],  self.pose[11*3]], [self.pose[17*3+1],  self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
        # R Shoulder
        #elif "Line2D(_line23)" in str(gco): 
        elif "Line2D(_child24)" in str(gco): 
            print("ddd")
            self.pose[6*3], self.pose[6*3+1] = x,y
            self.R_shoulder_line.remove()
            self.R_elbow_line.remove()
            self.R_shoulder_line,   = self.Axes.plot([self.pose[17*3],  self.pose[6*3]],  [self.pose[17*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='salmon') 
            self.R_elbow_line,      = self.Axes.plot([self.pose[6*3],  self.pose[8*3]],  [self.pose[6*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='orangered')
        # R Elbow
        #elif "Line2D(_line25)" in str(gco): 
        elif "Line2D(_child26)" in str(gco): 
            print("eee")
            self.pose[8*3], self.pose[8*3+1] = x,y
            self.R_elbow_line.remove()
            self.R_hand_line.remove()
            self.R_elbow_line,      = self.Axes.plot([self.pose[6*3],  self.pose[8*3]],  [self.pose[6*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='orangered')
            self.R_hand_line,       = self.Axes.plot([self.pose[8*3],  self.pose[10*3]],  [self.pose[8*3+1],  self.pose[10*3+1]],  linewidth=self.linewidth, color='red')
        # R Wrist
        #elif "Line2D(_line27)" in str(gco): 
        elif "Line2D(_child28)" in str(gco): 
            self.pose[10*3], self.pose[10*3+1] = x,y
            self.R_hand_line.remove()
            self.R_hand_line,       = self.Axes.plot([self.pose[8*3],  self.pose[10*3]],  [self.pose[8*3+1],  self.pose[10*3+1]],  linewidth=self.linewidth, color='red')
        # L Shoulder
        #elif "Line2D(_line22)"  in str(gco):
        elif "Line2D(_child23)"  in str(gco):
            self.pose[5*3], self.pose[5*3+1] = x,y
            self.L_shoulder_line.remove()
            self.L_elbow_line.remove()
            self.L_shoulder_line,   = self.Axes.plot([self.pose[17*3],  self.pose[5*3]],  [self.pose[17*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
            self.L_elbow_line,      = self.Axes.plot([self.pose[5*3],  self.pose[7*3]],  [self.pose[5*3+1],  self.pose[7*3+1]],  linewidth=self.linewidth, color='lime')
        # L Elbow
        #elif "Line2D(_line24)" in str(gco):
        elif "Line2D(_child25)" in str(gco):
            self.pose[7*3], self.pose[7*3+1] = x,y
            self.L_elbow_line.remove()
            self.L_hand_line.remove()        
            self.L_elbow_line,      = self.Axes.plot([self.pose[5*3],  self.pose[7*3]],  [self.pose[5*3+1],  self.pose[7*3+1]],  linewidth=self.linewidth, color='lime')
            self.L_hand_line,       = self.Axes.plot([self.pose[7*3],  self.pose[9*3]],  [self.pose[7*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='darkgreen')
        # L Wrist
        #elif "Line2D(_line26)" in str(gco):
        elif "Line2D(_child27)" in str(gco):
            self.pose[9*3], self.pose[9*3+1] = x,y
            self.L_hand_line.remove()
            self.L_hand_line,       = self.Axes.plot([self.pose[7*3],  self.pose[9*3]],  [self.pose[7*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='darkgreen')
        # Right Hip
        #elif "Line2D(_line29)" in str(gco):
        elif "Line2D(_child30)" in str(gco):
            self.pose[12*3], self.pose[12*3+1] = x,y
            self.R_hip_line.remove()
            self.R_knee_line.remove()
            self.R_hip_line,        = self.Axes.plot([self.pose[17*3],  self.pose[12*3]],  [self.pose[17*3+1],  self.pose[12*3+1]],  linewidth=self.linewidth, color='fuchsia')
            self.R_knee_line,       = self.Axes.plot([self.pose[12*3],  self.pose[14*3]], [self.pose[12*3+1],  self.pose[14*3+1]], linewidth=self.linewidth, color='pink')
        # Right Knee
        #elif "Line2D(_line31)" in str(gco):
        elif "Line2D(_child32)" in str(gco):
            self.pose[14*3], self.pose[14*3+1] = x,y
            self.R_knee_line.remove()
            self.R_ankle_line.remove()
            self.R_knee_line,       = self.Axes.plot([self.pose[12*3],  self.pose[14*3]], [self.pose[12*3+1],  self.pose[14*3+1]], linewidth=self.linewidth, color='pink')
            self.R_ankle_line,      = self.Axes.plot([self.pose[14*3], self.pose[16*3]], [self.pose[14*3+1], self.pose[16*3+1]], linewidth=self.linewidth, color='purple')
        # Right Ankle
        #elif "Line2D(_line33)" in str(gco):
        elif "Line2D(_child34)" in str(gco):
            self.pose[16*3], self.pose[16*3+1] = x,y
            self.R_ankle_line.remove()
            self.R_ankle_line,      = self.Axes.plot([self.pose[14*3], self.pose[16*3]], [self.pose[14*3+1], self.pose[16*3+1]], linewidth=self.linewidth, color='purple')
        # Left Hip
        #elif "Line2D(_line28)" in str(gco):
        elif "Line2D(_child29)" in str(gco):
            self.pose[11*3], self.pose[11*3+1] = x,y
            self.L_hip_line.remove()
            self.L_knee_line.remove()
            self.L_hip_line,        = self.Axes.plot([self.pose[17*3],  self.pose[11*3]], [self.pose[17*3+1],  self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
            self.L_knee_line,       = self.Axes.plot([self.pose[11*3], self.pose[13*3]], [self.pose[11*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
        # Left Knee
        #elif "Line2D(_line30)" in str(gco):
        elif "Line2D(_child31)" in str(gco):
            self.pose[13*3], self.pose[13*3+1] = x,y
            self.L_knee_line.remove()
            self.L_ankle_line.remove()
            self.L_knee_line,       = self.Axes.plot([self.pose[11*3], self.pose[13*3]], [self.pose[11*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
            self.L_ankle_line,      = self.Axes.plot([self.pose[13*3], self.pose[15*3]], [self.pose[13*3+1], self.pose[15*3+1]], linewidth=self.linewidth, color='blue')
        # Left Ankle
        #elif "Line2D(_line32)" in str(gco):
        elif "Line2D(_child33)" in str(gco):
            self.pose[15*3], self.pose[15*3+1] = x,y
            self.L_ankle_line.remove()
            self.L_ankle_line,      = self.Axes.plot([self.pose[13*3], self.pose[15*3]], [self.pose[13*3+1], self.pose[15*3+1]], linewidth=self.linewidth, color='blue')
        # Right Eye
        #elif "Line2D(_line19)" in str(gco):
        elif "Line2D(_child20)" in str(gco):
            self.pose[2*3], self.pose[2*3+1] = x,y
            self.R_eye_line.remove()
            self.R_ear_line.remove()
            self.R_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[2*3]], [self.pose[0*3+1],  self.pose[2*3+1]], linewidth=self.linewidth, color='gold')
            self.R_ear_line,        = self.Axes.plot([self.pose[2*3], self.pose[4*3]], [self.pose[2*3+1], self.pose[4*3+1]], linewidth=self.linewidth, color='yellow')
        # Left Eye
        #elif "Line2D(_line18)" in str(gco):
        elif "Line2D(_child19)" in str(gco):
            self.pose[1*3], self.pose[1*3+1] = x,y
            self.L_eye_line.remove()
            self.L_ear_line.remove()
            self.L_eye_line,        = self.Axes.plot([self.pose[0*3],  self.pose[1*3]], [self.pose[0*3+1],  self.pose[1*3+1]], linewidth=self.linewidth, color='orange')
            self.L_ear_line,        = self.Axes.plot([self.pose[1*3], self.pose[3*3]], [self.pose[1*3+1], self.pose[3*3+1]], linewidth=self.linewidth, color='darkorange')
        # Right Ear
        #elif "Line2D(_line21)" in str(gco):
        elif "Line2D(_child22)" in str(gco):
            self.pose[4*3], self.pose[4*3+1] = x,y
            self.R_ear_line.remove()
            self.R_ear_line,        = self.Axes.plot([self.pose[2*3], self.pose[4*3]], [self.pose[2*3+1], self.pose[4*3+1]], linewidth=self.linewidth, color='yellow')
        # Left Ear
        #elif "Line2D(_line20)" in str(gco):
        elif "Line2D(_child21)" in str(gco):
            self.pose[3*3], self.pose[3*3+1] = x,y
            self.L_ear_line.remove()
            self.L_ear_line,        = self.Axes.plot([self.pose[1*3], self.pose[3*3]], [self.pose[1*3+1], self.pose[3*3+1]], linewidth=self.linewidth, color='darkorange')
        
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
                save_pose[i] -= self.image.shape[0] #self.ylim
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
        
        # Load Image
        self.Axes.clear()
        try:
            self.image = cv2.imread(self.imageFiles[index], 1)
        except AttributeError:
            QMessageBox.warning(self, "Couldn't find image", 'Open Image Directory first')
            return
        
        # Invert Pose
        for i in range(len(pose)):
            if i%3==1:
                pose[i] *= -1
                pose[i] += self.image.shape[0] #self.ylim
        self.pose = pose
        self.PoseMemory = [self.pose]
        
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