import os
import glob
import shutil
from natsort import natsorted 

import cv2
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt

def arrangeButton(self):
    button_grid = QGridLayout()
    button_grid.setContentsMargins(0, 0, 0, 0)
    btn_select_savedir = QPushButton('Select Save Directory', self)
    btn_select_savedir.clicked.connect(lambda: selectSaveDir(self))
    btn_open_jsondir = QPushButton('Open JSON Directory', self)
    btn_open_jsondir.clicked.connect(lambda: openJsonDir(self))
    btn_open_imagedir = QPushButton('Open Image Directory', self)
    btn_open_imagedir.clicked.connect(lambda: openImageDir(self))
    btn_origin_pose = QPushButton('Reset to Original Pose', self)
    btn_origin_pose.clicked.connect(lambda: resetOriginalPose(self))
    btn_back = QPushButton('Go Back', self)
    btn_back.clicked.connect(lambda: backOneStep(self))

    button_grid.addWidget(btn_select_savedir, 0, 0)
    button_grid.addWidget(btn_open_jsondir, 0, 1)
    button_grid.addWidget(btn_open_imagedir, 0, 2)
    button_grid.addWidget(btn_origin_pose, 0, 3)
    button_grid.addWidget(btn_back, 0, 4)
    
    return button_grid

def selectSaveDir(self):
    self.save_dir = QFileDialog.getExistingDirectory(self, 'Select a directory to save the fixed JSON') + '/'
    if self.save_dir == '/': return
    self.isSelectSaveDir = True

def openJsonDir(self):
    if self.isSelectSaveDir is False:
        QMessageBox.warning(self, 'Open Json Directory', 'Select save directory first')
        return
    self.json_dir = QFileDialog.getExistingDirectory(self, 'Select a directory with json files you want to fix') + '/'
    if self.json_dir == '/': return
    # Load JSON
    self.jsonFiles = natsorted(glob.glob(self.json_dir + '*.json'))
    if len(self.jsonFiles) == 0:
        QMessageBox.warning(self, 'Open Json Directory', 'There is no json file in ' + self.json_dir)
        return
    isUpdateData = True
    if os.path.exists(self.save_dir + os.path.basename(self.jsonFiles[0])):
        ret = QMessageBox.warning(self, "alert", "A file with the same name already exists in the save directory. Do you want to update?", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.No:
            isUpdateData = False
    if isUpdateData is True:
        for f in self.jsonFiles:
            shutil.copy(f, self.save_dir)
    self.FileList.clear()
    for f in self.jsonFiles:
        self.FileList.addItem(os.path.basename(f))
    self.isOpenJsonDir = True

def openImageDir(self):
    if self.isSelectSaveDir is False:
        QMessageBox.warning(self, 'Open Json Directory', 'Open Json directory first')
        return
    self.image_dir = QFileDialog.getExistingDirectory(self, 'Select the directory with the image corresponding to the pose you want to fix') + '/'
    if self.image_dir == '/': return
    # Load Images
    jpg_images = natsorted(glob.glob(self.image_dir + '*.jpg'))
    bmp_images = natsorted(glob.glob(self.image_dir + '*.bmp'))
    png_images = natsorted(glob.glob(self.image_dir + '*.png'))
    images = jpg_images + bmp_images + png_images
    self.imageFiles = sorted(images)
    if len(self.imageFiles) == 0:
        QMessageBox.warning(self, 'Open Json Folder', 'There is no image in ' + self.image_dir)
        return
    sample_img = cv2.imread(self.imageFiles[0], cv2.IMREAD_COLOR)
    height, width, channels = sample_img.shape[:3]
    self.xlim = 640 #width
    self.ylim = 640 #height
    self.isOpenImageDir = True

def resetOriginalPose(self):
    if self.isOpenImageDir is False:
        return
    # Invert Pose
    pose = list(self.original_pose)
    for i in range(len(pose)):
        if i%3==1:
            pose[i] *= -1
            pose[i] += self.ylim
    self.PoseMemory.append(self.pose)
    updatePose(self, pose)
    self.original_pose = tuple(self.original_pose)

def backOneStep(self):
    if self.isOpenImageDir is False:
        return
    if len(self.PoseMemory) != 0:
        updatePose(self, self.PoseMemory.pop())

def updatePose(self, new_pose):
    self.Axes.clear()
    self.Axes_image = self.Axes.imshow(self.image)
    self.pose = new_pose
    self.initLine()
    self.initPoint()
    plt.xlim(0, self.xlim)
    plt.ylim(0, self.ylim)
    self.Figure.canvas.draw_idle()
    self.save()
