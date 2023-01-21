import numpy as np
import math
import json
import cv2
import matplotlib.pyplot as plt

class PoseEditor():
    def __init__(self, pose, image_path, xlim, ylim, confidence_rate=0.95, linewidth=3.0):
        gco = None
        self.original_pose = tuple(pose)
        self.xlim = xlim
        self.ylim = ylim
        self.confidence_rate = confidence_rate
        self.linewidth = linewidth
        # Invert Pose
        for i in range(len(pose)):
            if i%3==1:
                pose[i] *= -1
                pose[i] += self.ylim
        self.pose = pose
        
        # Load Image
        self.image = cv2.imread(image_path,1)
        # invert y
        self.image = cv2.flip(self.image, 0)
        # RGB -> BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        plt.imshow(self.image)
        
        # Initialize
        self.MARKER_SIZE = 5
        self.PICKER_SIZE = 3
        self.initLine()
        self.initPoint()
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.connect('motion_notify_event', self.motion)
        plt.connect('pick_event', self.onpick)
        plt.connect('button_release_event', self.release)
        plt.show()
    
    def initLine(self):
        self.neck_line,         = plt.plot([self.pose[0*3],  self.pose[1*3]],  [self.pose[0*3+1],  self.pose[1*3+1]],  linewidth=self.linewidth, color='crimson') 
        self.R_shoulder_line,   = plt.plot([self.pose[1*3],  self.pose[2*3]],  [self.pose[1*3+1],  self.pose[2*3+1]],  linewidth=self.linewidth, color='darkorange') 
        self.R_elbow_line,      = plt.plot([self.pose[2*3],  self.pose[3*3]],  [self.pose[2*3+1],  self.pose[3*3+1]],  linewidth=self.linewidth, color='goldenrod')
        self.R_hand_line,       = plt.plot([self.pose[3*3],  self.pose[4*3]],  [self.pose[3*3+1],  self.pose[4*3+1]],  linewidth=self.linewidth, color='yellow')
        self.L_shoulder_line,   = plt.plot([self.pose[1*3],  self.pose[5*3]],  [self.pose[1*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
        self.L_elbow_line,      = plt.plot([self.pose[5*3],  self.pose[6*3]],  [self.pose[5*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lawngreen')
        self.L_hand_line,       = plt.plot([self.pose[7*3],  self.pose[6*3]],  [self.pose[7*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lime')
        self.body_line,         = plt.plot([self.pose[1*3],  self.pose[8*3]],  [self.pose[1*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='red')
        self.R_hip_line,        = plt.plot([self.pose[8*3],  self.pose[9*3]],  [self.pose[8*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='limegreen')
        self.R_knee_line,       = plt.plot([self.pose[9*3],  self.pose[10*3]], [self.pose[9*3+1],  self.pose[10*3+1]], linewidth=self.linewidth, color='mediumturquoise')
        self.R_ankle_line,      = plt.plot([self.pose[10*3], self.pose[11*3]], [self.pose[10*3+1], self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
        self.L_hip_line,        = plt.plot([self.pose[8*3],  self.pose[12*3]], [self.pose[8*3+1],  self.pose[12*3+1]], linewidth=self.linewidth, color='deepskyblue')
        self.L_knee_line,       = plt.plot([self.pose[12*3], self.pose[13*3]], [self.pose[12*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
        self.L_ankle_line,      = plt.plot([self.pose[13*3], self.pose[14*3]], [self.pose[13*3+1], self.pose[14*3+1]], linewidth=self.linewidth, color='blue')
        self.R_eye_line,        = plt.plot([self.pose[0*3],  self.pose[15*3]], [self.pose[0*3+1],  self.pose[15*3+1]], linewidth=self.linewidth, color='mediumvioletred')
        self.L_eye_line,        = plt.plot([self.pose[0*3],  self.pose[16*3]], [self.pose[0*3+1],  self.pose[16*3+1]], linewidth=self.linewidth, color='deeppink')
        self.R_ear_line,        = plt.plot([self.pose[15*3], self.pose[17*3]], [self.pose[15*3+1], self.pose[17*3+1]], linewidth=self.linewidth, color='darkviolet')
        self.L_ear_line,        = plt.plot([self.pose[16*3], self.pose[18*3]], [self.pose[16*3+1], self.pose[18*3+1]], linewidth=self.linewidth, color='purple')
        self.L_b_toe_line,      = plt.plot([self.pose[14*3], self.pose[19*3]], [self.pose[14*3+1], self.pose[19*3+1]], linewidth=self.linewidth, color='blue')
        self.L_s_toe_line,      = plt.plot([self.pose[19*3], self.pose[20*3]], [self.pose[19*3+1], self.pose[20*3+1]], linewidth=self.linewidth, color='blue')
        self.L_heel_line,       = plt.plot([self.pose[14*3], self.pose[21*3]], [self.pose[14*3+1], self.pose[21*3+1]], linewidth=self.linewidth, color='blue')
        self.R_b_toe_line,      = plt.plot([self.pose[11*3], self.pose[22*3]], [self.pose[11*3+1], self.pose[22*3+1]], linewidth=self.linewidth, color='cyan')
        self.R_s_toe_line,      = plt.plot([self.pose[22*3], self.pose[23*3]], [self.pose[22*3+1], self.pose[23*3+1]], linewidth=self.linewidth, color='cyan')
        self.R_heel_line,       = plt.plot([self.pose[11*3], self.pose[24*3]], [self.pose[11*3+1], self.pose[24*3+1]], linewidth=self.linewidth, color='cyan')
    
    def initPoint(self):
        self.nose        = plt.plot(self.pose[0*3],  self.pose[0*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='crimson')
        self.neck        = plt.plot(self.pose[1*3],  self.pose[1*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='red')
        self.Rshoulder   = plt.plot(self.pose[2*3],  self.pose[2*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='darkorange')
        self.Relbow      = plt.plot(self.pose[3*3],  self.pose[3*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='goldenrod')
        self.Rhand       = plt.plot(self.pose[4*3],  self.pose[4*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='yellow')
        self.Lshoulder   = plt.plot(self.pose[5*3],  self.pose[5*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='yellowgreen')
        self.Lelbow      = plt.plot(self.pose[6*3],  self.pose[6*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='lawngreen')
        self.Lhand       = plt.plot(self.pose[7*3],  self.pose[7*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='lime')
        self.Midhip      = plt.plot(self.pose[8*3],  self.pose[8*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='orangered')
        self.Rhip        = plt.plot(self.pose[9*3],  self.pose[9*3+1],  marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='limegreen')
        self.Rknee       = plt.plot(self.pose[10*3], self.pose[10*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='mediumturquoise')
        self.Rankle      = plt.plot(self.pose[11*3], self.pose[11*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.Lhip        = plt.plot(self.pose[12*3], self.pose[12*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='deepskyblue')
        self.Lknee       = plt.plot(self.pose[13*3], self.pose[13*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='dodgerblue')
        self.Lankle      = plt.plot(self.pose[14*3], self.pose[14*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.Reye        = plt.plot(self.pose[15*3], self.pose[15*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='mediumvioletred')
        self.Leye        = plt.plot(self.pose[16*3], self.pose[16*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='deeppink')
        self.Rear        = plt.plot(self.pose[17*3], self.pose[17*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='darkviolet')
        self.Lear        = plt.plot(self.pose[18*3], self.pose[18*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='purple')
        self.LBigtoe     = plt.plot(self.pose[19*3], self.pose[19*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.LSmalltoe   = plt.plot(self.pose[20*3], self.pose[20*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.Lheel       = plt.plot(self.pose[21*3], self.pose[21*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='blue')
        self.RBigtoe     = plt.plot(self.pose[22*3], self.pose[22*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.RSmalltoe   = plt.plot(self.pose[23*3], self.pose[23*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')
        self.Rheel       = plt.plot(self.pose[24*3], self.pose[24*3+1], marker='o', markersize=self.MARKER_SIZE, picker=self.PICKER_SIZE, color='cyan')

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
            self.neck_line,         = plt.plot([self.pose[0*3],  self.pose[1*3]],  [self.pose[0*3+1],  self.pose[1*3+1]],  linewidth=self.linewidth, color='crimson') 
            self.R_eye_line,        = plt.plot([self.pose[0*3],  self.pose[15*3]], [self.pose[0*3+1],  self.pose[15*3+1]], linewidth=self.linewidth, color='mediumvioletred')
            self.L_eye_line,        = plt.plot([self.pose[0*3],  self.pose[16*3]], [self.pose[0*3+1],  self.pose[16*3+1]], linewidth=self.linewidth, color='deeppink')
        # Neck
        elif "Line2D(_line25)"  in str(gco): 
            self.pose[1*3], self.pose[1*3+1] = x,y
            self.neck_line.remove()
            self.L_shoulder_line.remove()
            self.R_shoulder_line.remove()
            self.body_line.remove()
            self.neck_line,         = plt.plot([self.pose[0*3],  self.pose[1*3]],  [self.pose[0*3+1],  self.pose[1*3+1]],  linewidth=self.linewidth, color='crimson') 
            self.R_shoulder_line,   = plt.plot([self.pose[1*3],  self.pose[2*3]],  [self.pose[1*3+1],  self.pose[2*3+1]],  linewidth=self.linewidth, color='darkorange') 
            self.L_shoulder_line,   = plt.plot([self.pose[1*3],  self.pose[5*3]],  [self.pose[1*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
            self.body_line,         = plt.plot([self.pose[1*3],  self.pose[8*3]],  [self.pose[1*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='red')
        # R Shoulder
        elif "Line2D(_line26)" in str(gco): 
            self.pose[2*3], self.pose[2*3+1] = x,y
            self.R_shoulder_line.remove()
            self.R_elbow_line.remove()
            self.R_shoulder_line,   = plt.plot([self.pose[1*3],  self.pose[2*3]],  [self.pose[1*3+1],  self.pose[2*3+1]],  linewidth=self.linewidth, color='darkorange') 
            self.R_elbow_line,      = plt.plot([self.pose[2*3],  self.pose[3*3]],  [self.pose[2*3+1],  self.pose[3*3+1]],  linewidth=self.linewidth, color='goldenrod')
        # R Elbow
        elif "Line2D(_line27)" in str(gco): 
            self.pose[3*3], self.pose[3*3+1] = x,y
            self.R_elbow_line.remove()
            self.R_hand_line.remove()
            self.R_elbow_line,      = plt.plot([self.pose[2*3],  self.pose[3*3]],  [self.pose[2*3+1],  self.pose[3*3+1]],  linewidth=self.linewidth, color='goldenrod')
            self.R_hand_line,       = plt.plot([self.pose[3*3],  self.pose[4*3]],  [self.pose[3*3+1],  self.pose[4*3+1]],  linewidth=self.linewidth, color='yellow')
        # R Wrist
        elif "Line2D(_line28)" in str(gco): 
            self.pose[4*3], self.pose[4*3+1] = x,y
            self.R_hand_line.remove()
            self.R_hand_line,       = plt.plot([self.pose[3*3],  self.pose[4*3]],  [self.pose[3*3+1],  self.pose[4*3+1]],  linewidth=self.linewidth, color='yellow')
        # L Shoulder
        elif "Line2D(_line29)"  in str(gco):
            self.pose[5*3], self.pose[5*3+1] = x,y
            self.L_shoulder_line.remove()
            self.L_elbow_line.remove()
            self.L_shoulder_line,   = plt.plot([self.pose[1*3],  self.pose[5*3]],  [self.pose[1*3+1],  self.pose[5*3+1]],  linewidth=self.linewidth, color='yellowgreen') 
            self.L_elbow_line,      = plt.plot([self.pose[5*3],  self.pose[6*3]],  [self.pose[5*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lawngreen')
        # L Elbow
        elif "Line2D(_line30)" in str(gco):
            self.pose[6*3], self.pose[6*3+1] = x,y
            self.L_elbow_line.remove()
            self.L_hand_line.remove()
            self.L_elbow_line,      = plt.plot([self.pose[5*3],  self.pose[6*3]],  [self.pose[5*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lawngreen')
            self.L_hand_line,       = plt.plot([self.pose[7*3],  self.pose[6*3]],  [self.pose[7*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lime')
        # L Wrist
        elif "Line2D(_line31)" in str(gco):
            self.pose[7*3], self.pose[7*3+1] = x,y
            self.L_hand_line.remove()
            self.L_hand_line,       = plt.plot([self.pose[7*3],  self.pose[6*3]],  [self.pose[7*3+1],  self.pose[6*3+1]],  linewidth=self.linewidth, color='lime')
        # Mid Hip
        elif "Line2D(_line32)" in str(gco):
            self.pose[8*3], self.pose[8*3+1] = x,y
            self.body_line.remove()
            self.L_hip_line.remove()
            self.R_hip_line.remove()
            self.body_line,         = plt.plot([self.pose[1*3],  self.pose[8*3]],  [self.pose[1*3+1],  self.pose[8*3+1]],  linewidth=self.linewidth, color='red')
            self.R_hip_line,        = plt.plot([self.pose[8*3],  self.pose[9*3]],  [self.pose[8*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='limegreen')
            self.L_hip_line,        = plt.plot([self.pose[8*3],  self.pose[12*3]], [self.pose[8*3+1],  self.pose[12*3+1]], linewidth=self.linewidth, color='deepskyblue')
        # Right Hip
        elif "Line2D(_line33)" in str(gco):
            self.pose[9*3], self.pose[9*3+1] = x,y
            self.R_hip_line.remove()
            self.R_knee_line.remove()
            self.R_hip_line,        = plt.plot([self.pose[8*3],  self.pose[9*3]],  [self.pose[8*3+1],  self.pose[9*3+1]],  linewidth=self.linewidth, color='limegreen')
            self.R_knee_line,       = plt.plot([self.pose[9*3],  self.pose[10*3]], [self.pose[9*3+1],  self.pose[10*3+1]], linewidth=self.linewidth, color='mediumturquoise')
        # Right Knee
        elif "Line2D(_line34)" in str(gco):
            self.pose[10*3], self.pose[10*3+1] = x,y
            self.R_knee_line.remove()
            self.R_ankle_line.remove()
            self.R_knee_line,       = plt.plot([self.pose[9*3],  self.pose[10*3]], [self.pose[9*3+1],  self.pose[10*3+1]], linewidth=self.linewidth, color='mediumturquoise')
            self.R_ankle_line,      = plt.plot([self.pose[10*3], self.pose[11*3]], [self.pose[10*3+1], self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
        # Right Ankle
        elif "Line2D(_line35)" in str(gco):
            self.pose[11*3], self.pose[11*3+1] = x,y
            self.R_ankle_line.remove()
            self.R_b_toe_line.remove()
            self.R_heel_line.remove()
            self.R_ankle_line,      = plt.plot([self.pose[10*3], self.pose[11*3]], [self.pose[10*3+1], self.pose[11*3+1]], linewidth=self.linewidth, color='cyan')
            self.R_b_toe_line,      = plt.plot([self.pose[11*3], self.pose[22*3]], [self.pose[11*3+1], self.pose[22*3+1]], linewidth=self.linewidth, color='cyan')
            self.R_heel_line,       = plt.plot([self.pose[11*3], self.pose[24*3]], [self.pose[11*3+1], self.pose[24*3+1]], linewidth=self.linewidth, color='cyan')
        # Left Hip
        elif "Line2D(_line36)" in str(gco):
            self.pose[12*3], self.pose[12*3+1] = x,y
            self.L_hip_line.remove()
            self.L_knee_line.remove()
            self.L_hip_line,        = plt.plot([self.pose[8*3],  self.pose[12*3]], [self.pose[8*3+1],  self.pose[12*3+1]], linewidth=self.linewidth, color='deepskyblue')
            self.L_knee_line,       = plt.plot([self.pose[12*3], self.pose[13*3]], [self.pose[12*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
        # Left Knee
        elif "Line2D(_line37)" in str(gco):
            self.pose[13*3], self.pose[13*3+1] = x,y
            self.L_knee_line.remove()
            self.L_ankle_line.remove()
            self.L_knee_line,       = plt.plot([self.pose[12*3], self.pose[13*3]], [self.pose[12*3+1], self.pose[13*3+1]], linewidth=self.linewidth, color='dodgerblue')
            self.L_ankle_line,      = plt.plot([self.pose[13*3], self.pose[14*3]], [self.pose[13*3+1], self.pose[14*3+1]], linewidth=self.linewidth, color='blue')
        # Left Ankle
        elif "Line2D(_line38)" in str(gco):
            self.pose[14*3], self.pose[14*3+1] = x,y
            self.L_ankle_line.remove()
            self.L_b_toe_line.remove()
            self.L_heel_line.remove()
            self.L_ankle_line,      = plt.plot([self.pose[13*3], self.pose[14*3]], [self.pose[13*3+1], self.pose[14*3+1]], linewidth=self.linewidth, color='blue')
            self.L_b_toe_line,      = plt.plot([self.pose[14*3], self.pose[19*3]], [self.pose[14*3+1], self.pose[19*3+1]], linewidth=self.linewidth, color='blue')
            self.L_heel_line,       = plt.plot([self.pose[14*3], self.pose[21*3]], [self.pose[14*3+1], self.pose[21*3+1]], linewidth=self.linewidth, color='blue')
        # Right Eye
        elif "Line2D(_line39)" in str(gco):
            self.pose[15*3], self.pose[15*3+1] = x,y
            self.R_eye_line.remove()
            self.R_ear_line.remove()
            self.R_eye_line,        = plt.plot([self.pose[0*3],  self.pose[15*3]], [self.pose[0*3+1],  self.pose[15*3+1]], linewidth=self.linewidth, color='mediumvioletred')
            self.R_ear_line,        = plt.plot([self.pose[15*3], self.pose[17*3]], [self.pose[15*3+1], self.pose[17*3+1]], linewidth=self.linewidth, color='darkviolet')
        # Left Eye
        elif "Line2D(_line40)" in str(gco):
            self.pose[16*3], self.pose[16*3+1] = x,y
            self.L_eye_line.remove()
            self.L_ear_line.remove()
            self.L_eye_line,        = plt.plot([self.pose[0*3],  self.pose[16*3]], [self.pose[0*3+1],  self.pose[16*3+1]], linewidth=self.linewidth, color='deeppink')
            self.L_ear_line,        = plt.plot([self.pose[16*3], self.pose[18*3]], [self.pose[16*3+1], self.pose[18*3+1]], linewidth=self.linewidth, color='purple')
        # Right Ear
        elif "Line2D(_line41)" in str(gco):
            self.pose[17*3], self.pose[17*3+1] = x,y
            self.R_ear_line.remove()
            self.R_ear_line,        = plt.plot([self.pose[15*3], self.pose[17*3]], [self.pose[15*3+1], self.pose[17*3+1]], linewidth=self.linewidth, color='darkviolet')
        # Left Ear
        elif "Line2D(_line42)" in str(gco):
            self.pose[18*3], self.pose[18*3+1] = x,y
            self.L_ear_line.remove()
            self.L_ear_line,        = plt.plot([self.pose[16*3], self.pose[18*3]], [self.pose[16*3+1], self.pose[18*3+1]], linewidth=self.linewidth, color='purple')
        # Left Big Toe
        elif "Line2D(_line43)" in str(gco):
            self.pose[19*3], self.pose[19*3+1] = x,y
            self.L_b_toe_line.remove()            
            self.L_s_toe_line.remove()     
            self.L_b_toe_line,      = plt.plot([self.pose[14*3], self.pose[19*3]], [self.pose[14*3+1], self.pose[19*3+1]], linewidth=self.linewidth, color='blue')
            self.L_s_toe_line,      = plt.plot([self.pose[19*3], self.pose[20*3]], [self.pose[19*3+1], self.pose[20*3+1]], linewidth=self.linewidth, color='blue')
        # Left Small Toe
        elif "Line2D(_line44)" in str(gco):
            self.pose[20*3], self.pose[20*3+1] = x,y
            self.L_s_toe_line.remove()      
            self.L_s_toe_line,      = plt.plot([self.pose[19*3], self.pose[20*3]], [self.pose[19*3+1], self.pose[20*3+1]], linewidth=self.linewidth, color='blue')
        # Left Heel Toe
        elif "Line2D(_line45)" in str(gco):
            self.pose[21*3], self.pose[21*3+1] = x,y
            self.L_heel_line.remove()            
            self.L_heel_line,       = plt.plot([self.pose[14*3], self.pose[21*3]], [self.pose[14*3+1], self.pose[21*3+1]], linewidth=self.linewidth, color='blue')
        # Right Big Toe
        elif "Line2D(_line46)" in str(gco):
            self.pose[22*3], self.pose[22*3+1] = x,y
            self.R_b_toe_line.remove()            
            self.R_s_toe_line.remove() 
            self.R_b_toe_line,      = plt.plot([self.pose[11*3], self.pose[22*3]], [self.pose[11*3+1], self.pose[22*3+1]], linewidth=self.linewidth, color='cyan')
            self.R_s_toe_line,      = plt.plot([self.pose[22*3], self.pose[23*3]], [self.pose[22*3+1], self.pose[23*3+1]], linewidth=self.linewidth, color='cyan')
        # Right Small Toe
        elif "Line2D(_line47)" in str(gco):
            self.pose[23*3], self.pose[23*3+1] = x,y
            self.R_s_toe_line.remove()            
            self.R_s_toe_line,      = plt.plot([self.pose[22*3], self.pose[23*3]], [self.pose[22*3+1], self.pose[23*3+1]], linewidth=self.linewidth, color='cyan')
        # Right Heel Toe
        elif "Line2D(_line48)" in str(gco):
            self.pose[24*3], self.pose[24*3+1] = x,y
            self.R_heel_line.remove()            
            self.R_heel_line,       = plt.plot([self.pose[11*3], self.pose[24*3]], [self.pose[11*3+1], self.pose[24*3+1]], linewidth=self.linewidth, color='cyan')
            
        plt.xlim(0, self.xlim)
        plt.ylim(0, self.ylim)
        plt.draw()

    def onpick(self, event):
        global gco
        gco = event.artist
        ind = event.ind[0]

    def release(self, event):
        global gco
        gco = None

    def save(self, save_path, json_file):

        def get_distance(x1, y1, x2, y2):
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Invert Pose
        for i in range(len(self.pose)):
            if i%3==0:
                self.pose[i] = round(self.pose[i], 3)
            elif i%3==1:
                self.pose[i] -= self.ylim
                self.pose[i] *= -1
                self.pose[i] = round(self.pose[i], 3)

        # modified joint --> update confidence rate
        for i in range(0, len(self.pose), 3):
            distance = get_distance(self.pose[i], self.pose[i+1], self.original_pose[i], self.original_pose[i+1])
            if distance > 0.01:
                self.pose[i+2] = self.confidence_rate
        
        # Save
        json_file['people'][0]['pose_keypoints_2d'] = self.pose
        with open(save_path, 'w') as f:
            json.dump(json_file ,f)
        print('Saved {}'.format(save_path))

def main():
    index = 100
    pose_path = "./data/json/motion1_json/test_cam0_000000000{}_keypoints.json".format(index)
    save_path = "./data/fixed_json/motion1_json/test_cam0_000000000{}_keypoints.json".format(index)
    image_path = "./data/images/test_cam0__{}.jpg".format(index)

    # Read JSON
    with open(pose_path, 'r') as f:
        json_file = json.load(f)
    pose = json_file['people'][0]['pose_keypoints_2d']

    # Run PoseEditor
    pe = PoseEditor(pose, image_path, xlim=384, ylim=512)
    pe.save(save_path, json_file)


if __name__=='__main__':
    main()