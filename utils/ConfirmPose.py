import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

class DisplayPose():
    def __init__(self, pose, image_path, xlim, ylim, linewidth=3.0):
        self.xlim = xlim
        self.ylim = ylim
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
        self.initLine()
        self.initPoint()
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
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
        self.marker_size = 5
        self.picker_size = 5
        self.nose        = plt.plot(self.pose[0*3],  self.pose[0*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='crimson')
        self.neck        = plt.plot(self.pose[1*3],  self.pose[1*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='red')
        self.Rshoulder   = plt.plot(self.pose[2*3],  self.pose[2*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='darkorange')
        self.Relbow      = plt.plot(self.pose[3*3],  self.pose[3*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='goldenrod')
        self.Rhand       = plt.plot(self.pose[4*3],  self.pose[4*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='yellow')
        self.Lshoulder   = plt.plot(self.pose[5*3],  self.pose[5*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='yellowgreen')
        self.Lelbow      = plt.plot(self.pose[6*3],  self.pose[6*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='lawngreen')
        self.Lhand       = plt.plot(self.pose[7*3],  self.pose[7*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='lime')
        self.Midhip      = plt.plot(self.pose[8*3],  self.pose[8*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='orangered')
        self.Rhip        = plt.plot(self.pose[9*3],  self.pose[9*3+1],  marker='o', markersize=self.marker_size, picker=self.picker_size, color='limegreen')
        self.Rknee       = plt.plot(self.pose[10*3], self.pose[10*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='mediumturquoise')
        self.Rankle      = plt.plot(self.pose[11*3], self.pose[11*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='cyan')
        self.Lhip        = plt.plot(self.pose[12*3], self.pose[12*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='deepskyblue')
        self.Lknee       = plt.plot(self.pose[13*3], self.pose[13*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='dodgerblue')
        self.Lankle      = plt.plot(self.pose[14*3], self.pose[14*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='blue')
        self.Reye        = plt.plot(self.pose[15*3], self.pose[15*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='mediumvioletred')
        self.Leye        = plt.plot(self.pose[16*3], self.pose[16*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='deeppink')
        self.Rear        = plt.plot(self.pose[17*3], self.pose[17*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='darkviolet')
        self.Lear        = plt.plot(self.pose[18*3], self.pose[18*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='purple')
        self.LBigtoe     = plt.plot(self.pose[19*3], self.pose[19*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='blue')
        self.LSmalltoe   = plt.plot(self.pose[20*3], self.pose[20*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='blue')
        self.Lheel       = plt.plot(self.pose[21*3], self.pose[21*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='blue')
        self.RBigtoe     = plt.plot(self.pose[22*3], self.pose[22*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='cyan')
        self.RSmalltoe   = plt.plot(self.pose[23*3], self.pose[23*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='cyan')
        self.Rheel       = plt.plot(self.pose[24*3], self.pose[24*3+1], marker='o', markersize=self.marker_size, picker=self.picker_size, color='cyan')

if __name__ == "__main__":
    pose_path = "./data/fixed_json/motion1_json/test_cam0_000000000000_keypoints.json"
    image_path = "./data/images/test_cam0__000.jpg"

    # Read JSON
    with open(pose_path, 'r') as f:
        json_file = json.load(f)
    pose = json_file['people'][0]['pose_keypoints_2d']

    # Run DisplayPose
    dp = DisplayPose(pose, image_path, xlim=384, ylim=512)
