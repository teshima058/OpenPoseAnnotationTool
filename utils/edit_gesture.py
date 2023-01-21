import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms

import numpy as np
import argparse
import csv

from plot import Plot


class GestureEditor():
    def __init__(self, pose, xlim, ylim, degree=0, linewidth=5.0):
        gco = None
        self.xlim, self.ylim = xlim, ylim
        self.linewidth = linewidth
        self.base = pyplot.gca().transData
        self.rot = transforms.Affine2D().rotate_deg(degree)
        self.pose = pose

        self.neck_line,       = plt.plot([self.pose[0], self.pose[3]],  [self.pose[1],  self.pose[4]], "b", transform=self.rot+self.base,linewidth=self.linewidth) #sholder 1
        self.Lshoulder_line,  = plt.plot([self.pose[3], self.pose[6]],  [self.pose[4],  self.pose[7]], "g", transform=self.rot+self.base,linewidth=self.linewidth) #sholder 1
        self.Lelbow_line,     = plt.plot([self.pose[6], self.pose[9]],  [self.pose[7],  self.pose[10]], "r", transform=self.rot+self.base, linewidth=self.linewidth)
        self.Lhand_line,      = plt.plot([self.pose[12],self.pose[9]],  [self.pose[13], self.pose[10]], "c", transform=self.rot+self.base, linewidth=self.linewidth) #arm1-1
        self.Rshoulder_line,  = plt.plot([self.pose[3], self.pose[15]], [self.pose[4],  self.pose[16]], "m", transform=self.rot+self.base, linewidth=self.linewidth) #sholder 2
        self.Relbow_line,     = plt.plot([self.pose[15],self.pose[18]], [self.pose[16], self.pose[19]], "y", transform=self.rot+self.base, linewidth=self.linewidth)
        self.Rhand_line,      = plt.plot([self.pose[21],self.pose[18]], [self.pose[22], self.pose[19]], "k", transform=self.rot+self.base, linewidth=self.linewidth) #arm2-1

        self.head        = plt.plot(pose[0*3], pose[0*3+1], "o", picker=15)
        self.neck        = plt.plot(pose[1*3], pose[1*3+1], "o", picker=15)
        self.Lshoulder   = plt.plot(pose[2*3], pose[2*3+1], "o", picker=15)
        self.Lelbow      = plt.plot(pose[3*3], pose[3*3+1], "o", picker=15)
        self.Lhand       = plt.plot(pose[4*3], pose[4*3+1], "o", picker=15)
        self.Rshoulder   = plt.plot(pose[5*3], pose[5*3+1], "o", picker=15)
        self.Relbow      = plt.plot(pose[6*3], pose[6*3+1], "o", picker=15)
        self.Rhand       = plt.plot(pose[7*3], pose[7*3+1], "o", picker=15)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.connect('motion_notify_event', self.motion)
        plt.connect('pick_event', self.onpick)
        plt.connect('button_release_event', self.release)
        plt.show()

    def motion(self, event):
        global gco
        x = event.xdata
        y = event.ydata
        try:
            gco.set_data(x,y)
        except Exception:
            return

        if   "Line2D(_line7)"  in str(gco): 
            self.pose[0*3], self.pose[0*3+1] = x,y
            self.neck_line.remove()
            self.neck_line, = plt.plot([self.pose[0], self.pose[3]], [self.pose[1], self.pose[4]], "b", transform=self.rot+self.base, linewidth=self.linewidth) #neck
        elif "Line2D(_line8)"  in str(gco): 
            self.pose[1*3], self.pose[1*3+1] = x,y
            self.neck_line.remove()
            self.Lshoulder_line.remove()
            self.Rshoulder_line.remove()
            self.neck_line,       = plt.plot([self.pose[0], self.pose[3]],  [self.pose[1],  self.pose[4]], "b", transform=self.rot+self.base,linewidth=self.linewidth) #sholder 1
            self.Lshoulder_line,  = plt.plot([self.pose[3], self.pose[6]],  [self.pose[4],  self.pose[7]], "g", transform=self.rot+self.base,linewidth=self.linewidth) #sholder 1
            self.Rshoulder_line,  = plt.plot([self.pose[3], self.pose[15]], [self.pose[4],  self.pose[16]], "m", transform=self.rot+self.base, linewidth=self.linewidth) #sholder 2
        elif "Line2D(_line9)"  in str(gco):
            self.pose[2*3], self.pose[2*3+1] = x,y
            self.Lshoulder_line.remove()
            self.Lelbow_line.remove()
            self.Lshoulder_line,  = plt.plot([self.pose[3], self.pose[6]],  [self.pose[4],  self.pose[7]], "g", transform=self.rot+self.base,linewidth=self.linewidth) #sholder 1
            self.Lelbow_line,     = plt.plot([self.pose[6], self.pose[9]],  [self.pose[7],  self.pose[10]], "r", transform=self.rot+self.base, linewidth=self.linewidth)
        elif "Line2D(_line10)" in str(gco):
            self.pose[3*3], self.pose[3*3+1] = x,y
            self.Lelbow_line.remove()
            self.Lhand_line.remove()
            self.Lelbow_line,     = plt.plot([self.pose[6], self.pose[9]],  [self.pose[7],  self.pose[10]], "r", transform=self.rot+self.base, linewidth=self.linewidth)
            self.Lhand_line,      = plt.plot([self.pose[12],self.pose[9]],  [self.pose[13], self.pose[10]], "c", transform=self.rot+self.base, linewidth=self.linewidth) #arm1-1
        elif "Line2D(_line11)" in str(gco):
            self.pose[4*3], self.pose[4*3+1] = x,y
            self.Lhand_line.remove()
            self.Lhand_line,      = plt.plot([self.pose[12], self.pose[9]], [self.pose[13], self.pose[10]], "c", transform=self.rot+self.base, linewidth=self.linewidth) #arm1-1
        elif "Line2D(_line12)" in str(gco): 
            self.pose[5*3], self.pose[5*3+1] = x,y
            self.Rshoulder_line.remove()
            self.Relbow_line.remove()
            self.Rshoulder_line,  = plt.plot([self.pose[3], self.pose[15]], [self.pose[4],  self.pose[16]], "m", transform=self.rot+self.base, linewidth=self.linewidth) #sholder 2
            self.Relbow_line,     = plt.plot([self.pose[15],self.pose[18]], [self.pose[16], self.pose[19]], "y", transform=self.rot+self.base, linewidth=self.linewidth)
        elif "Line2D(_line13)" in str(gco): 
            self.pose[6*3], self.pose[6*3+1] = x,y
            self.Relbow_line.remove()
            self.Rhand_line.remove()
            self.Relbow_line,     = plt.plot([self.pose[15], self.pose[18]], [self.pose[16], self.pose[19]], "y", transform=self.rot+self.base, linewidth=self.linewidth)
            self.Rhand_line,      = plt.plot([self.pose[21], self.pose[18]], [self.pose[22], self.pose[19]], "k", transform=self.rot+self.base, linewidth=self.linewidth) #arm2-1
        elif "Line2D(_line14)" in str(gco): 
            self.pose[7*3], self.pose[7*3+1] = x,y
            self.Rhand_line.remove()
            self.Rhand_line,      = plt.plot([self.pose[21], self.pose[18]], [self.pose[22], self.pose[19]], "k", transform=self.rot+self.base, linewidth=self.linewidth) #arm2-1

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.draw()

    def onpick(self, event):
        global gco
        gco = event.artist
        ind = event.ind[0]

    def release(self, event):
        global gco
        gco = None

def main():
    pose_path = "./pose/look at the big worl.csv"
    save_path = "./created_pose/look at the big worl(modified).csv"
    # pose_path = "./created_pose/test.csv"

    # CSVからポーズ読み取り
    with open(pose_path, "r") as f:
        reader = csv.reader(f)
        poses= []
        for row in reader:
            if row == []: continue
            frame = []
            for i in range(len(row)):
                frame.append(float(row[i]))
            poses.append(frame)

    frame = len(poses)
    print("{} frames".format(frame))

    # 編集するフレーム間を指定
    start_edit_frame = 10
    end_edit_frame = 30

    previous_poses = []
    for i in range(start_edit_frame, end_edit_frame):
        if i == start_edit_frame: 
            print("frame num : {}".format(i))
            GestureEditor(poses[i], (3, 10), (6, 13))
            previous_poses.append(poses[i].copy())
        else: 
            print("frame num : {}".format(i))
            poses[i][3*3] = poses[i-1][3*3]
            poses[i][3*3+1] = poses[i-1][3*3+1]
            poses[i][4*3] = poses[i-1][4*3]
            poses[i][4*3+1] = poses[i-1][4*3+1]
            poses[i][6*3] = poses[i-1][6*3]
            poses[i][6*3+1] = poses[i-1][6*3+1]
            poses[i][7*3] = poses[i-1][7*3]
            poses[i][7*3+1] = poses[i-1][7*3+1]
            GestureEditor(poses[i], (3, 10), (6, 13))
            previous_poses.append(poses[i].copy())

    for i in range(start_edit_frame, end_edit_frame):
        poses[i] = previous_poses[i - start_edit_frame]
    
    # CSVに保存
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(len(poses)):
            writer.writerow(poses[i])
    print("csv saved")

    # MP4に保存
    p = Plot((3, 10), (6, 13))
    anim = p.animate(poses, 1000/12)
    anim.save(save_path[:-4] + ".mp4")
    print("mp4 saved")


if __name__=='__main__':
    main()