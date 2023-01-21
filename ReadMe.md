# OpenPose Annotation Tool
This is a GUI to fix OpenPose mistakes.
## Install
This tool was developed in Python 3.7. <br>
You can install packages with the following command

``` pip install -r requirements.txt ```
## Usage
1. Run ``` PoseEditorGUI.py``` in the GUI folder.  
    - ``` python GUI\PoseEditorGUI.py ```
2. Press ``` Select Save Directory ``` button and choose an empty folder.
    - The fixed json files will be saved here. 
3. Press ``` Open JSON Directory ``` button and select the folder with the json files you want to modify. 
4. Press ```Open Image Directory ``` button and select the folder with the images used in OpenPose.
    - If you don't have any images, you can use ``` utils/video2imgs.py ``` to cut them out of the video. 
5. Fix the joints that are being detected incorrectly.
    - You can switch images from the list of files on the left side of the window.
    - Switching files will automatically save the file.
    - Reset to Original Pose by pressing the ``` Reset to Original Pose ``` button.
    - Go back one position by pressing the ``` Go back ``` button
    - You can copy the pose information by right-clicking on the file and pressing ``` Copy ```.
    - You can paste the copied pose into the file by right-clicking on the file and pressing ``` Right click``` and ``` Paste ```.

## Citation
```
@InProceedings{Kitamura_2022_WACV,
    author    = {Kitamura, Takumi and Teshima, Hitoshi and Thomas, Diego and Kawasaki, Hiroshi},
    title     = {Refining OpenPose With a New Sports Dataset for Robust 2D Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {672-681}
}
```