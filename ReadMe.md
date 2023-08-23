## Openpose annotation tool (COCO format : 17 keypoints)

The COCO dataset has **17 keypoints** when downloaded. \
On the other hand, our tool edits **18 keypoints** by adding the neck between the two shoulders. (for tool visibility and simplicity). \
So the process is ```17 keypoints -> 18 keypoints(neck added) -> 17 keypoints```.

### 1. Edit 17 keypoints in ``python utils/editCOCO17to18.py``.
- Create a folder for output.
- In your code, replace input_path and output_path with your own.
- In the Output folder, an annotation file (json) will be created for each image.

### 2. In ``python GUI/PoseEditor18GUI.py``, fix the wrong annotations.
- See the original ReadMe for details.

### 3. Change back to coco format in ``python utils/edit18toCOCO17.py``.
- In the code, edit ori_path , aftGUI_path , and out_path .
- The code works as follows: the original coco format json file specified in ori_path is modified with the result of your editor (aftGUI_path) and written to out_path.

### Notes.
- Currently, only one annotation per image can be edited.

### Citation
If our code is helpful, please kindly cite the following paper:
```
@INPROCEEDINGS{Kitamura2022:WACVws22,
   author={Takumi Kitamura and Hitoshi Teshima and Diego Thomas and Hiroshi Kawasaki },
   title={Refining OpenPose with a new sports dataset for robust 2D pose estimation},
   booktitle={2022 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVws)},
   year={2022},
}
```

<!-- ## Openpose annotation tool (17 keypoints)

COCO datasetはダウンロード時には17個のkeypointsを持っています.
一方で私達のツールでは、2つの肩の間に首を加えた18keypointsを編集しています.(ツール視認性及び簡易性の為)
そのため、17 keypoints → 18 keypoints(neck added) → 17 keypoints というプロセスを辿ります.

### 1.   ```python OpenPoseEditor/utils/editCOCO17to18.py```で、17keypointsにします. \
- output用のフォルダを作成してください.
- コード内で、input_pathとoutput_pathをあなたのものに書き換えてください.
- Outputフォルダには、画像ごとにannotationファイル(json)が作成されます.

### 2. ```python OpenPoseEditor/GUI/PoseEditor18GUI.py```で、間違ったannotationを修正.
詳しくは、オリジナルのReadMeを参照.

### 3. ```python OpenPoseEditor/utils/edit18toCOCO17.py```で,cocoフォーマットへと戻す.
- コード内で、ori_path , aftGUI_path , out_pathを編集してください.
- コードの動作イメージとしては、ori_pathで指定したオリジナルのcocoフォーマットのjsonファイルに、あなたのeditorでの編集結果(aftGUI_path)で修正して、out_pathへと書き出します.

### 注意
- 現在のところ、一つの画像につき、一人のannotationしか編集できません.
-->
