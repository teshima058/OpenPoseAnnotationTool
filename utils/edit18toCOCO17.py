import json
import glob
from pycocotools.coco import COCO
from natsort import natsorted
import os
import sys
import tqdm

ori_path    = r"D:\Downloads\openpose\OpenPoseEditor\tmp\person_keypoints_val2017.json"             #Input  : Json file   just after download     (keypoints num : 17)
aftGUI_path = r"D:\Downloads\openpose\OpenPoseEditor\tmp\output_editor"                          #Input  : Json folder after editing keypoints (keypoints num : 18)

out_path = r"D:\Downloads\openpose\OpenPoseEditor\tmp\person_keypoints_val2017_aft.json"         #Output : Json file   after editing keypoints (keypoints num : 17)


#in1読み込んで(COCOとして?)、in2から一つずつannotetionとってきて、in1の各annに保存する(上書きしてくれる?)
f = open(ori_path,'r')
ccj = json.load(f)

cc = COCO(ori_path)

ccj['annotations'] = []
files = natsorted(glob.glob(aftGUI_path + '/*.json'))
for i,file in tqdm.tqdm(enumerate(files)):
    img_id = int(os.path.splitext(os.path.basename(file))[0])
    #print("img_id : " , img_id)
    anno_ids = cc.getAnnIds(img_id)
    if len(anno_ids) == 0:
        continue
    Ann = cc.loadAnns(anno_ids[0])
    """
    print("i : " , i)
    print("Ann : " , Ann)
    print()
    """

    aft = open(file,'r')
    aftaft = json.load(aft)
    keypoints = aftaft['people'][0]['pose_keypoints_2d'] 
    Ann[0]['keypoints'] = keypoints[:51]
    #ccj['annotations'][i]['keypoints'] = keypoints[:51]
    #ccj['annotations'][anno_id]['keypoints'] = keypoints[:51]
    ccj['annotations'].append(Ann[0])
    aft.close()

fw = open(out_path,'w')
json.dump(ccj,fw,indent=2)

f.close()
fw.close()