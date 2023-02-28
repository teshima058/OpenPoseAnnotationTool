import json
from pycocotools.coco import COCO
import tqdm

input_path = r"D:\Downloads\openpose\OpenPoseEditor\tmp\person_keypoints_val2017.json"          #Input  : coco format       (keypoints num : 17)
out_path   = r"D:\Downloads\openpose\OpenPoseEditor\tmp\keypoints18"                            #Output : format for editor (keypoints num : 18 (neck is added))

cc = COCO(input_path)

for j in tqdm.tqdm(cc.getImgIds()):
    anno_ids = cc.getAnnIds(j)
    if len(anno_ids) == 0:
        tmp = {"version":1.2,"people":[{"pose_keypoints_2d":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
        for k in range(54):
            tmp['people'][0]['pose_keypoints_2d'].append(1)
        with open(out_path + '/' + str(j)  + '.json', 'w') as f:
            json.dump(tmp,f,indent=2,ensure_ascii=False)
        continue
    Ann = cc.loadAnns(anno_ids[0])
    l = len(Ann[0]['keypoints'])
    if(l==51):
        #make neck annotation
        neck_x = (Ann[0]['keypoints'][5*3] + Ann[0]['keypoints'][6*3])/2
        neck_y = (Ann[0]['keypoints'][5*3+1] + Ann[0]['keypoints'][6*3+1])/2
        neck_s = (Ann[0]['keypoints'][5*3+2] + Ann[0]['keypoints'][6*3+2])/2
        
        Ann[0]['keypoints'].append(neck_x)
        Ann[0]['keypoints'].append(neck_y)
        Ann[0]['keypoints'].append(neck_s)

    elif(0<l<51):
        while(1):
            Ann[0]['keypoints'].append(1)
            if(len(Ann[0]['keypoints'])>=54):
                break
    
    elif(l>51):
        while(1):
            Ann[0]['keypoints'].pop(-1)
            if(len(Ann[0]['keypoints'])<=54):
                break

    elif(l == 0):
        Ann_tmp = cc.loadAnns(j-1)
        Ann[0]['keypoints'] = Ann_tmp[0]["keypoints"]

    tmp = {"version":1.2,"people":[{"pose_keypoints_2d":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
    tmp['people'][0]['pose_keypoints_2d'] = Ann[0]['keypoints']
    with open(out_path + '/' + str(j)  + '.json', 'w') as f:
        json.dump(tmp,f,indent=2,ensure_ascii=False)

