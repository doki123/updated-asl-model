# dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=wlasl_class_list.txt
# note: until i can figure out how to deal w/parquet let's try this

import pandas as pd
import os
import shutil 

df = pd.read_json('static/kaggle_wsasl_data/WLASL_v0.3.json')
missing = open('static/kaggle_wsasl_data/missing.txt').read().split('\n')
# print(missing)

# print(df[df["gloss"] == "thank you"]) # thank you, no, yes
video_ids = []
unwanted_videos = []
wanted_signs = ["thank you", "no", "yes"]
for z in wanted_signs:
    for x in df[df["gloss"] == z]["instances"]:
        for y in x:
            # print(y['url'])
            # print(y['video_id'])
            if y['video_id'] not in missing:
                video_ids.append(y['video_id'] + '.mp4')
            else:
                print(y['video_id'], 'does not exist')
            

print(video_ids)

abs_path = '/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/updated-asl-model/static/kaggle_wsasl_data/'

existing_vids = os.listdir(abs_path + 'wanted_videos')

if existing_vids.sort() == video_ids.sort():
    print('good')

for move in video_ids:
    try: 
        shutil.move(abs_path + 'videos/' + move, abs_path + 'wanted_videos')
    except: 
        print('video not found', move)


