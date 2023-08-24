# dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=wlasl_class_list.txt
# note: until i can figure out how to deal w/parquet let's try this

import pandas as pd
import os
import shutil 

df = pd.read_json('static/kaggle_wsasl_data/WLASL_v0.3.json')
missing = open('static/kaggle_wsasl_data/missing.txt').read().split('\n') # list of unavailable videos
# print(missing)

# print(df[df["gloss"] == "thank you"]) # thank you, no, yes
video_ids = [] # list of all video titles
wanted_signs = ["thank you", "no", "yes"]

for signs in wanted_signs:
    for x in df[df["gloss"] == signs]["instances"]:
        for y in x:
            # print(y)
            if y['video_id'] not in missing:
                video_ids.append(y['video_id'] + '.mp4')
                print(y)
            else:
                # print(y['video_id'], 'does not exist')
                pass # this means that the video got deleted/isn't available
            
abs_path = '/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/updated-asl-model/static/kaggle_wsasl_data/'

for move in video_ids:
    try: 
        shutil.move(abs_path + 'videos/' + move, abs_path + 'wanted_videos') # moves all needed videos into the wanted_videos folder
    except: 
        pass # means the video does not exist/can't be moved

existing_vids = os.listdir(abs_path + 'wanted_videos')
print((existing_vids))

if existing_vids.sort() == video_ids.sort():
    print('good') # this means the videos in the wanted folder match everything in the videos_id