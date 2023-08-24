#https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

from re import L
# import time
import  string
import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

abs_path = '/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/updated-asl-model/static/kaggle_wsasl_data/'
wanted_files = os.listdir(abs_path + 'wanted_videos')

cap = cv2.VideoCapture(abs_path + 'wanted_videos/38524.mp4')
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frames)

frame_list = []

print(cap.get(7)) # amount of frames in video

# for x in range(0, int(cap.get(7)) + 1, 1):
#   _ , frame = cap.read()
#   frame_list.append(frame)
 
  # print(images)

# print(frame_list)

image_count = 0  # tracks the amount of signs that has been captured
df = {}
data_list = []
count = 0  # frame number --> used so that the columns in data all have different names


full_data = {}
full_df = pd.DataFrame(index=[0])

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read the video
while(cap.isOpened()):
    
    ret, frame = cap.read()
    # print('starting')
    if ret == False: 
         print('nothing')
         break
        
    
    elif ret == True:
        full_data.update({'hi' + str(count): 'bleh'})
        x = pd.DataFrame(full_data, index=[0])

    count += 1

    yy = pd.concat([x, full_df], axis=1)

print(yy)