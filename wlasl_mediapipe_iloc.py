#https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae?usp=sharing
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker


# WRIST_X2  WRIST_Y2      WRIST_Z2  THUMB_CMC_X2  THUMB_CMC_Y2  ...  PINKY_DIP_Y2  PINKY_DIP_Z2  PINKY_TIP_X2  PINKY_TIP_Y2  PINKY_TIP_Z2
# 0  0.592427  0.840359  8.885850e-08      0.619216      0.828645  ...      1.018375     -0.012893      0.610086      1.033212     -0.011975

# [1 rows x 79443 columns]
# RIGHT: 35 LEFT: 0
# expected: 35 * 21 * 3 = 2205 
# 77.0 77.0
# 
# nothing
#    WRIST_X2  WRIST_Y2      WRIST_Z2  THUMB_CMC_X2  THUMB_CMC_Y2  ...  PINKY_DIP_Y2  PINKY_DIP_Z2  PINKY_TIP_X2  PINKY_TIP_Y2  PINKY_TIP_Z2
# 0  0.592427  0.840359  8.885850e-08      0.619216      0.828645  ...      1.018375     -0.012893      0.610086      1.033212     -0.011975

# [1 rows x 270963 columns]
# RIGHT: 31 LEFT: 0
# expected = 31 * 21 * 3 = 1953

# 56.0 56.0
# nothing
#    WRIST_X2  WRIST_Y2      WRIST_Z2  THUMB_CMC_X2  THUMB_CMC_Y2  ...  PINKY_DIP_Y2  PINKY_DIP_Z2  PINKY_TIP_X2  PINKY_TIP_Y2  PINKY_TIP_Z2
# 0  0.429808  0.625976 -1.387361e-07      0.451126      0.613007  ...      1.018375     -0.012893      0.610086      1.033212     -0.011975

# [1 rows x 438795 columns]
# RIGHT: 55 LEFT: 0
# 54.0 54.0
# zsh: killed      
# (venv) shrutiladiwala@Shrutis-MacBook-Air updated-asl-model % 

# NOTE: WITHOUT SIMPLIFYING FACE THERE WERE 120K LANDMARKS. NOW THERE ARE 43461. STILL A LOT. IS THIS ACCEPTABLE?
# NOTE: SHOULD BE 43462. NEED TO ADD CLASSES OF SIGNS. 

# Left Eyebrow = [0->9]
# right Eyebrow = [10->19]
# Left Eye = [20->35]
# Right Eye = [36->51]
# Iner Lip = [52->71]
# outer Lip = [72->91]
# Face Boundary = [92->127]
# Left iris = [128->132]
# Right iris = [133->137]
# Nose = [138->140] 

# so for the structure of the dataset, we want it to be so:
# one video will be one row of data 
# each row will have all the collumns from each frame
# example: 0 [FRAME 1] [FRAME 2] [FRAME 3] [CLASS SIGN]
        #  1 [FRAME 1] [FRAME 2] [FRAME 3] [CLASS SIGN]
# each column should follow naming structure "WRIST_" + FRAME NUMBER


# TODO: FIRST MAKE IT SO THAT A SINGLE VIDEO GETS PROCESSED AND ALL FRAMES GET ADDED SIDE BY SIDE 
# NEXT MAKE IT NESTED SO THAT IT PROCESSES MULTIPLE VIDEOS
# NEXT FIGURE OUT HOW TO INTEGRATE THAT INTO THE PREVIOUS DATASETS
# TODO: ADD THE FRAMECOUNT VARIABLE TO THE END OF ALL COLUMN NAMES

# mp_holistic = mp.solutions.holistic

from itertools import chain
from re import L
# import time
import  string
import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd


rh_inf = 0 
lf_inf = 0 
abs_path = '/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/updated-asl-model/static/kaggle_wsasl_data/'
wanted_files = os.listdir(abs_path + 'wanted_videos')

# cap = cv2.VideoCapture(abs_path + 'wanted_videos/38524.mp4')
# total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print(total_frames)

# frame_list = []

# print(cap.get(7)) # amount of frames in video

# for x in range(0, int(cap.get(7)) + 1, 1):
#   _ , frame = cap.read()
#   frame_list.append(frame)
 
  # print(images)

# print(frame_list)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

image_count = 0  # tracks the amount of signs that has been captured
df = {}
data_list = []
count = 0  # frame number --> used so that the columns in data all have different names


full_data = {}
full_df = pd.DataFrame()


face_simp = [] # there are way too many landmarks in face, slows down processing time considerably
# this for now only has eyebrows, eyes, inner/outer lip, facial boundary, irises, and nose

def range_maker(start, end): 
   simp = []

   for x in range(start, end + 1, 1):
       simp.append(x)

       if x == end:
           return simp
       
face_simp.append(range_maker(0, 140))
# face_simp.append(range_maker(10, 140))
face_simp = list(chain(*face_simp)) # flattens list

# for indiv_video in wanted_files[:2]:

master_df = pd.DataFrame()
master_dict = {}

# wanted_files = ['/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/updated-asl-model/static/kaggle_wsasl_data/wanted_videos/38525.mp4']
for indiv_video in wanted_files[:5]:
  count = 0
  cap = cv2.VideoCapture(abs_path + 'wanted_videos/' + indiv_video)

  print(indiv_video)
  # cap = cv2.VideoCapture(indiv_video)
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

  print(cap.get(7), total_frames)

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
    
    else:        
      imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = mp_holistic.Holistic(static_image_mode=True).process(imgRGB)

# note: see if asl is affected by whether the left hand / right hand is used

      if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
        if results.right_hand_landmarks is not None:

            right_data = {
                      'RIGHT_WRIST_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x,
                      'RIGHT_WRIST_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y,
                      'RIGHT_WRIST_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z,


                      'RIGHT_THUMB_CMC_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x,
                      'RIGHT_THUMB_CMC_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y,
                      'RIGHT_THUMB_CMC_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].z,

                      'RIGHT_THUMB_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x,
                      'RIGHT_THUMB_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y,
                      'RIGHT_THUMB_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].z,

                      'RIGHT_THUMB_IP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x,
                      'RIGHT_THUMB_IP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y,
                      'RIGHT_THUMB_IP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].z,

                      'RIGHT_THUMB_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
                      'RIGHT_THUMB_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y,
                      'RIGHT_THUMB_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].z,

                      'RIGHT_INDEX_FINGER_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x,
                      'RIGHT_INDEX_FINGER_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y,
                      'RIGHT_INDEX_FINGER_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z,

                      'RIGHT_INDEX_FINGER_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x,
                      'RIGHT_INDEX_FINGER_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y,
                      'RIGHT_INDEX_FINGER_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z,

                      'RIGHT_INDEX_FINGER_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x,
                      'RIGHT_INDEX_FINGER_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y,
                      'RIGHT_INDEX_FINGER_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z,

                      'RIGHT_INDEX_FINGER_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                      'RIGHT_INDEX_FINGER_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y,
                      'RIGHT_INDEX_FINGER_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z,

                      'RIGHT_MIDDLE_FINGER_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x,
                      'RIGHT_MIDDLE_FINGER_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y,
                      'RIGHT_MIDDLE_FINGER_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z,

                      'RIGHT_MIDDLE_FINGER_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x,
                      'RIGHT_MIDDLE_FINGER_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y,
                      'RIGHT_MIDDLE_FINGER_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z,

                      'RIGHT_MIDDLE_FINGER_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x,
                      'RIGHT_MIDDLE_FINGER_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y,
                      'RIGHT_MIDDLE_FINGER_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z,

                      'RIGHT_MIDDLE_FINGER_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x,
                      'RIGHT_MIDDLE_FINGER_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y,
                      'RIGHT_MIDDLE_FINGER_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z,

                      'RIGHT_RING_FINGER_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x,
                      'RIGHT_RING_FINGER_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y,
                      'RIGHT_RING_FINGER_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z,

                      'RIGHT_RING_FINGER_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x,
                      'RIGHT_RING_FINGER_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y,
                      'RIGHT_RING_FINGER_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z,

                      'RIGHT_RING_FINGER_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x,
                      'RIGHT_RING_FINGER_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y,
                      'RIGHT_RING_FINGER_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z,

                      'RIGHT_RING_FINGER_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x,
                      'RIGHT_RING_FINGER_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y,
                      'RIGHT_RING_FINGER_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z,

                      'RIGHT_PINKY_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x,
                      'RIGHT_PINKY_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y,
                      'RIGHT_PINKY_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].z,

                      'RIGHT_PINKY_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x,
                      'RIGHT_PINKY_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y,
                      'RIGHT_PINKY_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].z,

                      'RIGHT_PINKY_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x,
                      'RIGHT_PINKY_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y,
                      'RIGHT_PINKY_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].z,

                      'RIGHT_PINKY_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x,
                      'RIGHT_PINKY_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y,
                      'RIGHT_PINKY_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].z,

                  }

            # right_df = pd.DataFrame(right_data, index=[0])
            full_data.update(right_data)

            rh_inf += 1
            # print(right_df)
        elif results.left_hand_landmarks is not None:
 
            left_data = {
                      'LEFT_WRIST_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x,
                      'LEFT_WRIST_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y,
                      'LEFT_WRIST_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z,


                      'LEFT_THUMB_CMC_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x,
                      'LEFT_THUMB_CMC_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y,
                      'LEFT_THUMB_CMC_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].z,

                      'LEFT_THUMB_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x,
                      'LEFT_THUMB_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y,
                      'LEFT_THUMB_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].z,

                      'LEFT_THUMB_IP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x,
                      'LEFT_THUMB_IP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y,
                      'LEFT_THUMB_IP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].z,

                      'LEFT_THUMB_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
                      'LEFT_THUMB_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y,
                      'LEFT_THUMB_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].z,

                      'LEFT_INDEX_FINGER_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x,
                      'LEFT_INDEX_FINGER_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y,
                      'LEFT_INDEX_FINGER_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z,

                      'LEFT_INDEX_FINGER_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x,
                      'LEFT_INDEX_FINGER_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y,
                      'LEFT_INDEX_FINGER_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z,

                      'LEFT_INDEX_FINGER_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x,
                      'LEFT_INDEX_FINGER_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y,
                      'LEFT_INDEX_FINGER_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z,

                      'LEFT_INDEX_FINGER_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                      'LEFT_INDEX_FINGER_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y,
                      'LEFT_INDEX_FINGER_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z,

                      'LEFT_MIDDLE_FINGER_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x,
                      'LEFT_MIDDLE_FINGER_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y,
                      'LEFT_MIDDLE_FINGER_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z,

                      'LEFT_MIDDLE_FINGER_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x,
                      'LEFT_MIDDLE_FINGER_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y,
                      'LEFT_MIDDLE_FINGER_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z,

                      'LEFT_MIDDLE_FINGER_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x,
                      'LEFT_MIDDLE_FINGER_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y,
                      'LEFT_MIDDLE_FINGER_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z,

                      'LEFT_MIDDLE_FINGER_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x,
                      'LEFT_MIDDLE_FINGER_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y,
                      'LEFT_MIDDLE_FINGER_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z,

                      'LEFT_RING_FINGER_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x,
                      'LEFT_RING_FINGER_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y,
                      'LEFT_RING_FINGER_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z,

                      'LEFT_RING_FINGER_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x,
                      'LEFT_RING_FINGER_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y,
                      'LEFT_RING_FINGER_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z,

                      'LEFT_RING_FINGER_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x,
                      'LEFT_RING_FINGER_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y,
                      'LEFT_RING_FINGER_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z,

                      'LEFT_RING_FINGER_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x,
                      'LEFT_RING_FINGER_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y,
                      'LEFT_RING_FINGER_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z,

                      'LEFT_PINKY_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x,
                      'LEFT_PINKY_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y,
                      'LEFT_PINKY_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].z,

                      'LEFT_PINKY_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x,
                      'LEFT_PINKY_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y,
                      'LEFT_PINKY_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].z,

                      'LEFT_PINKY_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x,
                      'LEFT_PINKY_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y,
                      'LEFT_PINKY_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].z,

                      'LEFT_PINKY_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x,
                      'LEFT_PINKY_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y,
                      'LEFT_PINKY_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].z,

                  }
 
            # left_df = pd.DataFrame(left_data)
            full_data.update(left_data)

            lf_inf += 1
      
      fdf = pd.DataFrame(full_data, index=[0])
      full_data = {}
      
    count += 1

    full_df = pd.concat([fdf, full_df], axis=1)
    fdf = pd.DataFrame()
  
  # print(full_df)
  expected_col = (rh_inf * 63) + (lf_inf * 63)
  print('RIGHT:', rh_inf, 'LEFT:', lf_inf, 'EXPECTED:', expected_col)
  

  count = rh_inf = lf_inf = 0

  print("this is full_df")
  print(full_df)

  print()

  print("this is master_df")
  print(master_df)
  

  col_names = full_df.columns

  master_df.loc[len(master_df.index)] = full_df

  print("this is master df")
  print(master_df)


print('fin, full_df')
print(full_df)


cap.release()
# cv2.destroyAllWindows()
