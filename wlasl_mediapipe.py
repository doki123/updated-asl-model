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

      # print(results.face_landmarks.landmark)


      ###
      # face = {}
      # if results.face_landmarks is not None: 
      #   for index, x in enumerate(results.face_landmarks.landmark): # since I couldn't find the names for each face landmark, I'm using the index instead
      #       if index in face_simp: 
      #         face['FACE_' + str(index) + '_X' + str(count)] = x.x # example: rather than CHEEKBONE_X it becose FACE_12_X, and so on
      #         face['FACE_' + str(index) + '_Y' + str(count)] = x.y
      #         face['FACE_' + str(index) + '_Z' + str(count)] = x.z
                
      #   full_data.update(face)
      
      # else: 
      #    print('no face detected')
      #    for indiv_frame in range(0, int(total_frames), 1):
      #       for face_landmark in range(0, len(face_simp), 1):
      #         face['FACE_' + str(indiv_frame) + '_X' + str(face_landmark)] = None
      #         face['FACE_' + str(indiv_frame) + '_Y' + str(face_landmark)] = None
      #         face['FACE_' + str(indiv_frame) + '_Z' + str(face_landmark)] = None

              ###


   #####   
      # if results.pose_world_landmarks is not None: 
      #   pose = { 
      #     'NOSE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x, 
      #     'NOSE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y,
      #     'NOSE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].z,
          
      #     'LEFT_EYE_INNER_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].x, 
      #     'LEFT_EYE_INNER_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].y,
      #     'LEFT_EYE_INNER_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_INNER].z,
            
      #     'LEFT_EYE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x, 
      #     'LEFT_EYE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y,
      #     'LEFT_EYE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].z,
            
      #     'LEFT_EYE_OUTER_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].x, 
      #     'LEFT_EYE_OUTER_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].y,
      #     'LEFT_EYE_OUTER_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE_OUTER].z,
            
      #     'RIGHT_EYE_INNER_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_INNER].x, 
      #     'RIGHT_EYE_INNER_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_INNER].y,
      #     'RIGHT_EYE_INNER_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_INNER].z,
            
      #     'RIGHT_EYE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x, 
      #     'RIGHT_EYE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y,
      #     'RIGHT_EYE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].z,
            
      #     'RIGHT_EYE_OUTER_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].x, 
      #     'RIGHT_EYE_OUTER_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].y,
      #     'RIGHT_EYE_OUTER_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER].z,
            
      #     'LEFT_EAR_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
      #     'LEFT_EAR_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y,
      #     'LEFT_EAR_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].z,
            
      #     'RIGHT_EAR_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x, 
      #     'RIGHT_EAR_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y,
      #     'RIGHT_EAR_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].z,
            
      #     'MOUTH_LEFT_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x, 
      #     'MOUTH_LEFT_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y,
      #     'MOUTH_LEFT_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].z,
            
      #     'MOUTH_RIGHT_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x, 
      #     'MOUTH_RIGHT_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y,
      #     'MOUTH_RIGHT_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].z,
            
      #     'LEFT_SHOULDER_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x, 
      #     'LEFT_SHOULDER_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y,
      #     'LEFT_SHOULDER_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z,
            
      #     'RIGHT_SHOULDER_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, 
      #     'RIGHT_SHOULDER_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y,
      #     'RIGHT_SHOULDER_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z,
            
      #     'LEFT_ELBOW_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x, 
      #     'LEFT_ELBOW_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y,
      #     'LEFT_ELBOW_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z,
            
      #     'RIGHT_ELBOW_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x, 
      #     'RIGHT_ELBOW_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y,
      #     'RIGHT_ELBOW_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].z,
            
      #     'RIGHT_WRIST_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x, 
      #     'RIGHT_WRIST_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y,
      #     'RIGHT_WRIST_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z,
            
      #     'LEFT_WRIST_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x, 
      #     'LEFT_WRIST_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y,
      #     'LEFT_WRIST_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z,
            
      #     'LEFT_PINKY_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].x, 
      #     'LEFT_PINKY_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].y,
      #     'LEFT_PINKY_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].z,
            
      #     'RIGHT_PINKY_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].x, 
      #     'RIGHT_PINKY_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].y,
      #     'RIGHT_PINKY_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].z,
            
      #     'LEFT_INDEX_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].x, 
      #     'LEFT_INDEX_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].y,
      #     'LEFT_INDEX_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].z,
            
      #     'RIGHT_INDEX_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x, 
      #     'RIGHT_INDEX_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y,
      #     'RIGHT_INDEX_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].z,
            
      #     'RIGHT_THUMB_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x, 
      #     'RIGHT_THUMB_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y,
      #     'RIGHT_THUMB_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].z,
            
      #     'LEFT_THUMB_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].x, 
      #     'LEFT_THUMB_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].y,
      #     'LEFT_THUMB_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].z,
            
      #     'LEFT_HIP_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x, 
      #     'LEFT_HIP_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y,
      #     'LEFT_HIP_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].z,
            
      #     'RIGHT_HIP_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x, 
      #     'RIGHT_HIP_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y,
      #     'RIGHT_HIP_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z,
            
      #     'RIGHT_KNEE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x, 
      #     'RIGHT_KNEE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y,
      #     'RIGHT_KNEE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].z,
            
      #     'LEFT_KNEE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x, 
      #     'LEFT_KNEE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y,
      #     'LEFT_KNEE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].z,
            
      #     'LEFT_ANKLE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].x, 
      #     'LEFT_ANKLE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].y,
      #     'LEFT_ANKLE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].z,
            
      #     'RIGHT_ANKLE_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].x, 
      #     'RIGHT_ANKLE_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].y,
      #     'RIGHT_ANKLE_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].z,
            
      #     'RIGHT_HEEL_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL].x, 
      #     'RIGHT_HEEL_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL].y,
      #     'RIGHT_HEEL_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL].z,
            
      #     'LEFT_HEEL_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL].x, 
      #     'LEFT_HEEL_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL].y,
      #     'LEFT_HEEL_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL].z,
            
      #     'LEFT_FOOT_INDEX_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].x, 
      #     'LEFT_FOOT_INDEX_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].y,
      #     'LEFT_FOOT_INDEX_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].z,
            
      #     'RIGHT_FOOT_INDEX_X' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].x, 
      #     'RIGHT_FOOT_INDEX_Y' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].y,
      #     'RIGHT_FOOT_INDEX_Z' + str(count): results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].z,
            
      # }

      # else: 
      #    print('no body detected')



      
      # full_data.update(pose)

      #####
      # pose_df = pd.DataFrame(pose, index=[0])

# note: see if asl is affected by whether the left hand / right hand is used

      if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
        if results.right_hand_landmarks is not None:
            # right_data = {
            #           'WRIST_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x,
            #           'WRIST_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y,
            #           'WRIST_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z,


            #           'THUMB_CMC_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x,
            #           'THUMB_CMC_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y,
            #           'THUMB_CMC_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].z,

            #           'THUMB_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x,
            #           'THUMB_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y,
            #           'THUMB_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].z,

            #           'THUMB_IP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x,
            #           'THUMB_IP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y,
            #           'THUMB_IP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].z,

            #           'THUMB_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
            #           'THUMB_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y,
            #           'THUMB_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].z,

            #           'INDEX_FINGER_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x,
            #           'INDEX_FINGER_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y,
            #           'INDEX_FINGER_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z,

            #           'INDEX_FINGER_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x,
            #           'INDEX_FINGER_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y,
            #           'INDEX_FINGER_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z,

            #           'INDEX_FINGER_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x,
            #           'INDEX_FINGER_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y,
            #           'INDEX_FINGER_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z,

            #           'INDEX_FINGER_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
            #           'INDEX_FINGER_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y,
            #           'INDEX_FINGER_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z,

            #           'MIDDLE_FINGER_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x,
            #           'MIDDLE_FINGER_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y,
            #           'MIDDLE_FINGER_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z,

            #           'MIDDLE_FINGER_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x,
            #           'MIDDLE_FINGER_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y,
            #           'MIDDLE_FINGER_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z,

            #           'MIDDLE_FINGER_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x,
            #           'MIDDLE_FINGER_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y,
            #           'MIDDLE_FINGER_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z,

            #           'MIDDLE_FINGER_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x,
            #           'MIDDLE_FINGER_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y,
            #           'MIDDLE_FINGER_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z,

            #           'RING_FINGER_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x,
            #           'RING_FINGER_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y,
            #           'RING_FINGER_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z,

            #           'RING_FINGER_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x,
            #           'RING_FINGER_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y,
            #           'RING_FINGER_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z,

            #           'RING_FINGER_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x,
            #           'RING_FINGER_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y,
            #           'RING_FINGER_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z,

            #           'RING_FINGER_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x,
            #           'RING_FINGER_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y,
            #           'RING_FINGER_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z,

            #           'PINKY_MCP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x,
            #           'PINKY_MCP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y,
            #           'PINKY_MCP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].z,

            #           'PINKY_PIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x,
            #           'PINKY_PIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y,
            #           'PINKY_PIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].z,

            #           'PINKY_DIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x,
            #           'PINKY_DIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y,
            #           'PINKY_DIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].z,

            #           'PINKY_TIP_X'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x,
            #           'PINKY_TIP_Y'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y,
            #           'PINKY_TIP_Z'+str(count): results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].z,

            #       }

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
            # left_data = {
            #           'WRIST_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x,
            #           'WRIST_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y,
            #           'WRIST_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z,


            #           'THUMB_CMC_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].x,
            #           'THUMB_CMC_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].y,
            #           'THUMB_CMC_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_CMC].z,

            #           'THUMB_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].x,
            #           'THUMB_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].y,
            #           'THUMB_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP].z,

            #           'THUMB_IP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].x,
            #           'THUMB_IP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].y,
            #           'THUMB_IP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP].z,

            #           'THUMB_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
            #           'THUMB_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].y,
            #           'THUMB_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].z,

            #           'INDEX_FINGER_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x,
            #           'INDEX_FINGER_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y,
            #           'INDEX_FINGER_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z,

            #           'INDEX_FINGER_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x,
            #           'INDEX_FINGER_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y,
            #           'INDEX_FINGER_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z,

            #           'INDEX_FINGER_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x,
            #           'INDEX_FINGER_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y,
            #           'INDEX_FINGER_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z,

            #           'INDEX_FINGER_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
            #           'INDEX_FINGER_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y,
            #           'INDEX_FINGER_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z,

            #           'MIDDLE_FINGER_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x,
            #           'MIDDLE_FINGER_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y,
            #           'MIDDLE_FINGER_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z,

            #           'MIDDLE_FINGER_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x,
            #           'MIDDLE_FINGER_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y,
            #           'MIDDLE_FINGER_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z,

            #           'MIDDLE_FINGER_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x,
            #           'MIDDLE_FINGER_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y,
            #           'MIDDLE_FINGER_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z,

            #           'MIDDLE_FINGER_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x,
            #           'MIDDLE_FINGER_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y,
            #           'MIDDLE_FINGER_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z,

            #           'RING_FINGER_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x,
            #           'RING_FINGER_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y,
            #           'RING_FINGER_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z,

            #           'RING_FINGER_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x,
            #           'RING_FINGER_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y,
            #           'RING_FINGER_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z,

            #           'RING_FINGER_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x,
            #           'RING_FINGER_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y,
            #           'RING_FINGER_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z,

            #           'RING_FINGER_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x,
            #           'RING_FINGER_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y,
            #           'RING_FINGER_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z,

            #           'PINKY_MCP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].x,
            #           'PINKY_MCP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].y,
            #           'PINKY_MCP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP].z,

            #           'PINKY_PIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].x,
            #           'PINKY_PIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].y,
            #           'PINKY_PIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP].z,

            #           'PINKY_DIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].x,
            #           'PINKY_DIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].y,
            #           'PINKY_DIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_DIP].z,

            #           'PINKY_TIP_X'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].x,
            #           'PINKY_TIP_Y'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].y,
            #           'PINKY_TIP_Z'+str(count): results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP].z,

            #       }
 
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
  
  # full_df.reset_index(inplace=True, drop=True)
  # master_df.reset_index(inplace=True, drop=True)

  # duplicates = master_df.index[master_df.index.duplicated()]
  # fdf_dup = full_df.index[full_df.index.duplicated()]

  # print("Duplicate master index values:", duplicates)
  # print("Duplicate full_df index values:", fdf_dup)

  col_names = full_df.columns
  master_df = pd.DataFrame(columns=col_names)

  master_df = pd.concat([master_df, full_df], ignore_index=True)


  print("this is master df")
  print(master_df)


print('fin, full_df')
print(full_df)
  
        # print(full_data)
        # data = pd.DataFrame(full_data)
        # print(data)
        
        
    
        # if results.pose_world_landmarks:  # if data being acquired
        #       print('data')
        #       image_hight, image_width, _ = imgRGB.shape

        #       print(
        #         f'Nose coordinates: ('
        #         f'{results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        #         f'{results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
        #       )    

#         if results.multi_hand_world_landmarks:  # if hand is visible on screen and data is being acquired
#             for results.right_hand_landmarks in results.multi_hand_world_landmarks:

#                 if start_capture:

#                     count += 1

#                     # if image_count % 10 == 0 and image_count != 0:  # makes sure every signs gets 10 data points
#                     #     sign_class += 1
#                     #     break

#                     df.update(data)

#         if count >= 40:
            
#             df.update({'CLASS': sign_class})
#             data_list.append(df)
#             print(df['CLASS'])

#             if image_count % 10 == 0 and image_count != 0:  # makes sure every signs gets 10 data points
#                 sign_class += 1

#             image_count += 1
#             print(df)
#             df = {}  # note: this is an empty dictionary, not a dataframe
#             start_capture = 0
#             count = 0

#         cv2.putText(frame, 'Frame: ' + str(int(count)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         cv2.putText(frame, 'You are on image: ' + str(int(image_count)), (10, 130), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         cv2.putText(frame, 'You are on sign: ' + string.ascii_lowercase[int(sign_class)], (10, 190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         cv2.imshow('Frame', frame)
#         key = cv2.waitKey(1)

#         if key & 0xFF == ord('q'):
#             data = pd.DataFrame(data_list, index_col=0)
#             data.to_csv('world_landmark_hands/world_handtracking_data2.csv')
#             break

#         if key == ord('c'):
#             start_capture = 1
#             count = 0
#             df = {}

#         if key == ord('a'):  # if you already have signs and need to add more, start directly at the sign you need to add and then add that in
#             existing_data = pd.read_csv('world_landmark_hands/world_handtracking_data2.csv')
#             data = pd.DataFrame(data_list)
#             print(existing_data['CLASS'])
#             new_data = existing_data.append(data)
#             print(new_data['CLASS'])
#             new_data.to_csv('world_landmark_hands/world_handtracking_data2.csv')
#             break

#     # Break the loop
#     else:
#         break

cap.release()
# cv2.destroyAllWindows()
