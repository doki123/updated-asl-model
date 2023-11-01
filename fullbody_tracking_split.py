from re import L
# import time
import  string
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# this is so that only one hand is detected at any time --> use above code if you want both hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

mp_holistic = mp.solutions.holistic # Holistic model 
mp_drawing = mp.solutions.drawing_utils # Drawing utilities 
mp_drawing_styles = mp.solutions.drawing_styles

# x = pd.read_csv('static/world_handtracking_data2.csv')
# print(x['CLASS'], x['STR_CLASS']), x['MOVING']
# print(x.columns)
# exit()

## Left Eyebrow = [0->9]
# right Eyebrow = [10->19]
# Left Eye = [20->35]
# Right Eye = [36->51]
# Iner Lip = [52->71]
# outer Lip = [72->91]
# Face Boundary = [92->127]
# Left iris = [128->132] try taking out left iris
# Right iris = [133->137] take out right iris
# Nose = [138->140] 

# 1. Pose landmarks: take out 23-downwards (thats left_hip etc, only want torso and basic face)

index_class = int(input('What sign do you want to start from (index of sign; 0-25 is reserved for alphabet!)?'))
str_class = input("what is the sign name?")
body_part = int(input("is this HANDS or FULL TORSO (0 for hands 1 for full)"))
moving = int(input("does it move? 0 for static 1 for moving"))

extra_info = {}

extra_info.update({str_class: [index_class, body_part]})

# for now you can skip this, but make note of it in the future

image_count = 0  # tracks the amount of signs that has been captured
df = {}
start_capture = 0  # when c has been pressed --> signals for camera to start recording
data_list = []
count = 0  # frame number --> used so that the columns in data all have different names

# instead of just doing index of alphabet, index can just refer to all signs in the order they appear
# NOTE: Actually let's reserve index numbers 0-25 for the alphabet for easiness, so 26 should be no and 27 should be yes
# """Index(['Unnamed: 0', 'WRIST_X1', 'WRIST_Y1', 'WRIST_Z1', 'THUMB_CMC_X1',
    #    'THUMB_CMC_Y1', 'THUMB_CMC_Z1', 'THUMB_MCP_X1', 'THUMB_MCP_Y1',
    #    'THUMB_MCP_Z1',
    #    ...
    #    'PINKY_PIP_Z40', 'PINKY_DIP_X40', 'PINKY_DIP_Y40', 'PINKY_DIP_Z40',
    #    'PINKY_TIP_X40', 'PINKY_TIP_Y40', 'PINKY_TIP_Z40', 'CLASS', 'STR_CLASS',
    #    'MOVING'], 

# there's STR_CLASS and CLASS so we good
# okay I've added str_class stuff, now we also need to deal with collecting the upper torso
# but for now "no" is a hand-only movement so it's fine
# actually i can also add a class for whether it's a full body thing or not --> if it's not, then i can collect only hands, etc
# this also gives me a way to merge with prior data too now that i think about it, i can retroactively add in a "hands" note in the column
# POG

# results.pose_landmarks

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read the video
with mp_holistic.Holistic( # automatically clears memory when section is exited
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            results_holistic = holistic.process(imgRGB)

            if results or results_holistic: 

                if results.multi_hand_landmarks:  # if hand is visible on screen and data is being acquired
                    for handLms in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                            h, w, c = frame.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                if results_holistic.pose_landmarks and body_part == 1: # if the sign requires the full torso and not just a hand
                        mp_drawing.draw_landmarks(
                            frame,
                            results_holistic.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        
                        mp_drawing.draw_landmarks(
                            frame,
                            results_holistic.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles
                            .get_default_pose_landmarks_style())


            if results.multi_hand_world_landmarks:  # if hand is visible on screen and data is being acquired
                # print(results.multi_hand_world_landmarks) # this is a list of dictionairies i guess
                # break
                for handLms in results.multi_hand_world_landmarks:

                    if start_capture:

                        count += 1

                        data = {

                            'WRIST_X'+str(count): handLms.landmark[mpHands.HandLandmark.WRIST].x,
                            'WRIST_Y'+str(count): handLms.landmark[mpHands.HandLandmark.WRIST].y,
                            'WRIST_Z'+str(count): handLms.landmark[mpHands.HandLandmark.WRIST].z,


                            'THUMB_CMC_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_CMC].x,
                            'THUMB_CMC_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_CMC].y,
                            'THUMB_CMC_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_CMC].z,

                            'THUMB_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_MCP].x,
                            'THUMB_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_MCP].y,
                            'THUMB_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_MCP].z,

                            'THUMB_IP_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_IP].x,
                            'THUMB_IP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_IP].y,
                            'THUMB_IP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_IP].z,

                            'THUMB_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x,
                            'THUMB_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y,
                            'THUMB_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.THUMB_TIP].z,

                            'INDEX_FINGER_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x,
                            'INDEX_FINGER_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y,
                            'INDEX_FINGER_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].z,

                            'INDEX_FINGER_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].x,
                            'INDEX_FINGER_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].y,
                            'INDEX_FINGER_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].z,

                            'INDEX_FINGER_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].x,
                            'INDEX_FINGER_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].y,
                            'INDEX_FINGER_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].z,

                            'INDEX_FINGER_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x,
                            'INDEX_FINGER_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y,
                            'INDEX_FINGER_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].z,

                            'MIDDLE_FINGER_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x,
                            'MIDDLE_FINGER_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y,
                            'MIDDLE_FINGER_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].z,

                            'MIDDLE_FINGER_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].x,
                            'MIDDLE_FINGER_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].y,
                            'MIDDLE_FINGER_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].z,

                            'MIDDLE_FINGER_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].x,
                            'MIDDLE_FINGER_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].y,
                            'MIDDLE_FINGER_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].z,

                            'MIDDLE_FINGER_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x,
                            'MIDDLE_FINGER_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y,
                            'MIDDLE_FINGER_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].z,

                            'RING_FINGER_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_MCP].x,
                            'RING_FINGER_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_MCP].y,
                            'RING_FINGER_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_MCP].z,

                            'RING_FINGER_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_PIP].x,
                            'RING_FINGER_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_PIP].y,
                            'RING_FINGER_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_PIP].z,

                            'RING_FINGER_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].x,
                            'RING_FINGER_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].y,
                            'RING_FINGER_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].z,

                            'RING_FINGER_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x,
                            'RING_FINGER_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y,
                            'RING_FINGER_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].z,

                            'PINKY_MCP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_MCP].x,
                            'PINKY_MCP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_MCP].y,
                            'PINKY_MCP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_MCP].z,

                            'PINKY_PIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_PIP].x,
                            'PINKY_PIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_PIP].y,
                            'PINKY_PIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_PIP].z,

                            'PINKY_DIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_DIP].x,
                            'PINKY_DIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_DIP].y,
                            'PINKY_DIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_DIP].z,

                            'PINKY_TIP_X'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_TIP].x,
                            'PINKY_TIP_Y'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_TIP].y,
                            'PINKY_TIP_Z'+str(count): handLms.landmark[mpHands.HandLandmark.PINKY_TIP].z,

                        }

                        # if image_count % 10 == 0 and image_count != 0:  # makes sure every signs gets 10 data points
                        #     index_class += 1
                        #     break

                        df.update(data)

                        if results_holistic.pose_world_landmarks and body_part == 1: # if you ALSO want the general torso (im putting this in here to conserve code)

                            pose_data = {}
                            # format: 'POSE_'+index+X+str(count): data_point.x,

                            for index, data_point in enumerate(results_holistic.pose_world_landmarks.landmark):
                                if index < 23: # 17 - 22 is hands, 23-32 is lower body; 17-22 is already covered by hands data
                                    pose_data.update({
                                        'POSE_' + str(index) + '_X' + str(count): data_point.x,
                                        'POSE_' + str(index) + '_Y' + str(count): data_point.y,
                                        'POSE_' + str(index) + '_Z' + str(count): data_point.z
                                    })


                            # print(pose_data)
                            df.update(pose_data)
                            
                            # print(pose_data)



            # if results_holistic.pose_world_landmarks and results and body_part == 1: # if the sign requires the full torso and not just a hand
            #     total_keypoints = []

            #     if start_capture:

            #         count += 1

            #         # POSE DATA:
                    
            #         total_keypoints = []
                    
            #         for index, data_point in enumerate(results_holistic.pose_world_landmarks.landmark):
            #             if index < 23:
            #                 total_keypoints.append({
            #                     'X': data_point.x,
            #                     'Y': data_point.y,
            #                     'Z': data_point.z
            #                     #  'visibility': data_point.visibility
            #                 })
                    
                
            #     print(total_keypoints)

            #     break

            #     # for x in range(0, len(results_holistic.pose_world_landmarks), 1): 
            #     #     print(results_holistic)
            #     print('get stuff')



            if count >= 40:
                
                df.update({'CLASS': index_class, 'STR_CLASS': str_class, 'TORSO': body_part, 'MOVING': moving}) 
                data_list.append(df)
                print(df['CLASS'], df['STR_CLASS'], df['TORSO'], df['MOVING'])

                if image_count % 100 == 0 and image_count != 0:  # makes sure every signs gets 100 data points
                    index_class += 1
                    str_class = input("what is the sign name")

                image_count += 1
                # print(df)
                df = {}  # note: this is an empty dictionary, not a dataframe
                start_capture = 0
                count = 0

            cv2.putText(frame, 'Frame: ' + str(int(count)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.putText(frame, 'You are on image count: ' + str(int(image_count)), (10, 130), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            # cv2.putText(frame, 'You are on sign: ' + string.ascii_lowercase[int(index_class)], (10, 190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            # the upper code is useful for if the index correlates to the alphabet and we're doing ONLY alphabet, but we are not
            # instead we will just use str_class
            cv2.putText(frame, 'You are on sign: ' + str_class, (10, 190), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.putText(frame, 'Index number: ' + str(index_class), (10, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                # data = pd.DataFrame(data_list, index_col=0)
                data = pd.DataFrame(data_list)
                if body_part == 0:
                    data.to_csv('static/hand_notorso.csv')
                    data_pd = pd.read_csv('static/hand_notorso.csv')
                    print(data_pd['CLASS'], data_pd['STR_CLASS'])

                elif body_part == 1:
                    data.to_csv('static/hand_withtorso.csv')
                    data_pd = pd.read_csv('static/hand_withtorso.csv')
                    print(data_pd['CLASS'], data_pd['STR_CLASS'])

                break

            if key == ord('c'):
                start_capture = 1
                count = 0
                df = {}

            if key == ord('a'):  # if you already have signs and need to add more, start directly at the sign you need to add and then add that in
                if body_part == 0:
                    existing_data = pd.read_csv('static/hand_notorso.csv', index_col=[0]) # old data
                elif body_part == 1:
                    existing_data = pd.read_csv('static/hand_withtorso.csv', index_col=[0]) # old data

                data = pd.DataFrame(data_list) # freshly harvested data
                print('EXISTING DATA')
                # print(existing_data['CLASS'], existing_data['STR_CLASS'], existing_data['TORSO'])
                print(existing_data)

                print('NEW DATA')
                # print(data['CLASS'], data['STR_CLASS'], data['TORSO'])
                print(data)

                # new_data = existing_data.append(data) # new, bigger dataset
                # print(new_data['CLASS'], existing_data['STR_CLASS'])
                # new_data.to_csv('static/world_handtracking_data3.csv')

                new_data = pd.concat([data, existing_data.loc[:]]).reset_index(drop=True)

                if body_part == 0:
                    new_data.to_csv('static/hand_notorso.csv')
                    df_test = pd.read_csv('static/hand_notorso.csv')

                elif body_part == 1:
                    new_data.to_csv('static/hand_withtorso.csv')
                    df_test = pd.read_csv('static/hand_withtorso.csv')
                # new_data.to_csv('static/world_handtracking_data3.csv')
                # df_test = pd.read_csv('static/world_handtracking_data3.csv')
                print('COMBINED DATA')
                # print(df_test['CLASS'], df_test['STR_CLASS'], df_test['TORSO'])
                
                print(df_test)
                # print(df_test)
                break

        # Break the loop
        else:
            break

cap.release()
cv2.destroyAllWindows()
