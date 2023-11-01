from cmath import isnan
from flask_pymongo import PyMongo
from urllib import request
import mediapipe as mp
from json import load
from flask import *
import pandas as pd
import joblib
import numpy as np
import string


loaded_model = joblib.load('static/world_landmark_model_hand.joblib')
loaded_model_pose = joblib.load('static/world_landmark_model_torso.joblib')


moving_points_plot = {
    'WRIST': 0, 
    'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4, 
    'INDEX_MCP': 5, 'INDEX_PIP': 6, 'INDEX_DIP': 7, 'INDEX_TIP': 8, 
    'MIDDLE_MCP': 9, 'MIDDLE_PIP': 10, 'MIDDLE_DIP': 11, 'MIDDLE_TIP': 12,
    'RING_MCP': 13, 'RING_PIP': 14, 'RING_DIP': 15, 'RING_TIP': 16,
    'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
} 

data = {

    'WRIST_X': '',
    'WRIST_Y': '',
    'WRIST_Z': '',

    'THUMB_CMC_X': '',
    'THUMB_CMC_Y': '',
    'THUMB_CMC_Z': '',

    'THUMB_MCP_X': '',
    'THUMB_MCP_Y': '',
    'THUMB_MCP_Z': '',

    'THUMB_IP_X': '',
    'THUMB_IP_Y': '',
    'THUMB_IP_Z': '',

    'THUMB_TIP_X': '',
    'THUMB_TIP_Y': '',
    'THUMB_TIP_Z': '',

    'INDEX_FINGER_MCP_X': '',
    'INDEX_FINGER_MCP_Y': '',
    'INDEX_FINGER_MCP_Z': '',

    'INDEX_FINGER_PIP_X': '',
    'INDEX_FINGER_PIP_Y': '',
    'INDEX_FINGER_PIP_Z': '',

    'INDEX_FINGER_DIP_X': '',
    'INDEX_FINGER_DIP_Y': '',
    'INDEX_FINGER_DIP_Z': '',

    'INDEX_FINGER_TIP_X': '',
    'INDEX_FINGER_TIP_Y': '',
    'INDEX_FINGER_TIP_Z': '',

    'MIDDLE_FINGER_MCP_X': '',
    'MIDDLE_FINGER_MCP_Y': '',
    'MIDDLE_FINGER_MCP_Z': '',

    'MIDDLE_FINGER_PIP_X': '',
    'MIDDLE_FINGER_PIP_Y': '',
    'MIDDLE_FINGER_PIP_Z': '',

    'MIDDLE_FINGER_DIP_X': '',
    'MIDDLE_FINGER_DIP_Y': '',
    'MIDDLE_FINGER_DIP_Z': '',

    'MIDDLE_FINGER_TIP_X': '',
    'MIDDLE_FINGER_TIP_Y': '',
    'MIDDLE_FINGER_TIP_Z': '',

    'RING_FINGER_MCP_X': '',
    'RING_FINGER_MCP_Y': '',
    'RING_FINGER_MCP_Z': '',

    'RING_FINGER_PIP_X': '',
    'RING_FINGER_PIP_Y': '',
    'RING_FINGER_PIP_Z': '',

    'RING_FINGER_DIP_X': '',
    'RING_FINGER_DIP_Y': '',
    'RING_FINGER_DIP_Z': '',

    'RING_FINGER_TIP_X': '',
    'RING_FINGER_TIP_Y': '',
    'RING_FINGER_TIP_Z': '',

    'PINKY_MCP_X': '',
    'PINKY_MCP_Y': '',
    'PINKY_MCP_Z': '',

    'PINKY_PIP_X': '',
    'PINKY_PIP_Y': '',
    'PINKY_PIP_Z': '',

    'PINKY_DIP_X': '',
    'PINKY_DIP_Y': '',
    'PINKY_DIP_Z': '',

    'PINKY_TIP_X': '',
    'PINKY_TIP_Y': '',
    'PINKY_TIP_Z': '',

}

og_data_keys = list(data.keys())


divided_threes = []
threes = []
count_threes = 0

for x in og_data_keys:
    threes.append(x)
    count_threes += 1
    if count_threes % 3 == 0: 
        divided_threes.append(threes)
        threes = []


# removes issue of 'Unnamed: 0' by treating it as index

handdf = pd.read_csv('static/hand_notorso.csv', index_col=[0])
torsodf = pd.read_csv('static/hand_withtorso.csv', index_col=[0])

hand_signs = list(set(list(handdf['STR_CLASS']))) # list of all signs that just use the hand
torso_signs = list(set(list(torsodf['STR_CLASS']))) # list of all signs that require the full torso

moving_signs_dict = {} # compilation of signs that move 

for x in range(0, len(handdf), 1): 
    moving = handdf['MOVING'][x]
    if moving != 0:
        sign_class = handdf.iloc[x]['STR_CLASS']
        moving_signs_dict[sign_class] = moving


for x in range(0, len(torsodf), 1): 
    moving = torsodf['MOVING'][x]
    if moving != '0':
        sign_class = torsodf.iloc[x]['STR_CLASS']
        moving_signs_dict[sign_class] = moving



app = Flask('Jumble')
app.config['MONGO_URI'] = "mongodb://Shruti:bfy6SeOsMbF02Ffp@cluster0-l0gvf.mongodb.net/Shruti_Dats?retryWrites=true&w=majority"
app.config['SECRET_KEY'] = "huh"
mongo = PyMongo(app)

#strclass_list = model_df['STR_CLASS'].unique()
strclass_list = handdf['STR_CLASS'].unique()
strclass_list = np.append(strclass_list, torsodf['STR_CLASS'].unique()) # all signs existing in database

class_index_dict = {} # dictionary of the sign meaning : numerical value assigned to them for classifying

print(strclass_list)

for c in strclass_list:
    if c in hand_signs: # if the sign only uses hand
        hand_index = handdf.loc[handdf['STR_CLASS'] == c, 'CLASS'].iloc[0]
        class_index_dict.update({hand_index: c})

    elif c in torso_signs: # if the sign uses the full torso
        pose_index = torsodf.loc[torsodf['STR_CLASS'] == c, 'CLASS'].iloc[0]
        class_index_dict.update({pose_index: c})


def what_signs_available(strclass_list):
    current_signs = ''
    for x in list(strclass_list): 
        if x != strclass_list[-1]:
            current_signs += str(x) + ', '
        else: 
            current_signs += str(x)
            
    sign_404 = 'Current signs in database: ' + current_signs + '.'
    return(sign_404, current_signs)


# Turn formats the collected data for the csv; this is for signs that only use hands 
def convert_signs(raw_data, int_class = None, str_class = None, moving = None):  # the = None is added so that if the argument is not necessary for that route, you can skip adding it while calling convert_signs()
    new_data = {}
    
    for big_index in range(0, 40, 1):
        for small_index in range(0, 21, 1):
            for indiv in ['x', 'y', 'z']:
                # new_data.update({og_data_keys[small_index * ['x', 'y', 'z'].index(indiv)]+str(big_index): raw_data[big_index][small_index][indiv]})
                new_data.update(
                    {og_data_keys[small_index * 3]+str(big_index + 1): raw_data[big_index][small_index]['x']})
                new_data.update(
                    {og_data_keys[small_index * 3 + 1]+str(big_index + 1): raw_data[big_index][small_index]['y']})
                new_data.update(
                    {og_data_keys[small_index * 3 + 2]+str(big_index + 1): raw_data[big_index][small_index]['z']})

    if int_class is not None: # in case int_class and str_class arguments are given when convert_signs() is called
        new_data.update({'CLASS': int_class, 'STR_CLASS': str_class})

    new_data.update({'MOVING': moving})

    return new_data

# Formats collected signs that use the full torso 
def convert_signs_pose(raw_data_pose, int_class = None, str_class = None, moving = None):  # the = None is added so that if the argument is not necessary for that route, you can skip adding it while calling convert_signs()
    new_data = {}
    
    for big_index in range(0, 40, 1):
        for small_index in range(0, 33, 1):
            for indiv in ['x', 'y', 'z']:
                # new_data.update({og_data_keys[small_index * ['x', 'y', 'z'].index(indiv)]+str(big_index): raw_data[big_index][small_index][indiv]})
                new_data.update(
                    {'POSE_' + str(small_index) + '_X' + str(big_index + 1): raw_data_pose[big_index][small_index]['x']})
                new_data.update(
                    {'POSE_' + str(small_index) + '_Y' + str(big_index + 1): raw_data_pose[big_index][small_index]['y']})
                new_data.update(
                    {'POSE_' + str(small_index) + '_Z' + str(big_index + 1): raw_data_pose[big_index][small_index]['z']})

    if int_class is not None: # in case int_class and str_class arguments are given when convert_signs() is called
        new_data.update({'CLASS': int_class, 'STR_CLASS': str_class})

    new_data.update({'MOVING': moving})

    return new_data

# Predicts hand signs
def predict_signs(data):
    data.pop('MOVING') # This is so that there is only numerical data being used in predictions
    prediction = loaded_model.predict([list(data.values())])
    strung_chars = class_index_dict[int(prediction)]
    return strung_chars

# Predicts full torso signs
def predict_signs_pose(data):
    data.pop('MOVING') # This is so that there is only numerical data being used in predictions
    prediction = loaded_model_pose.predict([list(data.values())])
    strung_chars = class_index_dict[int(prediction)]
    return strung_chars


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        return redirect('/')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else: 
        return redirect('/index')


@app.route('/add_collection', methods=['GET', 'POST']) # Route to add signs to database
def add_collection():
    if request.method == 'GET':
        return render_template('add_sign_resultstest.html')
    else:
        existing_data = pd.read_csv('static/hand_notorso.csv', index_col=0)
        existing_data_pose = pd.read_csv('static/hand_withtorso.csv', index_col=0)
        print(existing_data)

        sign_name = request.json['sign_name']
        moving = request.json['moving']
        torso = request.json['torso']
        collected_data_collection = request.json['collected_data']

        if moving == "No":
            moving = 0
        elif moving == "Yes":
            moving = 1

        if torso == "No":
            torso = 0
        elif torso == "Yes":
            torso = 1

                
        print('MOVING', moving, 'TORSO', torso)

        str_class = sign_name.lower()

        hand_classes = []
        pose_classes = []

        converted_data_collection = []
        converted_data_collection_pose = []


        for x in list(existing_data['STR_CLASS']):
            hand_classes.append(x.lower())
        for y in list(existing_data_pose['STR_CLASS']):
            pose_classes.append(y.lower())

        
        if torso == 0: # If the sign only uses the hand

            if str_class in hand_classes:
                print('Sign exists in database')
                x = existing_data.loc[existing_data['STR_CLASS'] == str_class, 'CLASS'].iloc[0]
                int_class = existing_data[existing_data['STR_CLASS'] == str_class]['CLASS'].iloc[0]

            else: 
                print('Sign does not currently exist in database')
                int_class = existing_data.iloc[-1]['CLASS'] + 1


            for signs in collected_data_collection: # This is the data for the signs that you add
                new_data = convert_signs(signs, int_class, str_class, moving)
                converted_data_collection.append(new_data)

            converted_dataframe = pd.DataFrame(converted_data_collection)
            converted_dataframe['MOVING'] = moving
            converted_dataframe['TORSO'] = torso
                
            print('NEW DATA', converted_dataframe)
            
            existing_data = pd.concat([existing_data, converted_dataframe])

            existing_data = existing_data.sort_values('CLASS')
            
            existing_data = existing_data.loc[:, ~existing_data.columns.str.match("Unnamed")] 
            existing_data.reset_index(inplace=True, drop=True)

            existing_data.to_csv('static/hand_notorso.csv')
            print('NEW!')

            x = pd.read_csv('static/hand_notorso.csv', index_col=0)
            print(x)

        elif torso == 1: # if the hand uses the full torso 
            hand_data = request.json['hand_data']
            pose_data = request.json['pose_data']

            if str_class in pose_classes:
                print('Sign in database')
                x = existing_data_pose.loc[existing_data_pose['STR_CLASS'] == str_class, 'CLASS'].iloc[0]
                int_class = existing_data_pose[existing_data_pose['STR_CLASS'] == str_class]['CLASS'].iloc[0]

            else: 
                print('Sign not in database')
                int_class = existing_data_pose.iloc[-1]['CLASS'] + 1

            for sign_40 in range(0, len(hand_data), 1): # formats the hand landmarks
                sign_total = hand_data[sign_40]
                dict_test = {}

                for frame in range(0, len(sign_total), 1):
                    frame_data = sign_total[frame]
                    frame_singelist = frame_data[0]

                    for x in range(0, len(frame_singelist), 1):
                        dict_vals = list(frame_singelist[x].values())
                        dict_test.update({divided_threes[x][0] + str(frame + 1): dict_vals[0]})
                        dict_test.update({divided_threes[x][1] + str(frame + 1): dict_vals[1]})
                        dict_test.update({divided_threes[x][2] + str(frame + 1): dict_vals[2]})                    
                
                converted_data_collection.append(dict_test)


            converted_dataframe = pd.DataFrame(converted_data_collection) # Dataframe of all the hand landmarks

            print(converted_dataframe)


            for pose_signs in pose_data:
                new_data_pose = convert_signs_pose(pose_signs, int_class, str_class, moving)
                new_data_pose.pop('CLASS')
                new_data_pose.pop('STR_CLASS')
                new_data_pose.pop('MOVING')

                for x in list(new_data_pose.keys()):
                    y = x.replace('POSE_', '')
                    if y[1] != '_':

                        if int(y[0] + y[1]) >= 23:
                            new_data_pose.pop(x)


                converted_data_collection_pose.append(new_data_pose)


            converted_dataframe_pose = pd.DataFrame(converted_data_collection_pose) # Dataframe of all the pose landmarks
            
            print(converted_dataframe_pose)

            merged = pd.concat([converted_dataframe, converted_dataframe_pose], axis=1) # Compiles data into one greater Dataframe
            merged['CLASS'] = int_class
            merged['STR_CLASS'] = str_class
            merged['MOVING'] = moving
            merged['TORSO'] = torso
            
            print(merged)

            existing_data_pose = pd.concat([existing_data_pose, merged])
            existing_data_pose = existing_data_pose.sort_values('CLASS')
            existing_data_pose = existing_data_pose.loc[:, ~existing_data_pose.columns.str.match("Unnamed")] 
            existing_data_pose.reset_index(inplace=True, drop=True)
            
            print(existing_data_pose.columns)

            existing_data_pose.to_csv('static/hand_withtorso.csv')
            x = pd.read_csv('static/hand_withtorso.csv', index_col=0)
            
            print(x)

        return redirect('/add_collection')


@app.route('/grid', methods=['GET', 'POST']) # Route to see signs in 3D grid
def grid():
    global strclass_list
    if request.method == 'GET':
        return render_template('only_grid_pose.html', sign_404 = what_signs_available(strclass_list)[0])

    else:
        example_sign = request.json["example_sign"].lower()
        hand_signs = list(set(list(handdf['STR_CLASS'])))
        torso_signs = list(set(list(torsodf['STR_CLASS'])))
        

        if (example_sign in hand_signs) or (example_sign in torso_signs):
            if example_sign in hand_signs:
                print('True')

                example_sign_dataset = handdf[handdf['STR_CLASS'] == example_sign]

                first_points = example_sign_dataset.iloc[0] # Takes the first frame (21 points for the hand)
                int_class = first_points['CLASS']

                print(int_class)

                moving = 0  # whether the sign is moving 

                if example_sign in moving_signs_dict.keys():
                    moving = 1

                count = 0

                first_points = first_points.drop('MOVING')
                first_points = first_points.drop('STR_CLASS')
                first_points = first_points.drop('TORSO')

                print(first_points)
                first_points = first_points.astype(float)

                # temporary dictionary that groups up the x-y-zs
                xyz_dictionary = {'x': 0, 'y': 0, 'z': 0}
                # list of the 21 hand points, each point being associated with a dictionary with x, y, z
                grouped_xyz_list = []

                for datapoints in first_points:  # runs through the averaged values
                    # each hand point has three columns: x, y, z

                    if count == 0:  # gets the x-column datapoint
                        xyz_dictionary['x'] = datapoints

                    elif count == 1:  # gets the y-column datapoint
                        xyz_dictionary['y'] = datapoints

                    elif count == 2:  # gets the z-column datapoint
                        xyz_dictionary['z'] = datapoints
                        grouped_xyz_list.append(xyz_dictionary)
                        xyz_dictionary = {'x': 0, 'y': 0, 'z': 0}
                        count = -1

                    count += 1

                print(grouped_xyz_list)
                return jsonify(
                    {
                        'example_datapoints': grouped_xyz_list, 'moving': moving, 
                        'sign_404': what_signs_available(strclass_list)[0], 'moving_points_plot': moving_points_plot, 'torso': 0
                    })
            

            elif example_sign in torso_signs:

                print('True')

                example_sign_dataset = torsodf[torsodf['STR_CLASS'] == example_sign]
                print(example_sign_dataset)

                first_points = example_sign_dataset.iloc[0] # Takes the first frame (21 points for the hand)
                
                int_class = first_points['CLASS']


                print(int_class)
                print(first_points)

                moving = 0  # whether the sign is moving 

                if example_sign in moving_signs_dict.keys():
                    moving = 1

                hand_count = 0
                pose_count = 0

                first_points = first_points.drop('MOVING')
                first_points = first_points.drop('STR_CLASS')
                first_points = first_points.drop('TORSO')

                first_points = first_points.astype(float)

                pose_index = {}
                hand_index = {}

                temp = first_points.to_dict()
                temp = list(temp.keys())
                temp_df = first_points.to_frame()

                for index, x in enumerate(temp):
                    if 'POSE' in x:
                        pose_index.update({index: x})
                    else: 
                        hand_index.update({index: x})

                print('ROW TEST')

                handsplit = example_sign_dataset.iloc[0][list(hand_index.values())]
                torsosplit = example_sign_dataset.iloc[0][list(pose_index.values())]

                # temporary dictionary that groups up the x-y-zs
                xyz_dictionary_hand = {'x': 0, 'y': 0, 'z': 0}
                xyz_dictionary_pose = {'x': 0, 'y': 0, 'z': 0}

                # list of the 21 hand points, each point being associated with a dictionary with x, y, z
                grouped_xyz_list_hand = []
                grouped_xyz_list_pose = []


                for datapoints in handsplit:  # runs through the averaged values
                    # each hand point has three columns: x, y, z

                    if hand_count == 0:  # gets the x-column datapoint
                        xyz_dictionary_hand['x'] = datapoints

                    elif hand_count == 1:  # gets the y-column datapoint
                        xyz_dictionary_hand['y'] = datapoints

                    elif hand_count == 2:  # gets the z-column datapoint
                        xyz_dictionary_hand['z'] = datapoints
                        grouped_xyz_list_hand.append(xyz_dictionary_hand)
                        xyz_dictionary_hand = {'x': 0, 'y': 0, 'z': 0}
                        hand_count = -1

                    hand_count += 1

                
                for datapoints in torsosplit:  # runs through the averaged values
                    # each hand point has three columns: x, y, z

                    if pose_count == 0:  # gets the x-column datapoint
                        xyz_dictionary_pose['x'] = datapoints

                    elif pose_count == 1:  # gets the y-column datapoint
                        xyz_dictionary_pose['y'] = datapoints

                    elif pose_count == 2:  # gets the z-column datapoint
                        xyz_dictionary_pose['z'] = datapoints
                        grouped_xyz_list_pose.append(xyz_dictionary_pose)
                        xyz_dictionary_pose = {'x': 0, 'y': 0, 'z': 0}
                        pose_count = -1

                    pose_count += 1


                print('MOVING', moving)
                print('HAND', grouped_xyz_list_hand)

                return jsonify(
                    {
                        'example_datapoints': grouped_xyz_list_hand, 'example_datapoints_pose': grouped_xyz_list_pose, 'moving': moving, 'torso': 1, 
                        'sign_404': what_signs_available(strclass_list)[0], 'moving_points_plot': moving_points_plot
                    })
        
        else: 
            return jsonify({'example_datapoints': [], 'moving': 0, 'trace_tip': 'NONE', 'sign_404': what_signs_available(strclass_list)[0]})


@app.route('/practice', methods=['GET', 'POST']) # Practice signs in front of webcam, predict if they're right/wrong
def practice():
    global strclass_list
    
    if request.method == 'GET':
        return render_template('practice_signs_pose.html', sign_505 = what_signs_available(strclass_list)[1], sign_404 = ('Current signs in database: ' + what_signs_available(strclass_list)[1]))
    else:
        str_class = request.json["picked_sign"].lower()  # string letter of the sign chosen 
        raw_data = request.json["raw_data"]
        raw_data_pose = request.json["raw_data_pose"]
                
        if str_class in strclass_list:
            if str_class in hand_signs:
                converted_data = convert_signs(raw_data)
                print('CONVERTED DATA', converted_data)
                strung_chars = predict_signs(converted_data)

                print('PREDICTED', strung_chars, 'INPUT', str_class)

                if str_class == strung_chars.lower():
                    result = "Correct"

                else:
                    result = "Incorrect"

                return jsonify({'prediction': result, 'in_database': True, 'sign_404': ''})
            
            elif str_class in torso_signs:
                converted_data_pose = convert_signs_pose(raw_data_pose)

                all_keys = list(converted_data_pose.keys())
                all_keys.remove('MOVING')
                for x in all_keys:
                    y = x.replace('POSE_', '')
                    print(x, y)
                    if y[1] != '_':
                        if int(y[0] + y[1]) >= 23:
                            converted_data_pose.pop(x)

                converted_data = convert_signs(raw_data)
                converted_data.update(converted_data_pose)
                strung_chars = predict_signs_pose(converted_data)
                print('PREDICTED', strung_chars, 'INPUT', str_class)


                if str_class == strung_chars.lower():
                    result = "Correct"

                else:
                    result = "Incorrect"


                return jsonify({'prediction': result, 'in_database': True, 'sign_404': ''})


        else: 
            return jsonify({'prediction': 'This sign is not currently in the database', 'in_database': False, 'sign_404': what_signs_available(strclass_list)[0]})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        raw_data = request.json

        converted_data = convert_signs(raw_data)
        strung_chars = predict_signs(converted_data)

        return jsonify({'success': True, 'prediction': strung_chars})


if '__main__' == __name__:
    app.run(debug=True, port=5001)


