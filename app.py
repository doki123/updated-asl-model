# NOTE: THE MODEL APP PYTHON CREATION FILE IS IN WORLD_LANDMARKS_HANDS/WORLD_LANDMARK_MODEL.PY

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


loaded_model = joblib.load('static/world_landmark_model.joblib')

# lines up with the static_sign dictionary for hand datapoints in two_grids_current --> this tracks with point of the finger is being traced
# syntax: mov = moving_points_plot['WRIST']; traceList.push(static_sign[0][mov]) --> creates a list of trace points to plot
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

# removes issue of 'Unnamed: 0' by treating it as index
model_df = pd.read_csv('static/backup_data.csv', index_col=[0])

moving_signs_dict = {} # compilation of signs that move and what is being traced in {'J': 'PINKY_TIP'} format

for x in range(0, len(model_df), 1): 
    moving = list(model_df['MOVING'])[x]
    if moving != 'NONE':
        sign_class = model_df.iloc[x]['STR_CLASS']
        moving_signs_dict[sign_class] = moving


app = Flask('Jumble', template_folder='/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/multihand_landmark_handtracking/templates')
app.config['MONGO_URI'] = "mongodb://Shruti:bfy6SeOsMbF02Ffp@cluster0-l0gvf.mongodb.net/Shruti_Dats?retryWrites=true&w=majority"
app.config['SECRET_KEY'] = "huh"
mongo = PyMongo(app)

strclass_list = model_df['STR_CLASS'].unique()

def what_signs_available(strclass_list):
    current_signs = ''
    for x in list(strclass_list): 
        if x != strclass_list[-1]:
            current_signs += str(x) + ', '
        else: 
            current_signs += str(x)
            
    sign_404 = 'Current signs in database: ' + current_signs + '.'
    return(sign_404, current_signs)


def convert_signs(raw_data, int_class = None, str_class = None, moving = "NONE"):  # the = None is added so that if the argument is not necessary for that route, you can skip adding it while calling convert_signs()
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


def predict_signs(data):
    data.pop('MOVING') # This is so that there is only numerical data being used in predictions
    prediction = loaded_model.predict([list(data.values())])
    strung_chars = string.ascii_lowercase[int(prediction)]
    return strung_chars

@app.route('/', methods=['GET', 'POST'])
# TODO: Instead of uselessly having that prediction, instead make a card layout to each page and explain what it does
# Example: https://getbootstrap.com/docs/5.2/examples/album/
def home():
    if request.method == 'GET':
        # return render_template('home.html')
        return render_template('index.html')
    else:
        return redirect('/')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else: 
        return redirect('/index')

@app.route('/add_collection', methods=['GET', 'POST'])
# TODO: Figure out how to deal with multiple moving points
def add_collection():
    if request.method == 'GET':
        return render_template('add_signs.html')
    else:
        existing_data = pd.read_csv('static/backup_data.csv')
        print(existing_data)

        sign_name = request.json['sign_name']
        trace_tip = request.json['trace_tip']
                
        str_class = sign_name.upper()
        
        if (str_class in list(existing_data['STR_CLASS'])):
            int_class = existing_data[existing_data['STR_CLASS'] == str_class]['CLASS'].iloc[0]

        else: 
            int_class = existing_data.iloc[-1]['CLASS'] + 1

        collected_data_collection = request.json['collected_data']
        converted_data_collection = []

        for signs in collected_data_collection: # This is the data for the signs that you add
            converted_data_collection.append(convert_signs(signs, int_class, str_class))
            # converted_data_collection['MOVING'] = trace_tip
            converted_dataframe = pd.DataFrame(converted_data_collection)
            converted_dataframe['MOVING'] = trace_tip
        
            existing_data = pd.concat([existing_data, converted_dataframe])
        existing_data = existing_data.sort_values('CLASS')
        
        # for if_nan in existing_data['MOVING']:
        #     print(if_nan)

        # existing_data.reset_index(inplace=True, drop=True)  # drop=True prevents the original index from being added as a new column
        # print(existing_data[existing_data['STR_CLASS'] ==  str_class])
        existing_data = existing_data.loc[:, ~existing_data.columns.str.match("Unnamed")] 

        print(existing_data)
        existing_data.to_csv('static/backup_data.csv')
            
        return redirect('/add_collection')


@app.route('/grid', methods=['GET', 'POST'])
# TODO: Clean up UI, maybe look into having a little 3D rendition of the hand/a clip of someone doing the gesture playing in tandem
def grid():
    global strclass_list
    if request.method == 'GET':
        # strclass_list = model_df['STR_CLASS'].unique()
        return render_template('only_grid_in_browser.html', sign_404 = what_signs_available(strclass_list)[0])
    else:
        example_sign = request.json["example_sign"].upper()
        strclass_list = model_df['STR_CLASS'].unique()

        if example_sign in strclass_list:
            print('True')

            example_sign_dataset = model_df[model_df['STR_CLASS'] == example_sign]
            first_points = example_sign_dataset.iloc[0] # Takes the first frame (21 points for the hand)
            int_class = first_points['CLASS']

            print(int_class)

            moving = 0  # whether the sign is moving 

            # this takes what part of the hand is supposed to be tracked to create a trace
            trace_tip = first_points['MOVING']

            if example_sign in moving_signs_dict.keys():
                moving = 1

            # THIS WAS TO AVERAGE OUT THE SIGNS, I FOUND IT BAD FOR MOVING GESTURES
            # averaged_points = example_sign_dataset.mean().values  # averages out the columns to get average value of points
            count = 0

            first_points = first_points.drop('MOVING')
            first_points = first_points.drop('STR_CLASS')
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


            print(trace_tip)
            return jsonify(
                {
                    'example_datapoints': grouped_xyz_list, 'moving': moving, 'trace_tip': trace_tip, 
                    'sign_404': what_signs_available(strclass_list)[0], 'moving_points_plot': moving_points_plot
                })
        
        else: 
            return jsonify({'example_datapoints': [], 'moving': 0, 'trace_tip': 'NONE', 'sign_404': what_signs_available(strclass_list)[0]})


@app.route('/practice', methods=['GET', 'POST'])
# TODO: Display what signs are currently in the database in a nice, UI fashion
def practice():
    global strclass_list
    
    if request.method == 'GET':
        return render_template('practice_signs copy.html', sign_505 = what_signs_available(strclass_list)[1], sign_404 = ('Current signs in database: ' + what_signs_available(strclass_list)[1]))
    else:
        str_class = request.json["picked_sign"].upper()  # string letter of the sign chosen 
        raw_data = request.json["raw_data"]
                
        # if str_class in list(model_df['STR_CLASS']): 
        if str_class in strclass_list:

            converted_data = convert_signs(raw_data)
            strung_chars = predict_signs(converted_data)

            print('PREDICTED', strung_chars)

            if str_class == strung_chars.upper():
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


