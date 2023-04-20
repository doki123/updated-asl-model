### FROM APP.PY

@app.route('/add', methods=['GET', 'POST'])  # THIS METHOD IS NOT CURRENTLY IN USE
def add():
    if request.method == 'GET':
        return render_template('add_signs.html')
    else:
        
        sign_name = request.json['sign_name']
                
        print(sign_name)

        if sign_name in string.ascii_letters:
            sign_name = sign_name.upper()
            sign_name = int(string.ascii_uppercase.index(sign_name))

        collected_data = request.json['collected_data']
        converted_data = convert_signs(collected_data)

        print(sign_name)
        converted_data.update({'CLASS': sign_name})

        existing_data = pd.read_csv('world_landmark_hands/world_handtracking_data2.csv', index_col=0)

        existing_data = existing_data.append(converted_data, ignore_index=True)
        existing_data = existing_data.sort_values('CLASS')
        
        existing_data.reset_index(inplace=True, drop=True)  # drop=True prevents the original index from being added as a new column

        existing_data.to_csv('world_landmark_hands/world_handtracking_data2.csv')

            # test_see = pd.read_csv('world_landmark_hands/world_handtracking_data2.csv', index_col=0)
            # print(test_see[test_see['CLASS'] == 2])
            
        return redirect('/add')



@app.route('/learn', methods=['GET', 'POST']) # THIS METHOD IS NOT CURRENTLY IN USE
def learn():
    if request.method == 'GET':
        return render_template('learn_sign.html')
    else:
        return redirect('/learn')





@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # # print(request.json)  # this is the current data taken from the HTML --> put everything in one long list and predict
        raw_data = request.json

        # # has 40 dictionaries for 40 frames, each dictionary has 21 dictionaries (thumb, wrist, etc) each with x-y-x coords
        # # data from fps_handtracking has the thumb-wrist-etc all in one big dictionary
        # # merge the smaller 21 dictionaries --> one large dictionary
        # # print(raw_data[0][0])
        # # raw_test.update(
        # #     {'WRIST_X1': raw_data[0][0]['x'], 'WRIST_Y1': raw_data[0][0]['y'], 'WRIST_z1': raw_data[0][0]['z']})
        # # print(raw_test)
        # for big_index in range(0, 40, 1):
        #     for small_index in range(0, 21, 1):
        #         for indiv in ['x', 'y', 'z']:
        #             # new_data.update({og_data_keys[small_index * ['x', 'y', 'z'].index(indiv)]+str(big_index): raw_data[big_index][small_index][indiv]})
        #             new_data.update({og_data_keys[small_index * 3]+str(big_index): raw_data[big_index][small_index]['x']})
        #             new_data.update({og_data_keys[small_index * 3 + 1]+str(big_index): raw_data[big_index][small_index]['y']})
        #             # NOTE: in the og dataset, Thumb_IP_Z + THUMB_TIP_Z wwere overwritten --> when a new dataset is created, and the dataset is 2520, remove the if
        #             if og_data_keys[small_index * 3 + 2] in ['THUMB_IP_Z', 'THUMB_TIP_Z']:
        #                 new_data.update({og_data_keys[small_index * 3 + 2]: raw_data[big_index][small_index]['z']})
        #             else:
        #                 new_data.update({og_data_keys[small_index * 3 + 2]+str(big_index): raw_data[big_index][small_index]['z']})

        # prediction = loaded_model.predict([list(new_data.values())])
        # strung_chars = string.ascii_lowercase[int(prediction)]
        # print(strung_chars)

        converted_data = convert_signs(raw_data)
        strung_chars = predict_signs(converted_data)
        # strung_chars = predict_signs(raw_data)

        return jsonify({'success': True, 'prediction': strung_chars})