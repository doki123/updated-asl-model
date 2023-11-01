#  -- attack: the message attacks another politician
# -- constituency: the message discusses the politician's constituency
# -- information: an informational message about news in government or the wider U.S.
# -- media: a message about interaction with the media
# -- mobilization: a message intended to mobilize supporters
# -- other: a catch-all category for messages that don't fit into the other
# -- personal: a personal message, usually expressing sympathy, support or condolences, or other personal opinions
# -- policy: a message about political policy
# -- support: a message of political support 
import pandas as pd

df = pd.read_csv('/Users/shrutiladiwala/Desktop/Backup/ladiw/Documents/Coding/pycharm_files/GraphsExercisesTests/handtracking_test/teacher_code_fps_timing/updated-asl-model/move_later/political_social_data.csv', encoding='latin1')
print(df[['text', 'message', 'audience']])