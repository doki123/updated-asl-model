# NOTE: WE ARE SPECIFICALLY WORKING W/DATA2. MAYBE RENAME STUFF TO LESSEN CONFUSION LATER ON

import pandas as pd
import string 
import numpy as np

class_junk_check = string.ascii_uppercase 
df = pd.read_csv('static/world_handtracking_data2.csv', index_col=[0])  # removes issue of 'Unnamed: 0' by treating it as index
classes = df['STR_CLASS']
bad_class = []
for check in classes: 
    if check not in class_junk_check:
        # print('BAD!', check)
        bad_class.append(check)

print('FINISHED!')

bad_class = set(bad_class) # removes any duplicate items

df_new = df[~df['STR_CLASS'].isin(bad_class)] # df['STR_CLASS'].isin(bad_class) finds all rows where str_class is a junk sign; ~ = negation; TLDR finds all rows without junk classes

print(set(df_new['STR_CLASS'].values))
print('CHECKING AGAIN!')

new_classes = df_new['STR_CLASS']
for check in new_classes:
    if check not in class_junk_check:
        print('BAD!', check)

print('FINISHED!')

# print(df_new['STR_CLASS'].values)

# print(df_new[:40]['MOVING'].values) #WHY DOES 'A' HAVE SO MANY SIGNS???? WHY DOES 'MOVING' HAVE MISSING VALUES???? DATASET POTENTIALLY HAUNTED
df_new = df_new.replace(np.nan, 'NONE') # NOTE: For some reason 'A' had NaNs in moving column; I replaced them here
print(df_new[:40]['MOVING'].values)
df_new.to_csv('static/world_handtracking_data2.csv')
