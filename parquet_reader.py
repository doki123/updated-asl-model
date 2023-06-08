#https://www.kaggle.com/competitions/asl-signs/data?select=train.csv

import pandas as pd

# df =  pd.read_parquet('download-test.parquet', engine='pyarrow')
# print(df)

# cols = ["type"] # this just reads out a specific column from the parquet
# df1 = pd.read_parquet('download-test.parquet', columns=cols) # note: the label for which sign this is --> train.csv from kaggle
# print(df1)

asldf = pd.read_csv('asl-kaggle-dataset-train.csv')
# print(asldf.sort_values(by=["sign"]))

# sign w/least data --> 299 inputs, sign w/most --> 415 inputs
# print(set(asldf["sign"]))

unwanted_signs = ["toy", "doll", "giraffe", "puzzle", "wolf", "goose", "alligator", "tiger", "pig", "cow", "lion", 
                  "zebra", "donkey", "hen", "elephant", "duck", "horse", "cowboy", "balloon", "helicopter", "clown", 
                  "farm", "puppy", "kitty", "owie", "yucky", "potty"]

starting_signs = ["no", "thankyou", "hello", "yes"]

cut_df = pd.DataFrame() # only has data for no / yes / thank you / hello

for wanted in starting_signs:
    wantedf = asldf[asldf["sign"] == wanted]
    cut_df = pd.concat([cut_df, wantedf])


cut_df = cut_df.reset_index()
cut_df = cut_df.drop(["index"], axis=1)

subpath = cut_df["participant_id"] # note: the path for the parquet files is in the format train_landmark_files/[participant_id]/[sequence_id].parquet
# sequence_id is the name of the parquet file, while participant_id is the name of the subfolder the parquet is in
print(len(subpath))
print(len(set(subpath)))
print()