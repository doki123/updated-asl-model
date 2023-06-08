#https://www.kaggle.com/competitions/asl-signs/data?select=train.csv
#TODO: create a function to create a new subfolder in kaggle_asl_data for each new sign and download relevant parquet files
import pandas as pd

# df =  pd.read_parquet('download-test.parquet', engine='pyarrow')
# print(df)

# cols = ["type"] # this just reads out a specific column from the parquet
# df1 = pd.read_parquet('download-test.parquet', columns=cols) # note: the label for which sign this is --> train.csv from kaggle

asldf = pd.read_csv('asl-kaggle-dataset-train.csv')
# sign w/least data --> 299 inputs, sign w/most --> 415 inputs

unwanted_signs = ["toy", "doll", "giraffe", "puzzle", "wolf", "goose", "alligator", "tiger", "pig", "cow", "lion", 
                  "zebra", "donkey", "hen", "elephant", "duck", "horse", "cowboy", "balloon", "helicopter", "clown", 
                  "farm", "puppy", "kitty", "owie", "yucky", "potty"]

starting_signs = ["no", "thankyou", "hello", "yes"]

cut_df = pd.DataFrame() # only has data for no / yes / thank you / hello

subpath_names = {} # format: {'sign_name': [path]}, this will be used to download pertinent files and sort them into subfolders

for wanted in starting_signs: 
    wantedf = asldf[asldf["sign"] == wanted]
    cut_df = pd.concat([cut_df, wantedf])
    paths = cut_df["path"].values
    subpath_names.update({wanted: paths})


cut_df = cut_df.reset_index()
cut_df = cut_df.drop(["index"], axis=1)

print(cut_df)

# note: the path for the parquet files is in the format train_landmark_files/[participant_id]/[sequence_id].parquet
# sequence_id is the name of the parquet file, while participant_id is the name of the subfolder the parquet is in

