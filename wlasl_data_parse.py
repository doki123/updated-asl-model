# dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?select=wlasl_class_list.txt
# note: until i can figure out how to deal w/parquet let's try this

import pandas as pd
df = pd.read_json('static/kaggle_wsasl_data/WLASL_v0.3.json')