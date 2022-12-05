# from basic import *
import numpy as np
import pandas as pd
label_df = pd.read_csv('../dataset/label.csv')
for i, row in label_df.iterrows():
    print(row)