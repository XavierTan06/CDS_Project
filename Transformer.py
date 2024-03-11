import os
import json
import pandas as pd
import numpy as np

script_dir = os.path.dirname(__file__)

df_train = pd.read_csv(os.path.join(script_dir, r"RECCON-main/data/transform/train.csv"))
df_test = pd.read_csv(os.path.join(script_dir, r"RECCON-main/data/transform/test.csv"))

