import os
import pandas as pd
from sklearn.model_selection import train_test_split
from labelMap import label2id, id2label

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(dir_path, r"data\SocialMedia\sentimentdataset_raw.csv"))

print(len(df["Sentiment"].unique()))

#TODO - Map??? Probably not worth it