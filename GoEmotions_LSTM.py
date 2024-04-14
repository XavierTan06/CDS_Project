import os
import pandas as pd
import numpy as np
import keras
from keras import layers
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, f1_score

dir_path = os.path.dirname(os.path.realpath(__file__))

df_train = pd.read_csv(os.path.join(dir_path, r"data/clean/GoEmotions_train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, r"data/clean/Merge_test.csv"))

X_train = df_train["text"]
X_test = df_test["text"]
y_train = df_train["labels"]
y_test = df_test["labels"]

# Train Word2Vec model
sentences = [text.split() for text in X_train]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)


# Convert text data to integers using Word2Vec model
X_train_w2v = [[word2vec_model.wv.key_to_index.get(word, 0) for word in text.split()] for text in X_train]
X_test_w2v = [[word2vec_model.wv.key_to_index.get(word, 0) for word in text.split()] for text in X_test]

# Pad sequences
X_train_w2v = keras.preprocessing.sequence.pad_sequences(X_train_w2v, maxlen=200)
X_test_w2v = keras.preprocessing.sequence.pad_sequences(X_test_w2v, maxlen=200)

inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(20000, 128)(inputs)
# Add 1 unidirectional LSTM
x = layers.LSTM(64, return_sequences=False)(x)
# Add a classifier
outputs = layers.Dense(7, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_w2v, y_train, batch_size=32, epochs=5, validation_data=(X_test_w2v, y_test))

val_pred = np.argmax(model.predict(X_test_w2v), axis=-1)
classification_rep = classification_report(y_test, val_pred)
print(classification_rep)
f1 = f1_score(y_test, val_pred, average="weighted")
print(f"Overall f1: {f1}")

with open("output/GoEmotions_LSTM.txt", "w") as file:
    file.write(classification_rep)
    file.write("\n")
    file.write("Overall f1-score: " + str(f1))