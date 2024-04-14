import os
import pandas as pd
import numpy as np
import keras
from keras import layers
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import SimpleRNN

dir_path = os.path.dirname(os.path.realpath(__file__))

df_train = pd.read_csv(os.path.join(dir_path, r"data/clean/Merge_train.csv"))
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

# define model
model = Sequential()
model.add(Embedding(len(word2vec_model.wv), 100, input_length=200))
model.add(SimpleRNN(64))
#model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='sigmoid'))
print(model.summary())

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_w2v, y_train, batch_size=32, epochs=5, validation_data=(X_test_w2v, y_test))

from sklearn.metrics import classification_report
val_pred = np.argmax(model.predict(X_test_w2v), axis=-1)
print(classification_report(y_test, val_pred))