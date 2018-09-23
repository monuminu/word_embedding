# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)
# Others
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from text_util import clean_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('tagged_plots_movielens.csv')
df = df.dropna()
df = df.iloc[:,2:4].reset_index(drop = True)
df.columns = ['text','labels']
df['text'] = df['text'].map(lambda x: clean_text(x))
le = LabelEncoder()
df['labels'] = le.fit_transform(df['labels'])

### Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)

#train test split
X_train, X_test, y_train, y_test = train_test_split(data, df['labels'], test_size=0.15, random_state=42)

## Network architecture
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Fit the model
model.fit(X_train, y_train, validation_split=0.4, epochs=3)

#Evaluate the model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred,axis = 1)
print(classification_report(y_test, y_pred_labels))
print(accuracy_score(y_test, y_pred_labels))