import pandas as pd 
import numpy as np 
from numpy import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim.models import word2vec
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import nltk

df = pd.read_csv('D:/Data_Science_Work/word_embedding/tagged_plots_movielens.csv')
df = df.dropna()
df['plot'].apply(lambda x: len(x.split(' '))).sum()
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

count_vectorizer = CountVectorizer(
    analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=3000) 

train_data_features = count_vectorizer.fit_transform(train_data['plot'])
