
# Load the libraries
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import io

df = pd.read_csv('training_data.csv')

# prints top few samples the dataset
print(df.head())

x_train = df.Text
y_train = df.Sentiment

# check the dimensions
print("Number of training samples: {}".format(x_train.shape[0]))

# bag of words model to convert text to numbers
cv = CountVectorizer(binary=False)

# transformed train reviews
train_reviews = cv.fit_transform(x_train)

# transformed sentiment data
sentiment_data = y_train

train_reviews.reshape(-1,1)
print(sentiment_data.shape)
print(train_reviews.shape)

# The data preparation is done, steps below prepare a sample model
# preparing the model
lr = LogisticRegression()
# training the model for Bag of words
lr_bow = lr.fit(train_reviews, sentiment_data)

df = pd.read_csv('contestant_judgment.csv')
x_test = df.Text
test_reviews = cv.transform(x_test)

lr_bow_predict = lr.predict(test_reviews)

df["predicted_sentiment"] = lr_bow_predict
df.to_csv("contestant_judgment.csv")
