import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Data Import
sms = pd.read_csv('spam.csv', encoding='ISO-8859-1')


# Data Preprocessing
cols_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
sms.drop(cols_to_drop, axis=1, inplace=True)
sms.columns = ['label', 'message']


# Count Vectorizer
cv = CountVectorizer(decode_error='ignore')
X = cv.fit_transform(sms['message'])


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, sms['label'])


# MNB model Training
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)

print(f'Train Accuracy: {mnb.score(X_train, Y_train)}')
print(f'Test Accuracy: {mnb.score(X_test, Y_test)}')