#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#import data
spam_df = pd.read_csv('spam.csv')
#inspect data
spam_df.groupby('Category').describe()
#convert spam column to binary
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x=='spam' else 0)
#create train/test split
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)
#find word counts and store in a matrix
cv=CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
x_train_count.toarray()
#train the model
model = MultinomialNB()
model.fit(x_train_count, y_train)
#test the model
#pretest ham
email = ['Hey Bach, can we get together to watch football game tomorrow?']
email_count = cv.transform(email)
print(model.predict(email_count))
#pretest spam
email = ['You have won a free vacation!']
email_count = cv.transform(email)
print(model.predict(email_count))
#test the model
x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))
