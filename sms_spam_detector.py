# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:33:21 2023

@author: dhair
"""
#***************** Import Libraries ******************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score , precision_score


#******************* Importing Dataset here ***********************

sms_df = pd.read_csv("/Users/dhair/OneDrive/Desktop/spam.csv",encoding_errors = 'replace' )
 
print(sms_df)

print(sms_df.info())

# ******************** Droping out the columns ******************

sms_df.drop(columns = ['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4'] , inplace = True)

print(sms_df)

#************ REname the columns name **************
sms_df.rename(columns = {'v1': 'target' , 'v2' : 'text'} , inplace = True)
print(sms_df)

#*************  Convert the categorical Data into binary form *************

encoder = LabelEncoder()
sms_df['target'] = encoder.fit_transform(sms_df['target'])

print(sms_df['target'])

#********** Check the null in dataset ********

print(sms_df.isnull().sum())

#********** Check the Duplicates in data ***************

print(sms_df.duplicated().sum())

sms_df = sms_df.drop_duplicates(keep = 'first')
print(len(sms_df))

#********************* EDA ( Exploratry Data Analysis ) *********************

print(sms_df['target'].value_counts())

plt.pie(sms_df['target'].value_counts() , labels = ['ham' , 'spam'] , autopct = '%0.2f' )
plt.show()

#*******Check the num_characters . num_words , num_sentences ********

nltk.download('punkt')

sms_df['num_character'] = sms_df['text'].apply(len)
sms_df['num_words'] = sms_df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
sms_df['num_sentences'] = sms_df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

print(sms_df[['num_character' , 'num_words' , 'num_sentences']].describe())

#************ Plot the histogram **********

plt.figure(figsize = (12,6))
sns.histplot(sms_df[sms_df['target']==0]['num_character'])
sns.histplot(sms_df[sms_df['target']==1]['num_character'] , color = 'red')
#plt.show()
sns.histplot(sms_df[sms_df['target']==0]['num_words'])
sns.histplot(sms_df[sms_df['target']==1]['num_words'] , color = 'red')
#plt.show()
sns.histplot(sms_df[sms_df['target']==0]['num_sentences'])
sns.histplot(sms_df[sms_df['target']==1]['num_sentences'] , color = 'red')
#plt.show()

#***********Find the relationship between them with Pairplot ***************

sns.pairplot(sms_df , hue = 'target')
plt.show()

# ********** Find the Correlation between them ****************

sns.heatmap(sms_df.corr() , annot = True)
plt.show()

#********** Donig the DataPreprocessing ***************
#1.lowercase
#2.Tokenization
#3.Removing Special Characters
#4.Removing stopwords & Punctuation
#5. stemming


#************ create the function ******************
 
from nltk.corpus import stopwords

#top_words= set(stopwords.words('english'))
#nltkownload('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#rint(stopwords.words('english'))

import string
#tring.punctuation

def transform_text (text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
            
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text =y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
sms_df['transformed_text'] = sms_df['text'].apply(transform_text)

#*************** Generate the word cloud ****************

from wordcloud import WordCloud
wc = WordCloud(width = 500 , height = 500 , min_font_size = 10 , background_color = 'black')

spam_wc = wc.generate(sms_df[sms_df['target']==1]['transformed_text'].str.cat(sep = " "))

plt.figure(figsize = (12,6))
#lt.imshow(spam_wc)

ham_wc= wc.generate(sms_df[sms_df['target']==0]['transformed_text'].str.cat(sep = " "))

plt.figure(figsize = (12,6))
#lt.imshow(ham_wc)

     
#*********Top 30 words in ham or spam msg *********************
'''
spam_corpus = []
for msg in sms_df[sms_df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_commo(30))[0],pd.DataFrame(Counter((spam_corpus).most_common(30))[1]
plt.xticks(rotation = 'vertical')             
                                                                              
ham_corpus = []
for msg in sms_df[sms_df['target']==0]['transformed_text'].tolist():
    
    for word in msg.split():
        ham_corpus.append(word)
from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_commo(30))[0],pd.DataFrame(Counter((spam_corpus).most_common(30))[1] 
'''    
                                                                          
#******************** Model Building *********************


cv = CountVectorizer()
tfidf = TfidfVectorizer()
X =tfidf.fit_transform(sms_df['transformed_text']).toarray()
#rint(X.shape())

y = sms_df['target'].values

#************** Train Test Split ****************

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.2 , random_state= 2)

#************** Apply naive bayes *************

from sklearn.naive_bayes import GaussianNB,MultinomialNB , BernoulliNB

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

#************* Find the confusion matrix and classification report ************

#1- Using Gaussian NaiveBayes

gnb.fit(X_train , y_train)
y_pred = gnb.predict(X_test)
print('GaussianNB')
print(accuracy_score(y_test , y_pred))
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))

#2- Using Multinomial NaiveBayes

mnb.fit(X_train , y_train)
y_pred1= mnb.predict(X_test)
print('MultinomialNB')
print(accuracy_score(y_test , y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

#3- Using Bernouli NaiveBayes

bnb.fit(X_train , y_train)
y_pred2= bnb.predict(X_test)
print('BernoulliNB')
print(accuracy_score(y_test , y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


        
            
            
            









