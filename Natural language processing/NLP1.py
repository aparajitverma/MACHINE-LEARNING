

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

    
    dataset= pd.read_csv('train1.csv')

dataset['tweet'][0]

processed_tweet=[]

for i in range(31962):
    tweet= re.sub('@[\w]*',' ',dataset['tweet'][0])
    tweet= re.sub('[^a-zA-Z#]',' ',tweet)
    tweet = tweet.lower()
    #change in list
    tweet= tweet.split()
    processed_tweet.append()
    
tweet= [ps.stem(token) for token in tweet if not token in stopwords.words('english')]

tweet= ''.join(tweet)

#textual encoding
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=3000) # sparse matrix
X=X.toarray()
y=dataset['label'].values


from sklearn.naive_bayes import GaussianNB
n_b=GaussianNB()
n_b.fit(X,y)
n_b.score(X,y)

#nltk.download('stopwords')






