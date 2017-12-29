"""Natural Language Processing

    Analysing Text !!!!!!!
    
    Here we are going to build a model that will predict that the review given 
    by the customer is good or bad in the dataset, this is a simple example but 
    we can use this model for other purpose as well!!!!



"""

import numpy as np
import matplotlib as mp
import pandas as pd

#quoting = 3 will ignore the double quotes
#importing the dataset as we are using tsv add the delimiter parameter
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

""" Cleaning the text, compulsory in NLP so that we can process further 
Removing "the" , "and" etc all the text we will remove which is not useful by 
machine learning model to predict the result """
import re

#Only keeping letter removing numbers symbols etc
#review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0]) #Only keep which you want to keep

#putting all the letters in lowercase 
#review = review.lower()

#removing words that are not relevant like removing "and" "the" ""this" like 
#in first example only loved is useful !!
import nltk
#importing a library that contains non-useful words like the,and etc
#download only once
nltk.download('stopwords')
from nltk.corpus import stopwords

#split into different words and converting into list
#review = review.split()

""" Stemming includes grouping of words like loved loving will be included as in love
only !!!
"""
#keeping the root only of the words 
from nltk.stem.porter import PorterStemmer
#ps=PorterStemmer()

#set is for faster execution
#
#As a sentence
#review= ' '.join(review)
lis=[]

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    lis.append(review)
    
"""Tokenisation : - It splits the reviews into different words like only relevant words """
#Creating BagOfWords model :- Clean the reviews for further simplification !!
"""
    Just to take all different words in the lis without duplicates, and creating one column
    for each word and rows will be the reviews !!! 
    
    We will get a table !!
    
    Rows = 1000 reviews
    Columns = number of different words 

    Matrix containing lot of zeros is called sparse matrix which we will be getting 
    here !!!
    
    And here we will try to reduce the sparcity ....
    
    WHY NEEDED ??? 
        To predict If the review is good or not, so for machine learning to do so 
        it is needed to be trained on all the reviews !!
        
        Here Classification is used but as in text so cleaning is needed because 
  independent and dependent variables are needed !!!!!
        
"""
#Creating Bag of Words Model By Tokenisation
from sklearn.feature_extraction.text import CountVectorizer
#Look at the parameters they are useful reducing the lines of code !!!!
#keeping only relevant 1500 words 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(lis).toarray()
Y = dataset.iloc[:,1].values

#Training the model ....
from sklearn.cross_validation import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size= 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , Y_pred)



