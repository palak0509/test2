# import libraties
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#--- HTML Tag Removal
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk



class Recommendation:
    
    def __init__(self):
        #nltk.data.path.append('./nltk_data/')
        #nltk.download('stopwords')
        #nltk.download('punkt')
        #nltk.download('averaged_perceptron_tagger')
        #nltk.download('wordnet')
        #self.data = pickle.load(open('data.pkl','rb'))
        self.user_final_rating = pickle.load(open('user_rating.pkl','rb'))
        self.model = pickle.load(open('logistic_model.pkl','rb'))
        self.data = pd.read_csv("sample30.csv")
        #self.raw_data = pd.read_csv("sample30.csv")
        #self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)
        
        
    def getTopProducts(self, user):
        tfs=pd.read_pickle('tfidf')
        mdl=pd.read_pickle('logistic_model.pkl')
        tfidffeatures = tfs.transform(self.data.reviews_data)
        classifiedsenti=self.model.predict(tfidfFeatures)
        #Merge the class to the dataframe
        sntmtClassSeries = pd.Series(classifiedsenti, name = "class_sent")
        self.data = self.data.join(sntmtClassSeries)
        #print(self.data[['manufacturer', 'name', 'reviews_text', 'class_sent']])
        groupedDf = self.data.groupby(['name'])
        product_class = groupedDf['class_sent'].agg(mean_class=np.mean)
        userrating=pd.read_pickle('user_rating.pkl.pkl')
        t20 = userrating.loc[user].sort_values(ascending=False)[0:20]
        for itmName in list(t20.index):
            t20[itmName] = product_class.loc[itmName][0]
        #t20.sort_values(ascending=False)[:5]
        return t20.sort_values(ascending=False)[:5]

    def getUsers(self):
        s= np.array(self.user_final_rating.index).tolist()
        #print(s)
        return ''.join(e+',' for e in s)