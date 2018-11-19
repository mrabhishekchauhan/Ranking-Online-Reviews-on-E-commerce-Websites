from keras.preprocessing.text import one_hot
#from keras.preprocessing.text import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Flatten
import numpy
#from keras.utils import plot_model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import zeros
#from keras.utils.vis_utils import plot_model
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.layers import Conv1D, MaxPooling1D
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import gensim 
import os
from gensim import models
from gensim.models import Word2Vec


####################preprocessing#######################
stop = set(stopwords.words('english'))
stop=list(stop)
stop_capital=[]
for word in stop:
    stop_capital.append( word.capitalize())
file= open('CombinedDataReview.csv','r+',encoding="utf8")
list1=file.read()
list1=word_tokenize(list1)
stop_all = stop + stop_capital
filtered = []
for word in list1:
    if word not in stop_all:
        filtered.append(word)
stoplist =set("""M'lady M'lord ma'am Mr. Mrs. Smt. Dr. Er. e.g""".split() )
filtered= [word for word in  filtered if word not in stoplist]
filtered_alphanumeric = []
for word in filtered:
    if(word.isalnum()):
        filtered_alphanumeric.append(word)
final_list_of_words = []
final_list_of_words.append(filtered_alphanumeric)
print(final_list_of_words)


####################Gensim model########################


model = Word2Vec(final_list_of_words,min_count =1 , size = 100 , window=25)
print(model)
model.wv.save_word2vec_format("gensim_model_combined_data.txt", binary=False)
words = list(model.wv.vocab)
model.save("gensim_model_combined_data_model")
gensim_model_combined_data_model = gensim.models.Word2Vec.load("gensim_model_combined_data_model")
vocab = list(model.wv.vocab.keys())