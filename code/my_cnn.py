
#######################################################################################################################
############################################# change epoch and kfold as per need ##########################################
##########################################################################################################################
number_of_kfold=10                               ##########################################################################
number_of_epoch=200                             #########################################################################
verbose_value = 2                                       ############################################
#filename ='D:/project_6th_sem/code_shared/Amazon Datasets/AmazonAllData.csv'
filename ='AmazonAllData.csv'
embedding_vecor_length = 100
word2vec_model = 'gensim_model_amazon_data.txt'
#######################################################################################################################
######################################################################################################################




from keras.preprocessing.text import one_hot
#from keras.preprocessing.text import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.models import Model
from keras.layers import Flatten
import numpy
#from keras.utils import plot_model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import zeros
#from keras.utils.vis_utils import plot_model
from keras.callbacks import History
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import  f1_score
from sklearn.metrics import  r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.layers import Conv1D, MaxPooling1D
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy
import numpy as np
from numpy import asarray
from numpy import zeros

import pylab

import matplotlib.pyplot as plt
import pydot

######Loading CSV file 
import pandas as pd
df1=pd.read_csv(filename)
df=df1['Review_Text']
####################################################
#####################################################
#####Calling Tokenizer function 
tk=Tokenizer(filters="""'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'""",lower=True, split=" ")
#####Fitting tokenizer object 'tk' on our data column 'df'
tk.fit_on_texts(df)

##### Printing words and index pair 
index=tk.word_index
#print (index)

###Vocabulary size
vocab_size = len(index)
#vocab_size=50
print (vocab_size)

encoded_docs=[one_hot(d,vocab_size) for d in df]
#print (encoded_docs)
max_length=100
padded_docs = sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='pre')
#print (padded_docs)
labels = df1['Helpfulness']
#print(labels)



#VALIDATION_SPLIT=0.2
#indices = np.arange(padded_docs.shape[0])
#np.random.shuffle(indices)
#padded_docs = padded_docs[indices]
#labels = labels[indices]
#nb_validation_samples = int(VALIDATION_SPLIT * padded_docs.shape[0])

#x_train = padded_docs[:-nb_validation_samples]
#y_train = labels[:-nb_validation_samples]
#x_val   = padded_docs[-nb_validation_samples:]
#y_val   = labels[-nb_validation_samples:]
#print(x_train)

#indices = np.arange(padded_docs.shape[0])
#np.random.shuffle(indices)
#padded_docs = padded_docs[indices]
#labels = labels[indices]



###############Load pre-trained word vector####################
# load the whole embedding into memory
embeddings_index = dict()

f = open(word2vec_model)                  
##########    it is the gensim model to use word2vec.

for line in f:
    values = line.split()
    #######       for splitting every terms.
    #print( values)
    word = values[0]
    #######       taking 1st term. 
    #print(word," :: ",end='')
    coefs = asarray(values[1:], dtype='float64')
    
    #print(np.shape(coefs))
    #######     taking rest terms
    #print(coefs)
    embeddings_index[word] = coefs
    #######     mapping words to its vector    matrix that will be used.
f.close()
vector_length = str(np.shape(coefs))
vector_length = vector_length[1:]
vector_length = vector_length[:-2]
vector_length = int(vector_length)
#print(vector_length)
#print(isinstance(vector_length,int))
#print('Loaded %s word vectors.' % len(embeddings_index))
###    printing the number of words.
###    printing matrix generated.
#print(embeddings_index)
#####for i in embeddings_index:
#####    print(i," :: ",end='')
#####    print(embeddings_index[i])
#print(embeddings_index['product'])


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size+1, 100))
vector_zeros = np.zeros(vector_length, dtype = 'float64')
for word,i in tk.word_index.items():
	embedding_vector = embeddings_index.get(word,vector_zeros)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
		#print (embedding_matrix[i])
#X_train, X_test, Y_train, Y_test = train_test_split(padded_docs,labels, test_size=0.2)
##X_train, X_test, Y_train, Y_test = train_test_split(padded_docs,labels, test_size=0.33, random_state=7)
##random_state : int, RandomState instance or None, optional (default=None)
##
##If int, random_state is the seed used by the random number generator;
##If RandomState instance, random_state is the random number generator;
##If None, the random number generator is the RandomState instance used by np.random.
##
##shuffle : boolean, optional (default=True)
##
##Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.



#predictions_train=[]
#predictions_test=[]
#filter_sizes = [3,4,5]
history = History()
# HISTORY - to save and use the data for further computation. data refers to epoch, acc, loss, val_loss , val_acc.
fold_training=numpy.zeros(shape=number_of_epoch)
fold_test=numpy.zeros(shape=number_of_epoch)
#predictions_test=numpy.zeros(shape=int(len(X_test)/2))
#print(X_test)
#print(predictions_test)
#predictions_test=np_utils.to_categorical(predictions_test)
#print(predictions_test)
#kfold = KFold(n_splits=5, shuffle=True, random_state=7)
kfold = KFold(n_splits=number_of_kfold, shuffle=True, random_state=42)
# if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2,
#i.e. your split will be always the same. It doesn't matter what the actual random_state number is 42, 0, 21, ... 
#The important thing is that everytime you use 42, you will always get the same output the first time you make the split.
#This is useful if you want reproducible results, for example in the documentation, so that everybody can consistently 
#see the same numbers when they run the examples. In practice I would say, you should set the random_state to some fixed 
#number while you test stuff, but then remove it in production if you really need a random (and not a fixed) split.
print(kfold)

ith=1
accuracy =[]
loss=[]
val_accuracy=[]
val_loss = []
#accuracy_single_list=[]
#loss_single_list=[]
#val_accuracy_single_list=[]
#val_loss_single_list=[]

for train, test in kfold.split(padded_docs, labels):
    #######################################
    print ("Fold=",ith)
    ith=ith+1
    print ("TRAIN= ",train)
    print ("TEST= ",test)
    model = Sequential()
    model.add(Embedding(vocab_size+1, vector_length,weights=[embedding_matrix], input_length=embedding_vecor_length,trainable=False))
    # input length is the length of words in review
    model.add(Conv1D(nb_filter=50,filter_length=5,border_mode="valid",activation="relu",subsample_length=1))
    model.add(MaxPooling1D(pool_length=1))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    # compile the model
    #loss ke liye mse(mean squared error) aur metrics me bhi
    # summarize the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    print(model.summary())
    history1 = model.fit(padded_docs[train], labels[train],validation_data=(padded_docs[test], labels[test]),epochs=number_of_epoch,batch_size=100,verbose=verbose_value,callbacks=[history])
    prediction1 = model.predict(padded_docs[test])
    print(r2_score(labels[test],prediction1))
    a=history1.history['mean_squared_error']
    va=history1.history['val_mean_squared_error']
    l = history1.history['loss']
    vl=history1.history['val_loss']
    #for i in range(0,number_of_epoch):
    #    accuracy_single_list=accuracy_single_list.append(a[i])
    #    loss_single_list=loss_single_list.append(l[i])
    #    val_accuracy_single_list=val_accuracy_single_list.append(va[i])
    #    val_loss_single_list=val_loss_single_list.append(vl[i])
    accuracy.append(a)
    loss.append(l)
    val_accuracy.append(va)
    val_loss.append(vl)
    #plot_model(model, to_file='model_plot.png',show_shapes=True, show_layer_names=True)
    #predict =np.asarray( model.predict(X_test))
    #predictions_test=np.add( predict, predictions_test)
    #for i in range(0,len(prediction_test)):
    #    print(predict[i],predictions_test[i])
    print(history1.history)
    #print()
    #print(history1.history.keys())
    test_acc=np.asarray(history.history['val_mean_squared_error'])
    #print test_acc
    train_acc=np.asarray(history.history['mean_squared_error'])
    #print train_acc
    fold_training=np.add( train_acc, fold_training)
    #print fold_training
    fold_test=np.add( test_acc, fold_test)
    #print fold_test
    #print(history.history.keys())
    #print(isinstance(history1.history,dict))
    #print(history1.history['acc'])	
    #print(accuracy)
    ##print(loss)
    #print(val_loss)
    ##print(accuracy)
    ##print(val_accuracy)
fold_training=numpy.divide(fold_training,number_of_kfold)
fold_test=numpy.divide(fold_test,number_of_kfold)

print("MSE :: ")
print(accuracy)
print("Val MSE :: ")
print(val_accuracy)
print("Loss :: ")
print(loss)
print("Val Loss :: ")
print(val_loss)

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20,10),
         'axes.labelsize': '25',
         'axes.titlesize':'20',
          'xtick.labelsize':'25',
         'ytick.labelsize':'25'}
pylab.rcParams.update(params)
### summarize history for loss
plt.plot(fold_training)
plt.plot(fold_test)
#plt.title('Validation Loss')
plt.ylabel('Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['Val_train', 'Val_test'], loc='upper left')
plt.savefig("cnn_fig1.png")
