from keras.preprocessing.text import one_hot
#from keras.preprocessing.text import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Input,Merge,Dropout
from keras.models import Model
from keras.layers import Flatten
import numpy
from keras.utils import plot_model
from operator import add
from keras.layers.embeddings import Embedding
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
import pandas as pd
from keras.callbacks import History 
import numpy as np
from numpy import asarray
from numpy import zeros
from keras.utils.vis_utils import plot_model
from sklearn import metrics
import matplotlib.pyplot as plt
#import pydot
#import graphviz
from keras import backend as K
#K.set_image_data_format('th')
from keras.preprocessing import sequence
from keras.layers import Conv1D, MaxPooling1D
from laplotter import LossAccPlotter
#######Loading CSV file using PandasDropoutDropout

df1=pd.read_csv('snapdeal_data.csv')
#df1=pd.read_csv('money.csv')
df=df1['Review_Text']
#df=df1['Body']

#####Calling Tokenizer function 
tk=Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ")

#####Fitting tokenizer object 'tk' on our data column 'df'
tk.fit_on_texts(df)

##### Printing words and index pair 
index=tk.word_index
print index

#####convert each sentence in 'info' into sequence of its index
#x = tk.texts_to_sequences(df)
#print x


###Vocabulary size
vocab_size = len(index)
#vocab_size=50
print vocab_size

# encode full sentence into vector
#from sklearn import metrics
##encoded_docs = tk.texts_to_matrix(df, mode='count')
##print encoded_docs
##max_length=4

encoded_docs=[one_hot(d,vocab_size) for d in df]
print encoded_docs


#####Padding encoded sequence of words
max_length=50
padded_docs = sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='pre')
print padded_docs

####################Defining Output Label
#labels = [1,1,1,1,1,0,0,0,0,0]
#labels = df1['labels']
#labels = df1['B_Helpfulness']
labels = df1['Helpfulness']


############################## split the data into a training set and a validation set
##VALIDATION_SPLIT=0.2
##indices = np.arange(padded_docs.shape[0])
##np.random.shuffle(indices)
##padded_docs = padded_docs[indices]
##labels = labels[indices]
##nb_validation_samples = int(VALIDATION_SPLIT * padded_docs.shape[0])
##
##x_train = padded_docs[:-nb_validation_samples]
##y_train = labels[:-nb_validation_samples]
##x_val = padded_docs[-nb_validation_samples:]
##y_val = labels[-nb_validation_samples:]
############################################################################


##indices = np.arange(padded_docs.shape[0])
##np.random.shuffle(indices)
##padded_docs = padded_docs[indices]
##labels = labels[indices]


###############Load pre-trained word vector####################
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt') 
for line in f:
	values = line.split()
	#print values
	word = values[0]
	#print word
	coefs = asarray(values[1:], dtype='float32')
	#print coefs
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size+1, 100))
for word,i in tk.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
		print embedding_matrix[i]
################################################################

###########Definining Embedding Layer for neural network
convs = []
predictions_train=[]
predictions_test=[]
filter_sizes = [3,4,5]
history = History()
fold_training=numpy.zeros(shape=500)
fold_test=numpy.zeros(shape=500)
#empty_test=[0]*3
#

embedding_layer=Embedding(vocab_size+1, 100,weights=[embedding_matrix], input_length=max_length,trainable=False)
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# define 10-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []
ith=1
for train, test in kfold.split(padded_docs, labels):
        print ("Fold=",ith)
        ith=ith+1
        print (train,test)

        ##

        
        ##
        ############For different size filters
        for fsz in filter_sizes:
                x = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
                x = MaxPooling1D(5)(x)
                convs.append(x)
        x = Merge(mode='concat', concat_axis=1)(convs)
        x = Conv1D(128, 3, activation='relu')(x)
        ##############################################

        #x = Conv1D(128, 3, activation='relu')(embedded_sequences)
        x = MaxPooling1D(3)(x)
        #x = Conv1D(128, 3, activation='relu')(x)
        #x = MaxPooling1D(3)(x)
        ##x = Conv1D(128, 5, activation='relu')(x)
        ##x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        #x = Dense(128, activation='relu')(x)
        #preds = Dense(len(labels_index), activation='softmax')(x)
        #preds = Dense(1, activation='softmax')(x)

        x=Dense(60)(x)
        x=Dropout(0.2)(x)
        x=Dense(30)(x)
        preds=Dropout(0.2)(x)
        preds = Dense(1)(preds)
        #preds = Dense(1, activation='softmax')(x)
        model = Model(sequence_input, preds)

        # compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # summarize the modelcvscores.
        print(model.summary())

        # fit the model
        history=model.fit(padded_docs[train], labels[train],validation_data=(padded_docs[test], labels[test]), epochs=500, batch_size=50, verbose=2,callbacks=[history])
        #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        test_acc=np.asarray(history.history['val_loss'])
        #print test_acc
        train_acc=np.asarray(history.history['loss'])
        print train_acc
        fold_training=np.add( train_acc, fold_training)
        print fold_training
        fold_test=np.add( test_acc, fold_test)
        #print fold_test
        #print(history.history.keys())
        scores_train = model.evaluate(padded_docs[train], labels[train], verbose=2)
        scores = model.evaluate(padded_docs[test], labels[test], verbose=2)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print(history.history.keys())


        # calculate predictions
        predictions_test = model.predict(padded_docs[test])
        predictions_train = model.predict(padded_docs[train])

        
        # summarize history for accuracy
##        plt.plot(train_acc)
##        plt.plot(test_acc)
##        plt.title('Accuracy')
##        plt.ylabel('Accuracy')
##        plt.xlabel('epoch')
##        plt.legend(['train', 'test'], loc='upper left')
##        plt.show()
        ### summarize history for loss
##        plt.plot(history.history['loss'])
##        plt.plot(history.history['val_loss'])
##        plt.title('model loss')
##        plt.ylabel('loss')
##        plt.xlabel('epoch')
##        plt.legend(['train', 'test'], loc='upper left')
##        plt.show()
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
fold_training=numpy.divide(fold_training,10)
print fold_training
fold_test=numpy.divide(fold_test,10)
#print fold_test
#loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
#print('Accuracy: %f' % (accuracy*100))

# round predictions
#rounded = [round(x_val[0]) for x_val in predictions]
##print(rounded)


#print metrics.classification_report(y_val, rounded)
#print metrics.confusion_matrix(y_val, model.predict(x_val))

### summarize history for loss
plt.plot(fold_training)
plt.plot(fold_test)
#plt.title('Validation Loss')
plt.ylabel('Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['Val_train', 'Val_test'], loc='upper left')
plt.show()




################################ plot metrics

# list all data in history








##pyplot.plot.xlabel('Mean Squared Error')
##pyplot.plot.ylabel('Number of Iteration')
##pyplot.plot.text(history.history['mean_squared_error'])
##pyplot.show()

###################Vizualize the model weights

##get_1st_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
##layer_output = get_1st_layer_output([X])[0]
###weights=[]
##for layer in model.layers:
##    weights = layer.get_weights()


#################################Ouput of each layer#########################################
