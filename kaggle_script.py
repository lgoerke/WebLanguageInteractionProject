## Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Activation, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pickle as pkl
import time
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# some definitions
MAXEPOCHS = 50
LSTM_1 = 10
EMBED = 20
EMBED_DROPOUT = 0.5
LSTM_DROPOUT = 0.5
LSTM_DROPOUT_REC = 0.5
LR_PATIENCE = 5
MIN_LR = .00001
LR_PLATEAUFACTOR = .2
ESC_PATIENCE = 10

# fix random seed for reproducibility
#np.random.seed(10)


# load data
#with open("data/X_train_h.pkl", "rb") as openfile:
#    X_train = pkl.load(openfile)[:,:,0]
#with open("data/y_train_h.pkl", "rb") as openfile:
#    y_train = pkl.load(openfile)[:,0]
#with open("data/X_val_h.pkl", "rb") as openfile:
#    X_val = pkl.load(openfile)[:,:,0]
#with open("data/y_val_h.pkl", "rb") as openfile:
#    y_val = pkl.load(openfile)[:,0]
#with open("data/X_test_h.pkl", "rb") as openfile:
#    X_test = pkl.load(openfile)[:,:,0]
#with open("data/y_test_h.pkl", "rb") as openfile:
#    y_test = pkl.load(openfile)[:,0]
#with open("data/char_to_int.pkl", "rb") as openfile:
#    char_to_int = pkl.load(openfile)
#with open("data/int_to_char.pkl", "rb") as openfile:
#    int_to_char = pkl.load(openfile)

# kaggle: import data
data = pd.read_csv('../../data/Combined_News_DJIA.csv')
train = data[data['Date'] < '2013-07-01'] # 2008 to mid 2013 (1300 samples) 65%
val = data[data['Date'] > '2013-06-31']   # mid 2013 to 2014 (350 samples)  17.5%
val = val[val['Date'] < '2015-01-02']
test = data[data['Date'] > '2014-12-31']  # 2014 to 2016 (350 samples)      17.5%


# kaggle: preprocess data
batch_size = 32
nb_classes = 2
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
if trainheadlines:
    print('trainheadlines exist!')
advancedvectorizer = TfidfVectorizer( min_df=0.04, max_df=0.3, max_features = 200000, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)

valheadlines = []
for row in range(0,len(val.index)):
    valheadlines.append(' '.join(str(x) for x in val.iloc[row,2:27]))
advancedval = advancedvectorizer.transform(valheadlines)
print(advancedtrain.shape)

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
print(advancedtest.shape)

X_train = advancedtrain.toarray()
X_val = advancedval.toarray()
X_test = advancedtest.toarray()


print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train["Label"])
y_val = np.array(val["Label"])
y_test = np.array(test["Label"])


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_val /= scale
X_test /= scale

mean = np.mean(X_train)
X_train -= mean
X_val -= mean
X_test -= mean




max_features = 10000
maxlen = 200
batch_size = 32
nb_classes = 2
# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(trainheadlines)
sequences_train = tokenizer.texts_to_sequences(trainheadlines)
sequences_test = tokenizer.texts_to_sequences(testheadlines)
sequences_val = tokenizer.texts_to_sequences(valheadlines)
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_val = sequence.pad_sequences(sequences_val, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)



def kaggle_network(lr):
    #EMBED_DROPOUT = LSTM_DROPOUT = LSTM_DROPOUT_REC = dropout
    t_0 = time.time()

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, EMBED))
    model.add(Dropout(EMBED_DROPOUT))
    model.add(LSTM(LSTM_1, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_DROPOUT_REC))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # configure optimizer, define callbacks...
    adam = optimizers.adam(lr=lr)
    #early_stop = EarlyStopping(monitor='val_loss', patience=ESC_PATIENCE, verbose=1)
    early_stop = EarlyStopping(monitor='loss', patience=ESC_PATIENCE, verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_PLATEAUFACTOR,
    #          patience=LR_PATIENCE, min_lr=MIN_LR)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=LR_PLATEAUFACTOR,
              patience=LR_PATIENCE, min_lr=MIN_LR)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=MAXEPOCHS,
              validation_data=(X_val, Y_val), callbacks=[early_stop, reduce_lr])
    score, acc = model.evaluate(X_test, Y_test,
                                batch_size=batch_size)
    print('Learning Rate:', lr)
    print('Test score:', score)
    print('Test accuracy:', acc)


    print("Generating test predictions...")
    preds15 = model.predict_classes(X_test, verbose=0)
    acc15 = accuracy_score(test['Label'], preds15)

    print('prediction accuracy: ', acc15)

        # predict
    predictions = model.predict(X_test)
    #print(predictions)

    return {
        'loss': 1-acc15,
        'status': STATUS_OK,
        # -- store other results like this
        'total_run_time': t_0 - time.time(),
        'parameters': {
                    'EMBED': EMBED,
                    'EMBED_DROPOUT': EMBED_DROPOUT,
                    'LSTM_1': LSTM_1,
                    'LSTM_DROPOUT': LSTM_DROPOUT,
                    'LSTM_DROPOUT_REC': LSTM_DROPOUT_REC,
                    'MAXEPOCHS': MAXEPOCHS,
                    'start_lr': lr,
                    'MIN_LR': MIN_LR,
                    'LR_PLATEAUFACTOR': LR_PLATEAUFACTOR,
                    'LR_PATIENCE': LR_PATIENCE,
                    'ESC_PATIENCE': ESC_PATIENCE,
                    'model_size': model.count_params()
                    },
        # -- attachments are handled differently
        'attachments':
            {
            'time_module': pkl.dumps(time.time),
            'model.predict': predictions,
            'model.predict_classes': preds15,
            }
        }


print(X_train.shape)
print(y_train.shape)

trials = Trials()
#best = fmin(fn=baseline_network,
best = fmin(fn=kaggle_network,
    #space=hp.uniform('x', 0.0001, 0.1),
    space=hp.loguniform('x', 0.1, 100),
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)
print(best)

# save all information about this optimization run to a pickle file
i = 1
while(os.path.isfile("trials-" + str(i) + ".pkl")):
    i = i+1
    print("trials-" + str(i) + ".pkl already exists.")
pkl.dump(trials, open("trials-" + str(i) + ".pkl", "wb"))
print("Saved trials object as pickle file: trials-"+str(i)+".pkl.")
print(trials.trials)
print(trials.results)

