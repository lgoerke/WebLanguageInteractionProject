# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pickle as pkl
import time
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# some definitions
savepath="" # redundant since each experiment gets its own folder, but eh let's keep it
MAXEPOCHS = 50
LSTM_1 = 10
EMBED = 20
EMBED_DROPOUT = LSTM_DROPOUT = LSTM_DROPOUT_REC = 0.5
LR_PATIENCE = 5
MIN_LR = .00001
LR_PLATEAUFACTOR = .5
ESC_PATIENCE = 12

# fix random seed for reproducibility
np.random.seed(10)

# load data
with open("../../data/X_train_h_token.pkl", "rb") as openfile:
    X_train = pkl.load(openfile)[:,:,0]
with open("../../data/y_train_h_token.pkl", "rb") as openfile:
    y_train = pkl.load(openfile)[:,0]
with open("../../data/X_val_h_token.pkl", "rb") as openfile:
    X_val = pkl.load(openfile)[:,:,0]
with open("../../data/y_val_h_token.pkl", "rb") as openfile:
    y_val = pkl.load(openfile)[:,0]
with open("../../data/X_test_h_token.pkl", "rb") as openfile:
    X_test = pkl.load(openfile)[:,:,0]
with open("../../data/y_test_h_token.pkl", "rb") as openfile:
    y_test = pkl.load(openfile)[:,0]
with open("../../data/char_to_int.pkl", "rb") as openfile:
    char_to_int = pkl.load(openfile)
with open("../../data/int_to_char.pkl", "rb") as openfile:
    int_to_char = pkl.load(openfile)


def baseline_network(lr):

    print('Learning Rate: '+str(lr))

    max_features = 10000
    #input_dim = X_train.shape[1]

    t_0 = time.time()

    # create the model
    model = Sequential()
    model.add(Embedding(max_features,EMBED))
    model.add(Dropout(EMBED_DROPOUT))
    #model.add(LSTM(10,return_sequences=True))
    model.add(LSTM(LSTM_1, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_DROPOUT_REC))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dropout(DENSE_DROPOUT))

    # configure optimizer...
    adam = optimizers.adam(lr=lr)


    # define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=ESC_PATIENCE, verbose=1)
    #early_stop = EarlyStopping(monitor='loss', patience=ESC_PATIENCE, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_PLATEAUFACTOR,
              patience=LR_PATIENCE, min_lr=MIN_LR)
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=LR_PLATEAUFACTOR,
    #          patience=LR_PATIENCE, min_lr=MIN_LR)
    checkpointpath=savepath+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpointpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [early_stop, reduce_lr, checkpoint]


    # compile model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=MAXEPOCHS, batch_size=64,validation_data=(X_val, y_val),callbacks=callbacks_list)


    # final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("uebergebe: " + str(1-scores[1]))

    # predict
    predictions = model.predict(X_test)
    print(predictions)

    return {
        'loss': 1-scores[1],
        'status': STATUS_OK,
        # -- store other results like this
        'total_run_time': t_0 - time.time(),
        'parameters': {
                    'EMBED': EMBED,
                    'LSTM_1': LSTM_1,
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
            'predictions': predictions
            }
        }

trials = Trials()
best = fmin(fn=baseline_network,
    #space=hp.loguniform('x', 0.0001, 0.1),
    space=hp.loguniform('x', 0.0001, .5),
    algo=tpe.suggest,
    max_evals=20,
    trials=trials)
print(best)

# save all information about this optimization run to a pickle file
i = 1
while(os.path.isfile(savepath + "trials-" + str(i) + ".pkl")):
    i = i+1
    print(savepath + "trials-" + str(i) + ".pkl already exists.")
pkl.dump(trials, open(savepath + "trials-" + str(i) + ".pkl", "wb"))
print("Saved trials object as pickle file: "+savepath+"trials-"+str(i)+".pkl.")
print(trials.trials)
print(trials.results)
