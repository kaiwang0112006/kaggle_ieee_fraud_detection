import pandas as pd
from project_demo.tools.features_engine import *
import copy
from sklearn.pipeline import *
from lightgbm.sklearn import LGBMClassifier
import pandas as pd
from project_demo.tools.multi_apply import *
from project_demo.tools.evaluate import *
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

def generate_arrays(fin,batch_size=256):
    while 1:
        chk = fin.get_chunk(batch_size)
        feature = [f for f in chk.columns if f != 'TransactionID' and f != 'split' and f != 'isFraud']
        train_x = chk[feature]
        train_y = chk['isFraud']
        yield (train_x,train_y)

def main():

    batch_size = 1024
    num_epoch = 20
    df = pd.read_csv('train.csv', iterator=True)
    clf = Sequential()
    clf.add(Dense(32, activation='relu', input_shape=(393,)))
    clf.add(Dense(32, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    checkpoint = ModelCheckpoint('clf_weights.{epoch:03d}-{acc:.4f}.hdf5', monitor='acc', verbose=1,
                                 save_best_only=True, mode='auto')
    clf.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = clf.fit_generator(generate_arrays(df, batch_size), epochs=num_epoch, steps_per_epoch=10,
                                callbacks=[checkpoint])
    clf.save_weights('modelv1.h5')
    json_string = clf.to_json()
    joblib.dump(json_string, 'modelv1.pkl')



if __name__ == '__main__':
    main()

