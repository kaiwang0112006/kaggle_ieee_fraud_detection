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
import argparse


def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]')
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--path', dest='path', help='path of dir', default='', required=True)
    parser.add_argument('--isvalid',dest='isvalid',help='is valid or not, 0 mean not valid data', default=0, type=int)
    args = parser.parse_args()

    return args

def generate_arrays(fin,batch_size=256):
    while 1:
        chk = fin.get_chunk(batch_size)
        feature = [f for f in chk.columns if f != 'TransactionID' and f != 'split' and f != 'isFraud']
        train_x = chk[feature]
        train_y = chk['isFraud']
        yield (train_x,train_y)

def main():
    options = getOptions()
    batch_size = 256
    num_epoch = 15

    model = model_from_json(joblib.load('modelv1.pkl'))
    model.load_weights('modelv1.h5')

    df = pd.read_csv(options.path,nrows=10)
    feature = [f for f in df.columns if f != 'TransactionID' and f != 'split' and f != 'isFraud']
    pred = model.predict(df[feature])
    print(pred)




if __name__ == '__main__':
    main()

