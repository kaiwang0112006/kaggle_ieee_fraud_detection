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


def main():
    train_transaction = pd.read_csv('../data/train_transaction.csv')
    test_transaction = pd.read_csv('../data/test_transaction.csv')
    test_transaction['split']=2
    train_transaction['split']=1
    transaction = pd.concat([train_transaction, test_transaction])

    aer = pd.read_csv('../data/ae_result.csv')
    transaction = pd.merge(transaction, aer[['TransactionID','autoscore']], on='TransactionID', how='left')

    categoricalDomain=['M1','M2','M3','M4','M5','M6','M7','M8','M9','P_emaildomain','ProductCD',
                       'R_emaildomain','card4','card6']
    continuousDomain = []
    for i in transaction:
        if i not in categoricalDomain and i!='TransactionID' and i!='split':
            continuousDomain.append(i)

    transaction = transaction.fillna(-1)
    #step1 = ('label_encode', label_encoder_sk(cols=categoricalDomain))
    step1 = ('label_encode', label_encoder_sk(cols=categoricalDomain))
    step2 = ('MinMaxScaler', minmaxScalerClass(cols=continuousDomain+categoricalDomain, target="TransactionID"))

    pipeline = Pipeline(steps=[step1,step2])
    transaction_new = pipeline.fit_transform(transaction)

    feature = [f for f in transaction_new.columns if f!='TransactionID' and f!='split' and f!='isFraud']

    transaction_new.to_csv('transaction_new.csv', index=False)

    data = transaction_new[transaction_new['split']==1]
    valid = transaction_new[transaction_new['split']==2]
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    valid.to_csv('valid.csv',index=False)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)





if __name__ == '__main__':
    main()

