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
    transaction
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
    step1 = ('onhot',OneHotClass(catego=categoricalDomain, miss='missing'))
    step2 = ('MinMaxScaler', minmaxScalerClass(cols=continuousDomain,target="TransactionID"))

    pipeline = Pipeline(steps=[step1,step2])
    transaction_new = pipeline.fit_transform(transaction)

    feature = [f for f in transaction_new.columns if f!='TransactionID' and f!='split' and f!='isFraud']

    transaction_new.to_csv('transaction_new.csv', index=False)

    data = transaction_new[transaction_new['split']==1]
    valid = transaction_new[transaction_new['split']==2]
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    train_x = train[feature]
    test_x = test[feature]
    train_y = train['isFraud']
    test_y = test['isFraud']
    valid_x = valid[feature]

    batch_size = 512
    num_epochs = 10
    clf = Sequential()
    clf.add(Dense(32, activation='relu', input_shape=(544,)))
    clf.add(Dense(32, activation='relu'))
    clf.add(Dense(1, activation='softmax'))
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    clf.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = clf.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs, callbacks=[checkpoint],
              validation_split=0.2, verbose=1)


    train_y_pred = np.array([i[0] for i in clf.predict(train_x)])
    train_ks = cal_ks_scipy(train_y_pred, train_y)
    y_pred = np.array([i[0] for i in clf.predict(test_x)])
    test_ks = cal_ks_scipy(y_pred, test_y)
    print(train_ks, test_ks)
    print(train_y_pred)
    tr_auc = metrics.roc_auc_score(train_y,train_y_pred)
    te_auc = metrics.roc_auc_score(test_y,y_pred)
    print(tr_auc,te_auc)

    valid['isFraud'] = [i[0] for i in clf.predict(valid_x)]
    valid[['TransactionID', 'isFraud']].to_csv('submit3.csv', index=False)




if __name__ == '__main__':
    main()


