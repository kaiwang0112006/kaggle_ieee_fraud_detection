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

def calf(row,feature,model):
    input = np.array([[row[f] for f in feature]])
    #print(input[0])
    output = model.predict(input)
    #print(output[0])
    return mean_squared_error(input[0],output[0])


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
    step1 = ('onhot',OneHotClass(catego=categoricalDomain, miss='missing'))
    step2 = ('MinMaxScaler', minmaxScalerClass(cols=continuousDomain,target="TransactionID"))

    pipeline = Pipeline(steps=[step1,step2])
    transaction_new = pipeline.fit_transform(transaction)

    transfeature = [f for f in transaction_new.columns if f!='TransactionID' and f!='split']

    # build model
    encoding_dim=5
    input_data = Input(shape=(545,))
    batch_size = 256
    # encoder layers
    encoded = Dense(400, activation='relu')(input_data)
    encoded = Dense(300, activation='relu')(encoded)
    encoded = Dense(200, activation='relu')(encoded)
    encoded = Dense(100, activation='relu')(encoded)
    encoded = Dense(50, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim,name='extract')(encoded)

    # decoder layers
    decoded = Dense(50, activation='relu')(encoder_output)
    decoded = Dense(100, activation='relu')(decoded)
    decoded = Dense(200, activation='relu')(decoded)
    decoded = Dense(300, activation='relu')(decoded)
    decoded = Dense(400, activation='relu')(decoded)
    decoded = Dense(545, activation='relu')(decoded)
    # construct the autoencoder model
    autoencoder = Model(input=input_data, output=decoded)
    checkpoint = ModelCheckpoint('autoencoder_weights.{epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1,save_best_only=True, mode='min')
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(transaction_new[transfeature], transaction_new[transfeature], epochs=2, batch_size =batch_size,verbose=0,validation_split=0.2,callbacks=[checkpoint])
    autoencoder.save_weights('model.h5')
    json_string = autoencoder.to_json()
    joblib.dump(json_string, 'model.pkl')
    transaction_new['score'] = transaction_new.apply(lambda row:calf(row,transfeature,autoencoder),axis=1)
    transaction_new[['score','TransactionID','isFraud']].to_csv('submit_disv1.csv',index=False)


if __name__ == '__main__':
    main()

