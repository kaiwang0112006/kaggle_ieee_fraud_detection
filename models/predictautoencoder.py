import pandas as pd
from project_demo.tools.features_engine import *
import copy
from sklearn.pipeline import *
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import pandas as pd
from project_demo.tools.multi_apply import *
from project_demo.tools.evaluate import *
from sklearn.externals import joblib
from sklearn.metrics import *

def calf(row,feature,model):
    input = np.array([[row[f] for f in feature]])
    #print(input[0])
    output = model.predict(input)
    #print(output[0])
    return mean_squared_error(input[0],output[0])


def main():
    # input data
    train_identity = pd.read_csv('../data/train_identity.csv')
    test_identity = pd.read_csv('../data/test_identity.csv')
    train_identity['split'] = 1
    test_identity['split'] = 2
    identity = pd.concat([train_identity, test_identity])

    categoricalDomain=['id_12','id_15','id_16','id_23','id_27','id_28','id_29','id_30','id_31','id_33','id_34',
                       'id_35','id_36','id_37','id_38','DeviceType','DeviceInfo']
    continuousDomain = []
    for i in identity:
        if i not in categoricalDomain and i!='TransactionID' and i!='split':
            continuousDomain.append(i)

    identity = identity.fillna(-1)
    #step1 = ('label_encode', label_encoder_sk(cols=categoricalDomain))
    step1 = ('onhot',OneHotClass(catego=categoricalDomain, miss='missing'))
    step2 = ('MinMaxScaler', minmaxScalerClass(cols=continuousDomain,target="TransactionID"))

    pipeline = Pipeline(steps=[step1,step2])
    identity_new = pipeline.fit_transform(identity)

    transfeature = [f for f in identity_new.columns if f!='TransactionID' and f!='split']

    # build model
    encoding_dim=5
    input_data = Input(shape=(3588,))
    batch_size = 256
    # encoder layers
    encoded = Dense(2000, activation='relu')(input_data)
    encoded = Dense(1000, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(100, activation='relu')(encoded)
    encoded = Dense(50, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim,name='extract')(encoded)

    # decoder layers
    decoded = Dense(50, activation='relu')(encoder_output)
    decoded = Dense(100, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(1000, activation='relu')(decoded)
    decoded = Dense(2000, activation='relu')(decoded)
    decoded = Dense(3588, activation='relu')(decoded)
    # construct the autoencoder model
    #autoencoder = Model(input=input_data, output=decoded)
    #checkpoint = ModelCheckpoint('autoencoder_weights.{epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1,save_best_only=True, mode='min')
    #autoencoder.compile(optimizer='adam', loss='mse')
    #autoencoder.fit(identity_new[transfeature], identity_new[transfeature], epochs=2, batch_size =batch_size,verbose=0,validation_split=0.2,callbacks=[checkpoint])
    autoencoder = model_from_json(joblib.load('model.pkl'))
    autoencoder.load_weights('model.h5')
    score = identity_new.apply(lambda row:calf(row,transfeature,autoencoder),axis=1)
    pd.DataFrame({'TransactionID':list(identity_new['TransactionID']),'split':list(identity_new['split']),'autoscore':score}).to_csv('ae_result.csv',index=False)

if __name__ == '__main__':
    main()

