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


def main():
    transaction = pd.read_csv('transaction_new.csv')
    dis = pd.read_csv('submit_disv1.csv')
    transaction_new = pd.merge(transaction,dis[['TransactionID','score']],on='TransactionID')
    feature = [f for f in transaction_new.columns if f!='TransactionID' and f!='split' and f!='isFraud']
    fmap = {}
    for f in feature:
        fmap[f] = f.replace(' ','_')
    transaction_new = transaction_new.rename(columns=fmap)
    data = transaction_new[transaction_new['split']==1]
    valid = transaction_new[transaction_new['split']==2]
    
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    train_x = train[list(fmap.values())]
    test_x = test[list(fmap.values())]
    train_y = train['isFraud'].astype('int')
    test_y = test['isFraud'].astype('int')

    clf = LGBMClassifier(boosting_type='gbdt',
            colsample_bytree=0.2, drop_rate=0.1,
            importance_type='split',
            learning_rate=0.04,
            max_bin=500,
            max_depth=4,
            min_child_samples=50,
            min_split_gain=0.1,
            n_estimators=500, n_jobs=-1,
            num_leaves=9, objective=None,
            random_state=24,
            reg_alpha=40, reg_lambda=10,
            sigmoid=0.4, silent=True,
            #class_weight={0:1,1:10},
            #subsample=0.3,
            subsample_for_bin=24000,
            is_unbalance=True,
            subsample_freq=1
            )
    clf.fit(train_x, train_y)
    train_y_pred = clf.predict_proba(train_x)[:,1]
    train_ks = cal_ks_scipy(train_y_pred, train_y)
    y_pred = clf.predict_proba(test_x)[:,1]
    test_ks = cal_ks_scipy(y_pred, test_y)
    print(train_ks, test_ks)
    tr_auc = metrics.roc_auc_score(train_y,train_y_pred)
    te_auc = metrics.roc_auc_score(test_y,y_pred)
    print(tr_auc,te_auc)

    valid['isFraud'] = clf.predict_proba(valid[clf._Booster.feature_name()])[:, 1]
    valid[['TransactionID', 'isFraud']].to_csv('submit6.csv', index=False)




if __name__ == '__main__':
    main()

