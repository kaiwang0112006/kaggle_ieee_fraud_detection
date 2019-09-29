import pandas as pd
from project_demo.tools.features_engine import *
import copy
from sklearn.pipeline import *
from lightgbm.sklearn import LGBMClassifier
import pandas as pd
from project_demo.tools.multi_apply import *
from project_demo.tools.evaluate import *
from project_demo.tools.optimize import *
from project_demo.tools.wejoy import *
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn import metrics


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

    pipeline = Pipeline(steps=[step1])
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

    parms = {
        # 'x_train':X_train,
        # 'y_train':y_train,
        'num_leaves': (5, 40),
        'colsample_bytree': (0.1, 0.5),
        'drop_rate': (0.1, 1),
        'learning_rate': (0.001, 0.1),
        'max_bin': (10, 1000),
        'max_depth': (2, 5),
        'min_split_gain': (0.1, 0.9),
        'min_child_samples': (2, 10000),
        'n_estimators': (50, 2000),
        'reg_alpha': (0.1, 1000),
        'reg_lambda': (0.1, 1000),
        'sigmoid': (0.1, 1),
        'subsample': (0.1, 1),
        'subsample_for_bin': (100, 50000),
        'subsample_freq': (1, 10)
    }

    def roc_auc_score_fix(y_true, y_score):
        score = metrics.roc_auc_score(y_true, y_score)
        if score > 0.8:
            return 0
        else:
            return score

    # 参数整理格式，其实只需要提供parms里的参数即可
    intdeal = ['max_bin', 'max_depth', 'max_drop', 'min_child_samples',
               'min_child_weight', 'n_estimators', 'num_leaves', 'scale_pos_weight',
               'subsample_for_bin', 'subsample_freq']  # int类参数
    middledeal = ['colsample_bytree', 'drop_rate', 'learning_rate',
                  'min_split_gain', 'skip_drop', 'subsample', '']  # float， 只能在0，1之间
    maxdeal = ['reg_alpha', 'reg_lambda', 'sigmoid']  # float，且可以大于1

    others = {'is_unbalance': True, 'random_state': 24}

    bayesopsObj = bayes_ops(estimator=LGBMClassifier, param_grid=parms, cv=5, intdeal=intdeal, middledeal=middledeal,
                            maxdeal=maxdeal,
                            score_func=make_scorer(score_func=roc_auc_score_fix, greater_is_better=True,
                                                   needs_threshold=True),
                            init_points=3, n_iter=10, acq="ucb", kappa=0.1, others=others
                            )
    bayesopsObj.run(X=train_x, Y=train_y)
    parms = bayesopsObj.baseparms
    print(parms)

    clf = LGBMClassifier(**parms)
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
    valid[['TransactionID', 'isFraud']].to_csv('submitops.csv', index=False)




if __name__ == '__main__':
    main()
