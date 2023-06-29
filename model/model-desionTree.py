import re
# import nltk
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from collections import Counter
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import pymongo
from sklearn import model_selection
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_validate, KFold, HalvingGridSearchCV
import xgboost


def connect_mongo(query={},host='localhost', port=27017, username=None, password=None, db='test'):
    if username and password:
        mongo_uri = "mongodb://%s:%s@%s:%s/%s" % (username, password, host, port, db)
        client = pymongo.MongoClient(mongo_uri)
    else:
        client = pymongo.MongoClient(host, port)
    return client

def pre(X_test, y_test, model, model_name):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    # 计算指标
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    auc_num = auc(fpr, tpr)
    # 输出指标
    print('--------' + model_name + '----------')
    print("accuracy: " + str(acc))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1 score: " + str(f1))
    print("auc: " + str(auc_num))

    return fpr, tpr


def Model(X_train, y_train):
    # 过采样
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # 创造参数空间 - 使用与网格搜索时完全一致的空间，以便于对比
    param_grid_simple = {'max_depth': [*range(20, 50, 2)]
        , 'min_samples_leaf': [*range(20, 50, 2)]
        , 'criterion': ['gini', 'entropy']
                         }

    # 建立分类器、交叉验证
    # 决策树
    model = DecisionTreeClassifier(random_state=1
                                   )
    cv = KFold(n_splits=10, shuffle=True, random_state=1)

    # 定义对半搜索
    search = HalvingGridSearchCV(estimator=model
                                 , param_grid=param_grid_simple
                                 , factor=1.5
                                 , min_resources=500
                                 , scoring="precision"
                                 , verbose=True
                                 , random_state=1
                                 , cv=cv
                                 , n_jobs=-1)

    # 训练对半搜索评估器
    search.fit(X_train, y_train)

    # 查看最佳评估器
    return search.best_estimator_



client = connect_mongo()
collect = client.issues.train_data

df = pd.DataFrame(list(collect.find()))
res = df.drop('_id',axis = 1)

# 提取feature和label
dataset = np.array(df)
X = dataset[:, 1:-1]
y = dataset[:, -1]
# 数据标准化
standardScaler = StandardScaler()
standardScaler.fit(X)
X = standardScaler.transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

best_model = Model(X_train, y_train)

# 预测-决策树
fpr, tpr = pre(X_test, y_test, best_model, 'Desion Tree')

# 绘图，新建画布
fig, ax = plt.subplots()
ax.plot(fpr, tpr, '-s', linewidth=0.5, markersize=3)
ax.set_title("Desion Tree ROC")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")