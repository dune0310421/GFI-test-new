from sklearn import model_selection
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


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


def Model(dataset):
    # 输入是feature和相应label组成的二维np.array

    #     dataset = []
    #     ltc_data = [usedt1,title1,body1,fromcmt1,clscmt1]
    #     ltc_data = np.array(ltc_data).T.tolist()
    #     for i in range(LTC_issue_num):
    #         ltc_data[i].extend(label_num1[i])
    #         ltc_data[i].append(1)
    #         dataset.append(ltc_data[i])
    #     # print(ltc_data)
    #     otc_data = [usedt2,title2,body2,fromcmt2,clscmt2]
    #     otc_data = np.array(otc_data).T.tolist()
    #     for i in range(OTC_issue_num):
    #         otc_data[i].extend(label_num2[i])
    #         otc_data[i].append(0)
    #         dataset.append(otc_data[i])
    #     # print(ltc_data)

    #     print(len(dataset))
    #     dataset = np.array(dataset)
    #     # print(ltc_data)

    # 提取feature和label
    X = dataset[:, 1:-1]
    y = dataset[:, -1]
    # 数据标准化
    standardScaler = StandardScaler()
    standardScaler.fit(X)
    X = standardScaler.transform(X)

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, stratify=y)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # 构造svm
    Svm = svm.SVC(C=0.5, kernel='sigmoid', probability=True)
    Svm.fit(X_train, y_train)  # 训练模型
    # 预测
    fpr, tpr = pre(X_test, y_test, Svm, 'svm')
    # 绘图，新建画布
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, '-s', label='svm', linewidth=0.5, markersize=3)