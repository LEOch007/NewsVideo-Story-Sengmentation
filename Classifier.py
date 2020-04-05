import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt

# constant parameters
gt_path = './data/gt_file_path'
feature_path = './data/output/'

def preproessing(df):
    # normalization
    from sklearn.preprocessing import StandardScaler
    df_transform = StandardScaler().fit_transform(
        df[[1,2,3,4]])
    df[[1,2,3,4]] = df_transform
    # oversampling

    return df

# get file name
def get_name(t):
    return 'hdcctv1_' + t[0].split('/')[-1]

# concat dataframes on the basis of 'bg' timestamp
def data_concat(gt_file_path_list,file_name_list):
    label_value = pd.read_csv(gt_file_path_list.iloc[0][0], header=None, skiprows=1, sep=' ',
                              names=['bg', 'label']).set_index('bg')
    feature_vector = pd.read_csv(feature_path + file_name_list.iloc[0], header=None).set_index(0)
    X = feature_vector.join(label_value)
    for i in range(1, file_name_list.shape[0]):
        label_value = pd.read_csv(gt_file_path_list.iloc[0][0], header=None, skiprows=1, sep=' ',
                                  names=['bg', 'label']).set_index('bg')
        feature_vector = pd.read_csv(feature_path + file_name_list.iloc[0], header=None).set_index(0)
        df = feature_vector.join(label_value)
        X = pd.concat([X, df], axis=0)
    X = X.dropna()  # ignore NaN vector
    X.index.names = ['bg']
    return X


# get training-set & validation-set
def get_train_vali(X,rate,seed):
    tt, vv = train_test_split(X, test_size=rate, random_state=seed)
    x_train, x_vali = tt.drop(columns='label'), vv.drop(columns='label')
    y_train,y_vali = tt['label'], vv['label']
    return x_train,y_train, x_vali,y_vali


# try machine learning methods
def try_different_model(model_, x_train,y_train, x_vali,y_vali, model_name):
    model_.fit(x_train, y_train)     # training process
    score = model_.score(x_vali, y_vali) # evaluation
    self_score = model_.score(x_train, y_train)  # self evaluation
    return [score, self_score]


if __name__ == '__main__':
    # get the file paths
    gt_file_path_list = pd.read_csv(gt_path,header=None)
    file_name_list = gt_file_path_list.apply(get_name,axis=1)

    # get the whole matrix
    data = data_concat(gt_file_path_list, file_name_list)
    # hold-out method: 30%
    x_train, y_train, x_vali, y_vali = get_train_vali(data,rate=0.3,seed=4)


    # random forest
    model_= ensemble.RandomForestClassifier(oob_score=True,random_state=10,n_estimators=20)
    # validation
    scores = try_different_model(model_,x_train, y_train, x_vali, y_vali,'RandomForest')

    #
    # # sklearn
    # from sklearn.linear_model import LogisticRegression
    # model_logistic = LogisticRegression()
    # logistic_scores = try_different_model(model_logistic,x_train, y_train, x_vali, y_vali,'LR')
    #
    # # SVM
    # from sklearn.svm import SVC
    # model_svm = SVC()
    # svm_scores = try_different_model(model_svm,x_train, y_train, x_vali, y_vali,'SVM')
    #
    # # xgb
    # from xgboost import XGBClassifier
    # model_xfb = XGBClassifier()
    # xgb_sccores = try_different_model(model_xfb,x_train, y_train, x_vali, y_vali,'XGB')
    #
    # # knn
    # from sklearn.neighbors import KNeighborsClassifier
    # model_knn = KNeighborsClassifier()
    # knn_scores = try_different_model(model_knn, x_train, y_train, x_vali, y_vali,'knn')