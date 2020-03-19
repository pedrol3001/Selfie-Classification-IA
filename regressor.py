import sys

import pandas as pd
import numpy as np

import joblib
import pickle

from IPython import get_ipython

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_absolute_error,explained_variance_score

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():


    data = pd.read_csv('selfie_dataset.txt', 
                        sep=" ", 
                        header=None, 
                        names=["Nome","Rate", "partial_faces", "is_female", "baby", "child","teenager", "youth", "middle_age","senior", "white", "black","asian", "oval_face", "round_face",
                                "heart_face", "smiling", "mouth_open","frowning", "wearing_glasses", "wearing_sunglasses","wearing_lipstick","2tongue_out0", "duck_face","black_hair",
                                 "blond_hair", "brown_hair","red_hair", "curly_hair", "straight_hair","braid_hair", "showing_cellphone", "using_earphone","using_mirror", "wearing_hat"
                                 ,"braces","harsh_lighting","dim_lighting"])

                                  
    

    
    labels = np.array(data['Rate'])
    features1= data.drop("Rate", axis = 1)
    features= features1.drop("Nome", axis = 1)

    feature_list = list(features.columns)
    features = np.array(features)


    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1,random_state=0)

    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    test_features = sc.transform(test_features)

    print('The shape of our train_features is:', train_features.shape)
    print('The shape of our test_features is:', test_features.shape)


    isTrained = False
    min_importance = 0.04
    n_estimators = 200
    retrain = True

    if(isTrained):

        if(retrain):
            crf = joblib.load("regressor.pkl")

            rf = SelectFromModel(crf, threshold=min_importance)
            rf.fit(train_features, train_labels)

            train_features = rf.transform(train_features)
            test_features = rf.transform(test_features)

            print('The shape of our important_train_features is:', train_features.shape)
            print('The shape of our important_test_features is:', test_features.shape)

            rf_important = RandomForestRegressor(n_estimators=n_estimators,random_state=1)


            rf_important.fit(train_features, train_labels)

            rf = rf_important

            print(rf_important)
            print("\n\n")
            predictions = rf_important.predict(test_features)
            importances = list(rf_important.feature_importances_)
            
        else:
            rf = joblib.load("regressor.pkl")
            print(rf)
            print("\n\n")
            predictions = rf.predict(test_features)
            importances = list(rf.feature_importances_)
    
    else:

        rf = RandomForestRegressor(n_estimators = n_estimators,oob_score=True,random_state=2)
        rf.fit(train_features, train_labels)
        joblib.dump(rf, 'regressor.pkl') 

        print(rf)
        print("\n\n")
        predictions = rf.predict(test_features)
        importances = list(rf.feature_importances_)

    
    
    print('Mean Absolute Error:', mean_absolute_error(test_labels,predictions))
    mape = np.mean(np.abs((test_labels - predictions) / test_labels)) * 100
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%')

    print('Variance Score: ', explained_variance_score(test_labels,predictions))

    print("\n\n")

    print("Importances: ")
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    for pair in feature_importances:
        print('{} : {}'.format(*pair))


    print()

    

if __name__ == "__main__":
    main()