import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
path = "/home/balazsa/Documents/Project-3.1/data"
data_list = os.listdir(path=path)
csvs = []
for csv_set in data_list:
    temp = os.listdir(path=f"{path}/{csv_set}")
    for entry in temp:
        csvs.append(f"{csv_set}/{entry}")
file = open("labels/combined_labels.json")
labels = json.load(file)
def isolation_forest(n_estimators, random_state):
    accuracies = []
    for csv in csvs:
        data = pd.read_csv(f'data/{csv}')
        scaler = StandardScaler().fit_transform(data.loc[:,data.columns!='timestamp'])
        scaled_data = scaler[0:len(data)]
        df = pd.DataFrame(data=scaled_data)
        x = data
        outlier_fraction = 0
        model = None
        anomalies=labels[csv]
        if len(anomalies) != 0:
            outlier_fraction=len(anomalies)/(len(data["value"])-len(anomalies))
            model = IsolationForest(n_estimators=n_estimators, contamination=outlier_fraction, random_state=random_state)
        else:
            model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        model.fit(df)
        scores_prediction = model.decision_function(df)
        y_pred = model.predict(df)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        y_feature = data["value"] 
        data['predicted_class'] = y_pred
        y = []
        if len(anomalies) != 0:
            for timestamp in data["timestamp"]:
                if timestamp in anomalies:
                    y.append(1)
                else:
                    y.append(0)
        else:
            for i in range(len(data["timestamp"])):
                y.append(0)
        accuracies.append(accuracy_score(y,y_pred))
    return (sum(accuracies)/len(accuracies))
    
def bruteforce_parameter_tuning(list_n_estimators, list_random_state):
    accuarcies = []
    for n_estimators in list_n_estimators:
        for random_state in list_random_state:
            result = isolation_forest(n_estimators=n_estimators,random_state=random_state)
            accuarcies.append(result)
    return max(accuarcies)

list_n_estimators = [50,100,150,200]
list_random_states = [25,50,75,100]
print(bruteforce_parameter_tuning(list_n_estimators=list_n_estimators,list_random_state=list_random_states))