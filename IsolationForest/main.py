import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


def get_csvs_and_labels():
    """
    Get list of CSV files and their corresponding labels from the data directory and labels JSON file.
    """
    path = Path(__file__).parent / "data"
    data_list = os.listdir(path=path)
    csvs = []
    for csv_set in data_list:
        temp = os.listdir(path=f"{path}/{csv_set}")
        for entry in temp:
            csvs.append(f"{csv_set}/{entry}")
    file = open(Path(__file__).parent / "labels" / "combined_labels.json")
    labels = json.load(file)
    return labels, csvs

def get_csv_data(csv):
    """
    Load CSV data from the directory.
    """
    data = pd.read_csv(Path(__file__).parent / 'data' / csv)
    return data

def scale_data(data):
    """
    Scale the data using StandardScaler, excluding the timestamp column.
    """
    scaler = StandardScaler().fit_transform(data.loc[:,data.columns!='timestamp'])
    scaled_data = scaler[0:len(data)]
    df = pd.DataFrame(data=scaled_data)
    return df

def isolation_forest(n_estimators, random_state, return_detailed_metrics=False):
    """
    Apply Isolation Forest algorithm on the datasets and return performance metrics.
    """
    # Store metrics for all datasets
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    all_results = []

    labels, csvs = get_csvs_and_labels() # get CSV files and labels

    for csv in csvs: # we treat each CSV file as a separate dataset
        data = get_csv_data(csv)
        anomalies = labels[csv] # get anomaly timestamps for the current dataset
        df = scale_data(data) # scale the data
        
        outlier_fraction = 0 
        model = None
        
        if len(anomalies) != 0:
            outlier_fraction=len(anomalies)/(len(data)) # calculate outlier fraction (anomaly rate)
            contamination = min(0.1, max(0.001, outlier_fraction * 2)) # dynamic contamination with bounds - how much data is considered anomalous
            model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state) # initialize model
        else:
            model = IsolationForest(n_estimators=n_estimators, random_state=random_state) # initialize model without labeled anomalies

        model.fit(df) # fit the model to the data
        scores_prediction = model.decision_function(df) # anomaly scores
        y_pred = model.predict(df) # predicted labels (-1 for anomaly, 1 for normal)

        y_pred[y_pred == 1] = 0 # convert to 0 (normal)
        y_pred[y_pred == -1] = 1 # convert to 1 (anomaly)

        
        data['predicted_class'] = y_pred 
        data['anomaly_score'] = scores_prediction
        y = []

        if len(anomalies) != 0:
            for timestamp in data["timestamp"]:
                if timestamp in anomalies:
                    y.append(1) # actual anomaly
                else:
                    y.append(0) # actual normal
        else:
            for i in range(len(data["timestamp"])):
                y.append(0) # all normal if no labeled anomalies

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        confusion_m = confusion_matrix(y, y_pred) # confusion matrix 
        tn = confusion_m[0, 0] if confusion_m.shape[0] > 1 else 0 # true negatives
        fp = confusion_m[0, 1] if confusion_m.shape[0] > 1 else 0 # false positives
        fn = confusion_m[1, 0] if confusion_m.shape[0] > 1 and confusion_m.shape[1] > 1 else 0 # false negatives
        tp = confusion_m[1, 1] if confusion_m.shape[0] > 1 and confusion_m.shape[1] > 1 else 0 # true positives

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        auc = None
        if len(set(y)) > 1:
            auc = roc_auc_score(y_true=y, y_score=-scores_prediction) # AUC score

        if return_detailed_metrics:
            all_results.append({
                'csv': csv,
                'data': data,
                'y_true': y,
                'y_pred': y_pred,
                'scores': scores_prediction,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
    

    if return_detailed_metrics:
        valid_aucs = [result['auc'] for result in all_results if result['auc'] is not None]
        avg_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else None
     
        return {
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'avg_precision': sum(precisions) / len(precisions),
            'avg_recall': sum(recalls) / len(recalls),
            'avg_f1': sum(f1_scores) / len(f1_scores),
            'detailed_results': all_results,
            'avg_auc': avg_auc,
        }

    return (sum(accuracies)/len(accuracies))

def bruteforce_parameter_tuning(list_n_estimators, list_random_state, return_detailed_metrics=True):
    """
    Perform brute-force parameter tuning over n_estimators and random_state - it checks which combination yields the best performance.
    """
    detailed_metrics = []
    for n_estimators in list_n_estimators:
        for random_state in list_random_state:
            result = isolation_forest(n_estimators=n_estimators,random_state=random_state, return_detailed_metrics=return_detailed_metrics)
            if return_detailed_metrics:
                detailed_metrics.append(result)
    return detailed_metrics

def calculate_micro_average(detailed_results):
    """
    Calculate micro-average metrics across all datasets. Micro-average aggregates TP, FP, TN, FN across all datasets and calculates metrics from the aggregated confusion matrix.
    """
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for result in detailed_results:
        total_tp += result['tp']
        total_fp += result['fp']
        total_tn += result['tn']
        total_fn += result['fn']

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0

    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'micro_accuracy': micro_accuracy,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_tn': total_tn,
        'total_fn': total_fn
    }

def test_model(n_estimators, random_state, return_detailed_metrics=True):
    """
    Test the Isolation Forest model with given parameters.
    """
    detailed_results = bruteforce_parameter_tuning(list_n_estimators=n_estimators,list_random_state=random_state, return_detailed_metrics=return_detailed_metrics)
    if not return_detailed_metrics: 
        return
    
    best_result = detailed_results[0] # initialize best result
    
    print("Complete Detailed Results for every dataset with the best parameters")
    print("="*70)
    for result in best_result['detailed_results']:
        print(f"\nDataset: {result['csv']}")
        print(f"  Number of estimators: {n_estimators}, Random State: {random_state}")
        print(f"  True Positives (TP):  {result['tp']}")
        print(f"  False Positives (FP): {result['fp']}")
        print(f"  True Negatives (TN):  {result['tn']}")
        print(f"  False Negatives (FN): {result['fn']}")
        print(f"  Total Anomalies:      {result['tp'] + result['fn']}")
        print(f"  Total Normal:         {result['tn'] + result['fp']}")

    print("\n" + "="*70)
    print("Micro-Averaged Metrics Across All Datasets")
    print("="*70)
    micro_stats = calculate_micro_average(best_result['detailed_results'])
    print(f"\nAggregated Confusion Matrix:")
    print(f"  Total True Positives (TP):  {micro_stats['total_tp']}")
    print(f"  Total False Positives (FP): {micro_stats['total_fp']}")
    print(f"  Total True Negatives (TN):  {micro_stats['total_tn']}")
    print(f"  Total False Negatives (FN): {micro_stats['total_fn']}")
    print(f"\nMicro-Average Performance:")
    print(f"  Micro-Average Accuracy:  {micro_stats['micro_accuracy']:.4f}")
    print(f"  Micro-Average Precision: {micro_stats['micro_precision']:.4f}")
    print(f"  Micro-Average Recall:    {micro_stats['micro_recall']:.4f}")
    print(f"  Micro-Average F1-Score:  {micro_stats['micro_f1']:.4f}")

if __name__ == "__main__":
    list_n_estimators = [50,100,150,200] # number of trees in the forest
    list_random_states = [25,50,75,100] # different random states for reproducibility
    test_model(n_estimators=list_n_estimators, random_state=list_random_states, return_detailed_metrics=True)