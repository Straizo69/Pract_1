# LR_1_task_5.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

df = pd.read_csv('data_metrics.csv')

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype(int)
df['predicted_LR'] = (df.model_LR >= thresh).astype(int)

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def my_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def my_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def my_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def my_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def my_f1_score(y_true, y_pred):
    recall = my_recall_score(y_true, y_pred)
    precision = my_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


assert np.array_equal(my_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), "RF confusion_matrix mismatch"
assert np.array_equal(my_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), "LR confusion_matrix mismatch"


print("RF Metrics:")
print("Accuracy:", my_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("Recall:  ", my_recall_score(df.actual_label.values, df.predicted_RF.values))
print("Precision:", my_precision_score(df.actual_label.values, df.predicted_RF.values))
print("F1:      ", my_f1_score(df.actual_label.values, df.predicted_RF.values))
print()
print("LR Metrics:")
print("Accuracy:", my_accuracy_score(df.actual_label.values, df.predicted_LR.values))
print("Recall:  ", my_recall_score(df.actual_label.values, df.predicted_LR.values))
print("Precision:", my_precision_score(df.actual_label.values, df.predicted_LR.values))
print("F1:      ", my_f1_score(df.actual_label.values, df.predicted_LR.values))


fpr_RF, tpr_RF, _ = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, _ = roc_curve(df.actual_label.values, df.model_LR.values)
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
