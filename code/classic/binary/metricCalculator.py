from sklearn.metrics import ( accuracy_score, precision_score, recall_score,f1_score, confusion_matrix)
from datetime import datetime

def metricCalculator(expected, predicted, average, zero_division, modelname, logfile):
   
    accuracy  = accuracy_score(expected, predicted)
    precision = precision_score(expected, predicted, average=average, zero_division=zero_division)
    recall    = recall_score(expected, predicted, average=average, zero_division=zero_division)
    f1        = f1_score(expected, predicted, average=average, zero_division=zero_division)

    cm = confusion_matrix(expected, predicted)
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"""
==================================================
Model: {modelname}
Time : {run_time}
==================================================

Confusion matrix [[TN FP] [FN TP]]:
{cm}

Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
F1-score : {f1:.4f}
TPR      : {tpr:.4f}
FPR      : {fpr:.4f}

"""

    with open(logfile, "a") as f:
        f.write(log_text)

    print(log_text)
