import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def trainingmodels_ovr(
    model,
    modelname,
    pos_label,
    traindata,
    trainlabel,
    testdata,
    testlabel,
    logfile,
    unlabeled_mode=False,
    attack2family=None,
    zero_division=0
):
    y_train_bin = (np.asarray(trainlabel) == pos_label).astype(int)

    model.fit(traindata, y_train_bin)
    pred_bin = model.predict(testdata).astype(int)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = (
        "\n" + "="*50 + "\n"
        f"Model: {modelname}\n"
        f"Pos label: {pos_label}\n"
        f"Time : {run_time}\n"
        + "="*50 + "\n"
    )

    # ---------- UNLABELED ----------
    if unlabeled_mode or testlabel is None:
        total = len(pred_bin)
        pos = int(np.sum(pred_bin))
        neg = total - pos

        text = header
        text += "Mode : UNLABELED TEST (no ground truth)\n\n"
        text += f"Total samples: {total}\n"
        text += f"Predicted POS (== {pos_label}): {pos} ({pos/total:.2%})\n"
        text += f"Predicted NEG (rest):          {neg} ({neg/total:.2%})\n"

        if attack2family is not None:
            fam = attack2family.get(pos_label, "unknown")
            text += f"\nFamily (pos_label): {fam}\n"

        print(text)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        return pred_bin

    # ---------- LABELED ----------
    y_test_bin = (np.asarray(testlabel) == pos_label).astype(int)

    acc = accuracy_score(y_test_bin, pred_bin)
    prec = precision_score(y_test_bin, pred_bin, zero_division=zero_division)
    rec  = recall_score(y_test_bin, pred_bin, zero_division=zero_division)
    f1   = f1_score(y_test_bin, pred_bin, zero_division=zero_division)

    cm = confusion_matrix(y_test_bin, pred_bin, labels=[0, 1])
    report = classification_report(y_test_bin, pred_bin, zero_division=zero_division)

    text = header
    text += "Mode : LABELED TEST (OvR)\n\n"
    text += "Confusion matrix [[TN FP] [FN TP]]:\n"
    text += f"{cm}\n\n"
    text += f"Accuracy : {acc:.4f}\n"
    text += f"Precision: {prec:.4f}\n"
    text += f"Recall   : {rec:.4f}\n"
    text += f"F1-score : {f1:.4f}\n\n"
    text += report + "\n"

    print(text)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    return pred_bin
