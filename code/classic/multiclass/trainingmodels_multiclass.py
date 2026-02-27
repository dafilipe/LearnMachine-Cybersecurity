import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def trainingmodels_multiclass(
    model,
    modelname,
    traindata,
    trainlabel,
    testdata,
    testlabel,
    logfile,
    unlabeled_mode=False,
    attack2family=None,
    average="macro",
    zero_division=0
):
    model.fit(traindata, trainlabel)
    predicted = model.predict(testdata)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = (
        "\n" + "="*50 + "\n"
        f"Model: {modelname}\n"
        f"Time : {run_time}\n"
        + "="*50 + "\n"
    )

    # ---------- UNLABELED ----------
    if unlabeled_mode:
        total = len(predicted)

        attack_counts = Counter(predicted)

        text = header
        text += "Mode : UNLABELED TEST (no ground truth)\n\n"
        text += f"Total samples: {total}\n"
        text += "Top predicted attacks:\n"
        for atk, c in attack_counts.most_common(15):
            text += f"  {atk:15s} : {c} ({c/total:.2%})\n"

        if attack2family is not None:
            fam_pred = [attack2family.get(a, "unknown") for a in predicted]
            fam_counts = Counter(fam_pred)
            text += "\nTop predicted families:\n"
            for fam, c in fam_counts.most_common():
                text += f"  {fam:10s} : {c} ({c/total:.2%})\n"

        print(text)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        return predicted

    # ---------- LABELED ----------
    acc = accuracy_score(testlabel, predicted)
    prec = precision_score(testlabel, predicted, average=average, zero_division=zero_division)
    rec  = recall_score(testlabel, predicted, average=average, zero_division=zero_division)
    f1   = f1_score(testlabel, predicted, average=average, zero_division=zero_division)

    report = classification_report(testlabel, predicted, zero_division=zero_division)

    text = header
    text += "Mode : LABELED TEST\n\n"
    text += f"Accuracy : {acc:.4f}\n"
    text += f"Precision: {prec:.4f} ({average})\n"
    text += f"Recall   : {rec:.4f} ({average})\n"
    text += f"F1-score : {f1:.4f} ({average})\n\n"
    text += report + "\n"

    print(text)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    return predicted
