from rawfile2Binary import rawfile2Binary
import numpy as np
import os
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from trainingmodels import trainingmodels
from metricCalculator import metricCalculator
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys
from log_train_distribution import log_train_distribution
from eval_global_and_difficulty import eval_global_and_difficulty

RUN_DIR  = os.path.join("results", datetime.now().strftime("run_%Y-%m-%d_%H%M"))
os.makedirs(RUN_DIR, exist_ok=True)
LOGFILE = os.path.join(RUN_DIR, "metrics.txt")

unlabeled_mode = False
nsl_mode = False
diff_test = None

# Windows root
#rootfolderKDD9 = r"C:\Users\difil\Desktop\PIIC\data"
#rootfolderNSLKDD = r"C:\Users\difil\Desktop\PIIC\dataNSL-KDD"


#linux root 
rootfolderKDD9 = "/home/diogo/disk/data/dataKDD99"
rootfolderNSLKDD = "/home/diogo/disk/data/dataNSL-KDD"


choice = int(input("Choose the dataset: [1] KDDCup99  [2] NSL-KDD :"))

if choice == 1:

    choicelabel = int(input("Choose if Test with Labeled or Unlabeled data: [1] Labeled  [2] Unlabeled :"))

    if choicelabel == 1:
        # ---- LABELED: split para avaliar com métricas ----
        TRAIN_CSV, proto_map, serv_map, flag_map = rawfile2Binary(rootfolderKDD9, choice=1, choicelabel=1)
        train_df = pd.read_csv(TRAIN_CSV, header=None)

        X = train_df.iloc[:, :-1]
        y = train_df.iloc[:, -1]

        TEST_SIZE = float(input("Choose the test size (e.g. 0.2): "))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=42, shuffle=True
        )

        unlabeled_mode = False

        # baseline do treino no início do logfile
        log_train_distribution(y_train, LOGFILE)

        print("Labeled test enabled.")
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    else:
        TRAIN_CSV, proto_map, serv_map, flag_map = rawfile2Binary(rootfolderKDD9, choice=1, choicelabel=1)
        train_df = pd.read_csv(TRAIN_CSV, header=None)

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]

        log_train_distribution(y_train, LOGFILE)

        TEST_CSV, _, _, _ = rawfile2Binary(     rootfolderKDD9,      choice=1,     choicelabel=2,     protocol_map=proto_map,     service_map=serv_map,     flag_map=flag_map )
       
        test_df = pd.read_csv(TEST_CSV, header=None)

        X_test = test_df
        y_test = None
        unlabeled_mode = True

        print("Unlabeled test enabled (train=100% labeled).")
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test :", X_test.shape,  "(UNLABELED)")
else:
    # =========================
    # NSL-KDD
    # =========================

    nsl_mode = True

    TRAIN_CSV, proto_map, serv_map, flag_map = rawfile2Binary(rootfolderNSLKDD, choice=2, choicelabel=1)
    TEST_CSV, _, _, _ = rawfile2Binary(rootfolderNSLKDD, choice=2, choicelabel=2,protocol_map=proto_map, service_map=serv_map, flag_map=flag_map)

    train_df = pd.read_csv(TRAIN_CSV, header=None)
    test_df  = pd.read_csv(TEST_CSV, header=None)

    X_train = train_df.iloc[:, :-2]
    y_train = train_df.iloc[:, -2]
    diff_train = train_df.iloc[:, -1].astype(int)

    X_test  = test_df.iloc[:, :-2]
    y_test  = test_df.iloc[:, -2]
    diff_test = test_df.iloc[:, -1].astype(int)

    log_train_distribution(y_train, LOGFILE)

    print("NSL-KDD labeled test enabled.")
    print("X_train:", X_train.shape, "y_train:", y_train.shape, "diff_train:", diff_train.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape,  "diff_test :", diff_test.shape)


scaler = Normalizer()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

traindata   = np.array(X_train_norm)
trainlabel  = np.array(y_train)

testdata    = np.array(X_test_norm)
testlabel = None if unlabeled_mode else np.array(y_test)




# Logistic Regression
model = LogisticRegression()
skip = (not unlabeled_mode) and nsl_mode
pred = trainingmodels(model, "LR",traindata, trainlabel, testdata, testlabel, "binary", 0, "Logistic Regression", LOGFILE, unlabeled_mode=unlabeled_mode, skip_metrics=skip)

if skip:
    eval_global_and_difficulty( testlabel,pred,np.array(diff_test), LOGFILE, "Logistic Regression")

# Gaussian Naive Bayes 
model = GaussianNB()
pred = trainingmodels(model, "GNB", traindata, trainlabel, testdata, testlabel, "binary", 0, "Gaussian Naive Bayes", LOGFILE, unlabeled_mode=unlabeled_mode, skip_metrics=skip)
if skip:
    eval_global_and_difficulty(testlabel,pred, np.array(diff_test), LOGFILE, "Gaussian Naive Bayes")

# K-Nearest Neighbors
model = KNeighborsClassifier()  
pred = trainingmodels(model, "KNN", traindata, trainlabel, testdata, testlabel, "binary", 0, "K-Nearest Neighbors", LOGFILE, unlabeled_mode=unlabeled_mode, skip_metrics=skip)
if skip:
    eval_global_and_difficulty(testlabel, pred, np.array(diff_test), LOGFILE, "K-Nearest Neighbors")

#decision tree
model = DecisionTreeClassifier()
pred = trainingmodels(model, "DT", traindata, trainlabel, testdata, testlabel, "binary", 0, "Decision Tree", LOGFILE, unlabeled_mode=unlabeled_mode, skip_metrics=skip)
if skip:
    eval_global_and_difficulty(testlabel, pred, np.array(diff_test), LOGFILE, "Decision Tree")

#AdaBoost
model = AdaBoostClassifier(n_estimators=100)
pred = trainingmodels(model, "AB", traindata, trainlabel, testdata, testlabel, "binary", 0, "AdaBoost", LOGFILE, unlabeled_mode=unlabeled_mode, skip_metrics=skip)
if skip:
    eval_global_and_difficulty(testlabel, pred, np.array(diff_test), LOGFILE, "AdaBoost")

#Random Forest
model = RandomForestClassifier(n_estimators=100)    
pred = trainingmodels(model, "RF", traindata, trainlabel, testdata, testlabel, "binary", 0, "Random Forest", LOGFILE, unlabeled_mode=unlabeled_mode, skip_metrics=skip)
if skip:
    eval_global_and_difficulty(testlabel, pred, np.array(diff_test), LOGFILE, "Random Forest")

# SVM with GridSearchCV
tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]},
    {"kernel": ["linear"], "C": [1, 10, 100]}
]
scores = ["precision", "recall"]

for score in scores:

    print(f"\n# Tuning hyper-parameters for {score}\n")

    clf = GridSearchCV(
        SVC(),
        tuned_parameters,
        cv=3,
        scoring=f"{score}_macro",
        n_jobs=-1,
        verbose=1
    )

    clf.fit(traindata, trainlabel)
    print("Best params:", clf.best_params_)

    predicted = clf.predict(testdata)

    if unlabeled_mode:
        total = len(predicted)
        attacks = int((predicted == 1).sum())
        normal  = int((predicted == 0).sum())

        msg = (
            "\n" + "="*55 + "\n"
            "Model: SVM (GridSearchCV)\n"
            f"Score: {score}\n"
            "Mode : UNLABELED TEST\n"
            f"Total samples    : {total}\n"
            f"Predicted normal : {normal} ({normal/total:.2%})\n"
            f"Predicted attack : {attacks} ({attacks/total:.2%})\n"
            + "="*55 + "\n"
        )

        print(msg, end="")
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(msg)
            f.flush()

    else:
        if nsl_mode:
            eval_global_and_difficulty(
                testlabel,
                predicted,
                np.array(diff_test),
                LOGFILE,
                f"SVM GridSearchCV [{score}]"
            )
        else:
            metricCalculator(
                testlabel,
                predicted,
                "binary",
                0,
                f"SVM with GridSearchCV [{score}]",
                LOGFILE
            )

