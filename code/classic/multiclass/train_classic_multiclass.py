import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rawfile2Multiclass import rawfile2Multiclass
from attack_mapping import attack2family
from trainingmodels_multiclass import trainingmodels_multiclass
from trainingmodels_ovr import trainingmodels_ovr
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from latex_logger import LatexLogger
from log_multiclass_latex import log_multiclass_result_to_latex

protocol_map = {}
service_map  = {}
flag_map     = {}
label_map    = {}
unlabeled_mode = False
is_kdd99_official = False

rootfolderKDD9 = "/home/diogo/disk/data/dataKDD99"
rootfolderNSLKDD = "/home/diogo/disk/data/dataNSL-KDD"

print("\n==================================================")
print("Escolhe o modelo experimental:")
print("  [1] Modelo A — Multiclass (attack-type → stats por family)")
print("  [2] Modelo B — OvR (attack-type vs rest)")
print("==================================================")
mode = int(input("Opção: "))

choice = int(input("Choose the dataset: [1] KDDCup99  [2] NSL-KDD :"))
rootfolder = rootfolderKDD9 if choice == 1 else rootfolderNSLKDD

print(f"\nModo selecionado: {mode}")
timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

models = [
    (RandomForestClassifier(n_estimators=200, random_state=42), "Random Forest"),
    (LogisticRegression(max_iter=2000), "Logistic Regression"),
    (GaussianNB(), "Gaussian Naive Bayes"),
    (DecisionTreeClassifier(random_state=42), "Decision Tree"),
    (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "K-Nearest Neighbors"),
]

if mode == 1:
    split_type = None
    split_details = {}

    results_root = os.path.join("results", "modeloA_multiclass")
    RUN_DIR = os.path.join(results_root, timestamp)
    os.makedirs(RUN_DIR, exist_ok=True)

    if choice == 1:
        choicelabel = int(input("Choose if Test with Labeled or Unlabeled data: [1] Labeled  [2] Unlabeled :"))

        if choicelabel == 1:
            TRAIN_CSV, protocol_map, service_map, flag_map, label_map = rawfile2Multiclass(
                rootfolder=rootfolder,
                choice_dataset=1,
                choice_split=1
            )

            if TRAIN_CSV is None:
                print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV.")
                raise SystemExit(1)

            train_df = pd.read_csv(TRAIN_CSV, header=None)

            X = train_df.iloc[:, :-2]
            y = train_df.iloc[:, -2]
            family = train_df.iloc[:, -1]

            test_sizes = np.arange(0.1, 1.0, 0.1)

            for TEST_SIZE in test_sizes:
                run_tag = f"testsize_{TEST_SIZE:.1f}".replace(".", "_")
                RUN_DIR_TS = os.path.join(RUN_DIR, run_tag)
                os.makedirs(RUN_DIR_TS, exist_ok=True)
                LOGFILE = os.path.join(RUN_DIR_TS, "metrics.txt")
                TEXFILE = os.path.join(RUN_DIR_TS, "metrics.tex")

                tex = LatexLogger(TEXFILE, title="Resultados Multiclass", author="Diogo")
                tex.open()
                try:
                    tex.section("Configuração")
                    tex.kv_table([
                        ("Modo", str(mode)),
                        ("Dataset", "KDDCup99"),
                        ("RUN_DIR", RUN_DIR_TS),
                        ("Timestamp", timestamp),
                        ("TEST_SIZE", f"{TEST_SIZE:.1f}"),
                        ("random_state", "42"),
                        ("shuffle", "True"),
                        ("stratify", "attack"),
                    ])

                    X_train, X_test, y_train, y_test, fam_train, fam_test = train_test_split(
                        X, y, family,
                        test_size=TEST_SIZE,
                        stratify=y,
                        random_state=42,
                        shuffle=True
                    )

                    split_type = "Holdout split (train_test_split)"
                    split_details = {
                        "test_size": TEST_SIZE,
                        "stratify": "attack",
                        "random_state": 42,
                        "shuffle": True,
                    }

                    unlabeled_mode = False
                    is_kdd99_official = False
                    original_test_counts = None

                    traindata = X_train.to_numpy(dtype=np.float32)
                    testdata  = X_test.to_numpy(dtype=np.float32)
                    trainlabel = y_train.to_numpy()
                    testlabel  = y_test.to_numpy()
                    fam_test_arr = fam_test.to_numpy()

                    tex.section("Dataset & Split")
                    tex.kv_table(
                        [("Dataset", "KDDCup99"),
                         ("Split type", split_type)]
                        + [(k, v) for k, v in split_details.items()],
                        caption="Configuração do dataset e split"
                    )

                    for model, name in models:
                        predicted = trainingmodels_multiclass(
                            model=model,
                            modelname=f"{name} (Multiclass Attack)",
                            traindata=traindata,
                            trainlabel=trainlabel,
                            testdata=testdata,
                            testlabel=testlabel,
                            logfile=LOGFILE,
                            unlabeled_mode=False,
                            attack2family=attack2family
                        )

                        log_multiclass_result_to_latex(
                            tex=tex,
                            modelname=f"{name} (Multiclass Attack)",
                            y_true=testlabel,
                            y_pred=np.array(predicted),
                            fam_test_arr=fam_test_arr,
                            original_test_counts=original_test_counts,
                            is_kdd99_official=is_kdd99_official,
                        )

                        stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
                        for fam, yt, yp in zip(fam_test_arr, testlabel, predicted):
                            stats[fam]["total"] += 1
                            if yt == yp:
                                stats[fam]["correct"] += 1
                            else:
                                stats[fam]["wrong"] += 1

                        header = "\n--- Acertos/Erros por family---\n"
                        header += f"{'family':10s} {'total':>8s} {'correct':>10s} {'wrong':>8s} {'acc':>8s}\n"

                        print(header, end="")
                        with open(LOGFILE, "a", encoding="utf-8") as f:
                            f.write(header)
                            for fam, s in sorted(stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
                                acc = s["correct"] / s["total"] if s["total"] else 0.0
                                line = f"{fam:10s} {s['total']:8d} {s['correct']:10d} {s['wrong']:8d} {acc:8.2%}\n"
                                print(line, end="")
                                f.write(line)

                finally:
                    tex.close()
                    print("LaTeX escrito em:", TEXFILE)

        elif choicelabel == 2:
            unlabeled_mode = True
            is_kdd99_official = True

            RUN_DIR_TS = os.path.join(RUN_DIR, "unlabeled")
            os.makedirs(RUN_DIR_TS, exist_ok=True)
            LOGFILE = os.path.join(RUN_DIR_TS, "metrics.txt")
            TEXFILE = os.path.join(RUN_DIR_TS, "metrics.tex")

            tex = LatexLogger(TEXFILE, title="Resultados Multiclass", author="Diogo")
            tex.open()
            try:
                tex.section("Configuração")
                tex.kv_table([
                    ("Modo", str(mode)),
                    ("Dataset", "KDDCup99"),
                    ("RUN_DIR", RUN_DIR_TS),
                    ("Timestamp", timestamp),
                    ("Mode", "UNLABELED"),
                ])

                TRAIN_CSV, protocol_map, service_map, flag_map, label_map = rawfile2Multiclass(
                    rootfolder=rootfolder,
                    choice_dataset=1,
                    choice_split=1
                )
                if TRAIN_CSV is None:
                    print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV.")
                    raise SystemExit(1)

                train_df = pd.read_csv(TRAIN_CSV, header=None)
                train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

                X_train = train_df.iloc[:, :-2]
                y_train = train_df.iloc[:, -2]

                traindata = X_train.to_numpy(dtype=np.float32)
                trainlabel = y_train.to_numpy()

                TEST_CSV, _, _, _, _ = rawfile2Multiclass(
                    rootfolder=rootfolder,
                    choice_dataset=1,
                    choice_split=2,
                    protocol_map=protocol_map,
                    service_map=service_map,
                    flag_map=flag_map,
                    label_map=label_map
                )
                if TEST_CSV is None:
                    print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV de teste.")
                    raise SystemExit(1)

                test_df = pd.read_csv(TEST_CSV, header=None)
                test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

                X_test_raw = test_df.iloc[:, :-2]
                X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

                testdata = X_test.to_numpy(dtype=np.float32)
                testlabel = None
                original_test_counts = Counter({"unlabeled": len(test_df)})

                header = "\n--- TESTE UNLABELED: total de amostras ---\n"
                header += f"{'label':20s} {'count':>10s}\n"
                print(header, end="")
                with open(LOGFILE, "a", encoding="utf-8") as f:
                    f.write(header)
                    for atk, cnt in original_test_counts.most_common():
                        line = f"{atk:20s} {cnt:10d}\n"
                        print(line, end="")
                        f.write(line)

                tex.section("Dataset & Split")
                tex.kv_table(
                    [("Dataset", "KDDCup99"),
                     ("Split type", "Official unlabeled file"),
                     ("train", "kddcup99 corrected (labeled)"),
                     ("test", "kddcup99 unlabeled")],
                    caption="Configuração do dataset e split"
                )

                for model, name in models:
                    predicted = trainingmodels_multiclass(
                        model=model,
                        modelname=f"{name} (Multiclass Attack)",
                        traindata=traindata,
                        trainlabel=trainlabel,
                        testdata=testdata,
                        testlabel=None,
                        logfile=LOGFILE,
                        unlabeled_mode=True,
                        attack2family=attack2family
                    )

                    detected_counts = Counter(predicted)

                    header = "\n--- Distribuição DETETADA no TESTE por attack-type ---\n"
                    header += f"{'attack':20s} {'count':>10s}\n"
                    print(header, end="")
                    with open(LOGFILE, "a", encoding="utf-8") as f:
                        f.write(header)
                        for atk, cnt in detected_counts.most_common():
                            line = f"{atk:20s} {cnt:10d}\n"
                            print(line, end="")
                            f.write(line)

                    log_multiclass_result_to_latex(
                        tex=tex,
                        modelname=f"{name} (Multiclass Attack)",
                        y_true=None,
                        y_pred=np.array(predicted),
                        fam_test_arr=None,
                        original_test_counts=original_test_counts,
                        is_kdd99_official=is_kdd99_official,
                    )

            finally:
                tex.close()
                print("LaTeX escrito em:", TEXFILE)

        else:
            print("Opção inválida para labeled/unlabeled.")
            raise SystemExit(1)

    elif choice == 2:
        RUN_DIR_TS = os.path.join(RUN_DIR, "official_split")
        os.makedirs(RUN_DIR_TS, exist_ok=True)
        LOGFILE = os.path.join(RUN_DIR_TS, "metrics.txt")
        TEXFILE = os.path.join(RUN_DIR_TS, "metrics.tex")

        tex = LatexLogger(TEXFILE, title="Resultados Multiclass", author="Diogo")
        tex.open()
        try:
            tex.section("Configuração")
            tex.kv_table([
                ("Modo", str(mode)),
                ("Dataset", "NSL-KDD"),
                ("RUN_DIR", RUN_DIR_TS),
                ("Timestamp", timestamp),
            ])

            TRAIN_CSV, protocol_map, service_map, flag_map, label_map = rawfile2Multiclass(
                rootfolder=rootfolder,
                choice_dataset=2,
                choice_split=1
            )
            train_df = pd.read_csv(TRAIN_CSV, header=None)

            TEST_CSV, _, _, _, _ = rawfile2Multiclass(
                rootfolder=rootfolder,
                choice_dataset=2,
                choice_split=2,
                protocol_map=protocol_map,
                service_map=service_map,
                flag_map=flag_map,
                label_map=label_map
            )
            if TEST_CSV is None:
                print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV de teste (NSL).")
                raise SystemExit(1)

            test_df = pd.read_csv(TEST_CSV, header=None)

            X_train = train_df.iloc[:, :-3]
            y_train = train_df.iloc[:, -3]
            fam_train = train_df.iloc[:, -1]

            X_test = test_df.iloc[:, :-3]
            y_test = test_df.iloc[:, -3]
            fam_test = test_df.iloc[:, -1]

            unlabeled_mode = False
            is_kdd99_official = False
            original_test_counts = None

            scaler = Normalizer()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm  = scaler.transform(X_test)

            traindata = np.array(X_train_norm, dtype=np.float32)
            testdata  = np.array(X_test_norm,  dtype=np.float32)
            trainlabel = np.array(y_train)
            testlabel  = np.array(y_test)
            fam_test_arr = np.array(fam_test)

            split_type = "Official NSL-KDD Train/Test"
            split_details = {
                "train": "KDDTrain+",
                "test": "KDDTest+",
                "difficulty": "enabled",
            }

            tex.section("Dataset & Split")
            tex.kv_table(
                [("Dataset", "NSL-KDD"),
                 ("Split type", split_type)]
                + [(k, v) for k, v in split_details.items()],
                caption="Configuração do dataset e split"
            )

            for model, name in models:
                predicted = trainingmodels_multiclass(
                    model=model,
                    modelname=f"{name} (Multiclass Attack)",
                    traindata=traindata,
                    trainlabel=trainlabel,
                    testdata=testdata,
                    testlabel=testlabel,
                    logfile=LOGFILE,
                    unlabeled_mode=False,
                    attack2family=attack2family
                )

                log_multiclass_result_to_latex(
                    tex=tex,
                    modelname=f"{name} (Multiclass Attack)",
                    y_true=testlabel,
                    y_pred=np.array(predicted),
                    fam_test_arr=fam_test_arr,
                    original_test_counts=original_test_counts,
                    is_kdd99_official=is_kdd99_official,
                )

                stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
                for fam, yt, yp in zip(fam_test_arr, testlabel, predicted):
                    stats[fam]["total"] += 1
                    if yt == yp:
                        stats[fam]["correct"] += 1
                    else:
                        stats[fam]["wrong"] += 1

                header = "\n--- Acertos/Erros por family---\n"
                header += f"{'family':10s} {'total':>8s} {'correct':>10s} {'wrong':>8s} {'acc':>8s}\n"

                print(header, end="")
                with open(LOGFILE, "a", encoding="utf-8") as f:
                    f.write(header)
                    for fam, s in sorted(stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
                        acc = s["correct"] / s["total"] if s["total"] else 0.0
                        line = f"{fam:10s} {s['total']:8d} {s['correct']:10d} {s['wrong']:8d} {acc:8.2%}\n"
                        print(line, end="")
                        f.write(line)

        finally:
            tex.close()
            print("LaTeX escrito em:", TEXFILE)

    else:
        print("Dataset inválido.")
        raise SystemExit(1)

elif mode == 2:
    results_root = os.path.join("results", "modeloB_ovr")
    RUN_DIR = os.path.join(results_root, timestamp)
    os.makedirs(RUN_DIR, exist_ok=True)

    split_type = None
    split_details = {}
    original_test_counts = None
    is_kdd99_official = False
    unlabeled_mode = False

    if choice == 1:
        choicelabel = int(input("Choose if Test with Labeled or Unlabeled data: [1] Labeled  [2] Unlabeled :"))

        if choicelabel == 1:
            TRAIN_CSV, protocol_map, service_map, flag_map, label_map = rawfile2Multiclass(
                rootfolder=rootfolder,
                choice_dataset=1,
                choice_split=1
            )
            if TRAIN_CSV is None:
                print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV.")
                raise SystemExit(1)

            train_df = pd.read_csv(TRAIN_CSV, header=None)

            X = train_df.iloc[:, :-2]
            y = train_df.iloc[:, -2]

            test_sizes = np.arange(0.1, 1.0, 0.1)

            for TEST_SIZE in test_sizes:
                run_tag = f"testsize_{TEST_SIZE:.1f}".replace(".", "_")
                RUN_DIR_TS = os.path.join(RUN_DIR, run_tag)
                os.makedirs(RUN_DIR_TS, exist_ok=True)
                LOGFILE = os.path.join(RUN_DIR_TS, "metrics.txt")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=TEST_SIZE,
                    stratify=y,
                    random_state=42,
                    shuffle=True
                )

                traindata = X_train.to_numpy(dtype=np.float32)
                testdata  = X_test.to_numpy(dtype=np.float32)
                trainlabel = y_train.to_numpy()
                testlabel  = y_test.to_numpy()

                header = "\n" + "="*50 + "\n"
                header += "Modelo B — OvR\n"
                header += "Dataset: KDDCup99\n"
                header += f"TEST_SIZE: {TEST_SIZE:.1f}\n"
                header += "="*50 + "\n"
                print(header)
                with open(LOGFILE, "a", encoding="utf-8") as f:
                    f.write(header + "\n")

                counts = Counter(trainlabel)
                min_pos = 20
                attacks_to_run = [a for a, c in counts.items() if c >= min_pos and a != "normal"]
                attacks_to_run = sorted(attacks_to_run)

                header2 = "\n--- Attacks (OvR) selecionados ---\n"
                header2 += f"{'attack':20s} {'train_count':>12s}\n"
                print(header2, end="")
                with open(LOGFILE, "a", encoding="utf-8") as f:
                    f.write(header2)
                    for atk in attacks_to_run:
                        line = f"{atk:20s} {counts[atk]:12d}\n"
                        print(line, end="")
                        f.write(line)

                for model, name in models:
                    for atk in attacks_to_run:
                        trainingmodels_ovr(
                            model=model,
                            modelname=f"{name} (OvR)",
                            pos_label=atk,
                            traindata=traindata,
                            trainlabel=trainlabel,
                            testdata=testdata,
                            testlabel=testlabel,
                            logfile=LOGFILE,
                            unlabeled_mode=False,
                            attack2family=attack2family,
                        )

        elif choicelabel == 2:
            unlabeled_mode = True
            is_kdd99_official = True

            RUN_DIR_TS = os.path.join(RUN_DIR, "unlabeled")
            os.makedirs(RUN_DIR_TS, exist_ok=True)
            LOGFILE = os.path.join(RUN_DIR_TS, "metrics.txt")

            TRAIN_CSV, protocol_map, service_map, flag_map, label_map = rawfile2Multiclass(
                rootfolder=rootfolder,
                choice_dataset=1,
                choice_split=1
            )
            if TRAIN_CSV is None:
                print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV.")
                raise SystemExit(1)

            train_df = pd.read_csv(TRAIN_CSV, header=None)
            train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

            X_train = train_df.iloc[:, :-2]
            y_train = train_df.iloc[:, -2]

            traindata = X_train.to_numpy(dtype=np.float32)
            trainlabel = y_train.to_numpy()

            TEST_CSV, _, _, _, _ = rawfile2Multiclass(
                rootfolder=rootfolder,
                choice_dataset=1,
                choice_split=2,
                protocol_map=protocol_map,
                service_map=service_map,
                flag_map=flag_map,
                label_map=label_map
            )
            if TEST_CSV is None:
                print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV de teste.")
                raise SystemExit(1)

            test_df = pd.read_csv(TEST_CSV, header=None)
            test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

            X_test_raw = test_df.iloc[:, :-2]
            X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

            testdata = X_test.to_numpy(dtype=np.float32)
            testlabel = None

            header = "\n" + "="*50 + "\n"
            header += "Modelo B — OvR\n"
            header += "Dataset: KDDCup99\n"
            header += "Mode: UNLABELED\n"
            header += "="*50 + "\n"
            print(header)
            with open(LOGFILE, "a", encoding="utf-8") as f:
                f.write(header + "\n")

            counts = Counter(trainlabel)
            min_pos = 20
            attacks_to_run = [a for a, c in counts.items() if c >= min_pos and a != "normal"]
            attacks_to_run = sorted(attacks_to_run)

            header2 = "\n--- Attacks (OvR) selecionados ---\n"
            header2 += f"{'attack':20s} {'train_count':>12s}\n"
            print(header2, end="")
            with open(LOGFILE, "a", encoding="utf-8") as f:
                f.write(header2)
                for atk in attacks_to_run:
                    line = f"{atk:20s} {counts[atk]:12d}\n"
                    print(line, end="")
                    f.write(line)

            for model, name in models:
                for atk in attacks_to_run:
                    trainingmodels_ovr(
                        model=model,
                        modelname=f"{name} (OvR)",
                        pos_label=atk,
                        traindata=traindata,
                        trainlabel=trainlabel,
                        testdata=testdata,
                        testlabel=None,
                        logfile=LOGFILE,
                        unlabeled_mode=True,
                        attack2family=attack2family,
                    )

        else:
            print("Opção inválida para labeled/unlabeled.")
            raise SystemExit(1)

    elif choice == 2:
        RUN_DIR_TS = os.path.join(RUN_DIR, "official_split")
        os.makedirs(RUN_DIR_TS, exist_ok=True)
        LOGFILE = os.path.join(RUN_DIR_TS, "metrics.txt")

        TRAIN_CSV, protocol_map, service_map, flag_map, label_map = rawfile2Multiclass(
            rootfolder=rootfolder,
            choice_dataset=2,
            choice_split=1
        )
        train_df = pd.read_csv(TRAIN_CSV, header=None)

        TEST_CSV, _, _, _, _ = rawfile2Multiclass(
            rootfolder=rootfolder,
            choice_dataset=2,
            choice_split=2,
            protocol_map=protocol_map,
            service_map=service_map,
            flag_map=flag_map,
            label_map=label_map
        )
        if TEST_CSV is None:
            print("Erro: rawfile2Multiclass não conseguiu gerar/encontrar o CSV de teste (NSL).")
            raise SystemExit(1)

        test_df = pd.read_csv(TEST_CSV, header=None)

        X_train = train_df.iloc[:, :-3]
        y_train = train_df.iloc[:, -3]

        X_test = test_df.iloc[:, :-3]
        y_test = test_df.iloc[:, -3]

        scaler = Normalizer()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm  = scaler.transform(X_test)

        traindata = np.array(X_train_norm, dtype=np.float32)
        testdata  = np.array(X_test_norm,  dtype=np.float32)
        trainlabel = np.array(y_train)
        testlabel  = np.array(y_test)

        header = "\n" + "="*50 + "\n"
        header += "Modelo B — OvR\n"
        header += "Dataset: NSL-KDD\n"
        header += "Split: Official Train/Test\n"
        header += "="*50 + "\n"
        print(header)
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(header + "\n")

        counts = Counter(trainlabel)
        min_pos = 20
        attacks_to_run = [a for a, c in counts.items() if c >= min_pos and a != "normal"]
        attacks_to_run = sorted(attacks_to_run)

        header2 = "\n--- Attacks (OvR) selecionados ---\n"
        header2 += f"{'attack':20s} {'train_count':>12s}\n"
        print(header2, end="")
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(header2)
            for atk in attacks_to_run:
                line = f"{atk:20s} {counts[atk]:12d}\n"
                print(line, end="")
                f.write(line)

        for model, name in models:
            for atk in attacks_to_run:
                trainingmodels_ovr(
                    model=model,
                    modelname=f"{name} (OvR)",
                    pos_label=atk,
                    traindata=traindata,
                    trainlabel=trainlabel,
                    testdata=testdata,
                    testlabel=testlabel,
                    logfile=LOGFILE,
                    unlabeled_mode=False,
                    attack2family=attack2family,
                )

    else:
        print("Dataset inválido.")
        raise SystemExit(1)

else:
    print("Modo inválido.")
    raise SystemExit(1)
