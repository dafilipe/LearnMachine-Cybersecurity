import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D

from rawfile2Binary import rawfile2Binary
from log_train_distribution import log_train_distribution
from plot_utils import plot_training_history

np.random.seed(1337)
keras.utils.set_random_seed(1337)

RUN_DIR = os.path.join("results", datetime.now().strftime("run_%Y-%m-%d_%H%M"))
os.makedirs(RUN_DIR, exist_ok=True)

LOGFILE = os.path.join(RUN_DIR, "metrics.txt")
out_dir = RUN_DIR

print(f"--> Todos os resultados serão guardados em: {out_dir}")

unlabeled_mode = False
nsl_mode = False
diff_test = None

rootfolderKDD9 = "/home/diogo/disk/data/dataKDD99"
rootfolderNSLKDD = "/home/diogo/disk/data/dataNSL-KDD"

choice = int(input("Choose the dataset: [1] KDDCup99  [2] NSL-KDD :"))

if choice == 1:
    choicelabel = int(input("Choose if Test with Labeled or Unlabeled data: [1] Labeled  [2] Unlabeled :"))

    if choicelabel == 1:
        TRAIN_CSV, proto_map, serv_map, flag_map = rawfile2Binary(rootfolderKDD9, choice=1, choicelabel=1)
        train_df = pd.read_csv(TRAIN_CSV, header=None)

        X = train_df.iloc[:, :-1]
        y = train_df.iloc[:, -1]

        TEST_SIZE = float(input("Choose the test size (e.g. 0.2): "))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=42, shuffle=True
        )

        unlabeled_mode = False
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

        TEST_CSV, _, _, _ = rawfile2Binary(
            rootfolderKDD9, choice=1, choicelabel=2,
            protocol_map=proto_map, service_map=serv_map, flag_map=flag_map
        )
        test_df = pd.read_csv(TEST_CSV, header=None)

        X_test = test_df
        y_test = None
        unlabeled_mode = True

        print("Unlabeled test enabled (train=100% labeled).")
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test :", X_test.shape,  "(UNLABELED)")

else:
    nsl_mode = True
    TRAIN_CSV, proto_map, serv_map, flag_map = rawfile2Binary(rootfolderNSLKDD, choice=2, choicelabel=1)
    TEST_CSV, _, _, _ = rawfile2Binary(
        rootfolderNSLKDD, choice=2, choicelabel=2,
        protocol_map=proto_map, service_map=serv_map, flag_map=flag_map
    )

    train_df = pd.read_csv(TRAIN_CSV, header=None)
    test_df  = pd.read_csv(TEST_CSV, header=None)

    X_train = train_df.iloc[:, :-2]
    y_train = train_df.iloc[:, -2]
    X_test  = test_df.iloc[:, :-2]
    y_test  = test_df.iloc[:, -2]

    log_train_distribution(y_train, LOGFILE)

    unlabeled_mode = False
    print("NSL-KDD labeled test enabled.")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape,  "y_test :", y_test.shape)

scaler = Normalizer()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

trainX = np.array(X_train_norm, dtype=np.float32)
testX  = np.array(X_test_norm, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
if not unlabeled_mode:
    y_test = np.array(y_test, dtype=np.float32)

F = trainX.shape[1]
X_train_cnn = trainX.reshape((trainX.shape[0], F, 1))
X_test_cnn  = testX.reshape((testX.shape[0], F, 1))

def build_cnn(version: int, input_len: int):
    m = Sequential()
    if version == 1:
        m.add(Conv1D(64, 3, padding="same", activation="relu", input_shape=(input_len, 1)))
        m.add(MaxPooling1D(pool_size=2))
        m.add(Flatten())
        m.add(Dense(128, activation="relu"))
        m.add(Dropout(0.5))
        m.add(Dense(1, activation="sigmoid"))
    elif version == 2:
        m.add(Conv1D(64, 3, padding="same", activation="relu", input_shape=(input_len, 1)))
        m.add(Conv1D(64, 3, padding="same", activation="relu"))
        m.add(MaxPooling1D(pool_size=2))
        m.add(Flatten())
        m.add(Dense(128, activation="relu"))
        m.add(Dropout(0.5))
        m.add(Dense(1, activation="sigmoid"))
    elif version == 3:
        m.add(Conv1D(64, 3, padding="same", activation="relu", input_shape=(input_len, 1)))
        m.add(Conv1D(64, 3, padding="same", activation="relu"))
        m.add(MaxPooling1D(pool_size=2))
        m.add(Conv1D(128, 3, padding="same", activation="relu"))
        m.add(Conv1D(128, 3, padding="same", activation="relu"))
        m.add(MaxPooling1D(pool_size=2))
        m.add(Flatten())
        m.add(Dense(128, activation="relu"))
        m.add(Dropout(0.5))
        m.add(Dense(1, activation="sigmoid"))
    else:
        raise ValueError("version must be 1, 2, or 3")
    return m

v = int(input("Choose CNN version: [1] cnn1  [2] cnn2  [3] cnn3 : "))

model = build_cnn(v, F)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

epochs = int(input("Epochs (e.g. 50, 200, 1000): "))
batch_size = int(input("Batch size (e.g. 128): "))

config_log = f"""
==================================================
MODEL CONFIGURATION
==================================================
Dataset Choice : {choice}
CNN Version    : {v}
Epochs         : {epochs}
Batch Size     : {batch_size}
Input Shape    : {X_train_cnn.shape}
Output Dir     : {out_dir}
Unlabeled Mode : {unlabeled_mode}
==================================================
"""
with open(LOGFILE, "a") as f:
    f.write(config_log)

if unlabeled_mode:
    ckpt_monitor = "loss"
    ckpt_mode = "min"
    save_best_only = True
else:
    ckpt_monitor = "val_accuracy"
    ckpt_mode = "max"
    save_best_only = True

checkpointer = callbacks.ModelCheckpoint(
    filepath=os.path.join(out_dir, "checkpoint-{epoch:02d}.keras"),
    verbose=1,
    save_best_only=save_best_only,
    monitor=ckpt_monitor,
    mode=ckpt_mode
)

csv_logger = callbacks.CSVLogger(
    os.path.join(out_dir, f"cnntrainanalysis{v}.csv"),
    separator=",",
    append=False
)

print("\n[INFO] Starting training...")

fit_kwargs = dict(
    x=X_train_cnn,
    y=y_train,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpointer, csv_logger],
    verbose=1
)

if not unlabeled_mode:
    fit_kwargs["validation_data"] = (X_test_cnn, y_test)

history = model.fit(**fit_kwargs)

print("\n[INFO] Generating training plots...")
plot_training_history(history, out_dir)

print("\n[INFO] Loading best checkpoint for final stage...")

list_of_files = glob.glob(os.path.join(out_dir, "checkpoint-*.keras"))
if list_of_files:
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"--> Carregando: {os.path.basename(latest_file)}")
    model.load_weights(latest_file)
else:
    print("--> AVISO: Nenhum checkpoint encontrado. Usando o modelo final do treino.")
    latest_file = "Final Epoch Model"

print("[INFO] Predicting on test set...")
y_pred_prob = model.predict(X_test_cnn, batch_size=batch_size, verbose=1).reshape(-1)
y_pred = (y_pred_prob > 0.5).astype("int32")

if unlabeled_mode:
    pred_path = os.path.join(out_dir, "unlabeled_predictions.csv")
    pd.DataFrame({
        "y_pred_prob": y_pred_prob,
        "y_pred": y_pred
    }).to_csv(pred_path, index=False)

    final_log = f"""
==================================================
FINAL INFERENCE (UNLABELED TEST)
==================================================
Best Model Loaded: {latest_file}
Test Samples     : {X_test_cnn.shape[0]}
Saved Predictions: {pred_path}
==================================================
"""
    with open(LOGFILE, "a") as f:
        f.write(final_log)
    print(final_log)

else:
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Normal", "Attack"])

    final_log = f"""
==================================================
FINAL EVALUATION REPORT
==================================================
Best Model Loaded: {latest_file}

CONFUSION MATRIX:
{cm}
(Format: [TN, FP]
         [FN, TP])

CLASSIFICATION REPORT:
{cr}
==================================================
"""
    with open(LOGFILE, "a") as f:
        f.write(final_log)

    print(final_log)

    try:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks([0, 1], ["Normal", "Attack"])
        plt.yticks([0, 1], ["Normal", "Attack"])
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, str(val), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
        plt.close()
        print("[INFO] Confusion Matrix image saved.")
    except Exception as e:
        print(f"[WARN] Could not save Confusion Matrix image: {e}")

model.save(os.path.join(out_dir, "cnn_model_final.keras"))
print(f"\n[DONE] Processo concluído. Verifica a pasta: {out_dir}")
