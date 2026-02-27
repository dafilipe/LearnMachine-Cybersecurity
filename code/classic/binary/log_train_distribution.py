from datetime import datetime
import numpy as np

def log_train_distribution(y_train, logfile):

    y = np.asarray(y_train)
    total = len(y)
    attacks = int((y == 1).sum())
    normal  = int((y == 0).sum())

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_text = f"""
==================================================
RUN BASELINE — TRAIN DISTRIBUTION
Time : {run_time}
==================================================
Total samples     : {total}
True normal (0)   : {normal} ({normal/total:.2%})
True attack (1)   : {attacks} ({attacks/total:.2%})
==================================================

"""
    with open(logfile, "a") as f:
        f.write(log_text)

    print(log_text)
