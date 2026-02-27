from metricCalculator import metricCalculator
def eval_global_and_difficulty(y_true, y_pred, diff, logfile, modelname):
    # Global
    metricCalculator(y_true, y_pred, "binary", 0, f"{modelname} (GLOBAL)", logfile)

    # Buckets of difficulty
    buckets = [
        ("Hard  1-7",   1, 7),
        ("Med   8-14",  8, 14),
        ("Easy  15-21", 15, 21),
    ]


    for name, a, b in buckets:
        mask = (diff >= a) & (diff <= b)
        n = int(mask.sum())
        if n == 0:
            continue
        metricCalculator(y_true[mask], y_pred[mask], "binary", 0, f"{modelname} ({name}) n={n}", logfile)

    
