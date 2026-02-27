import numpy as np
from datetime import datetime
from metricCalculator import metricCalculator

def trainingmodels(model,model_name,traindata,trainlabel, testdata, testlabel, average,  zero_division, modelname, logfile,unlabeled_mode=False, skip_metrics=False):

    model.fit(traindata, trainlabel)
    predicted = model.predict(testdata)

    if unlabeled_mode:
        total = len(predicted)
        attacks = int((predicted == 1).sum())
        normal  = int((predicted == 0).sum())

        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_text = (
            "\n" + "="*50 + "\n"
            f"Model: {modelname}\n"
            "Mode : UNLABELED TEST\n"
            f"Time : {run_time}\n"
            + "="*50 + "\n\n"
            f"Total samples     : {total}\n"
            f"Predicted normal  : {normal} ({normal/total:.2%})\n"
            f"Predicted attack  : {attacks} ({attacks/total:.2%})\n"
        )

        print(log_text)

        with open(logfile, "a", encoding="utf-8") as f:
            f.write(log_text + "\n")
            f.flush()

    else:
        if not skip_metrics:
            metricCalculator(expected=testlabel, predicted=predicted, average=average,zero_division=zero_division,modelname=modelname,  logfile=logfile)

    return predicted
