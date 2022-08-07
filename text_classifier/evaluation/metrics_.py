from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

def get_performance_figures(estimator, x, y):
    pr_curve , ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(estimator, x, y, ax=ax)
    plt.title("Precision Recall Curve")
    
    roc_curve , ax = plt.subplots()
    RocCurveDisplay.from_estimator(estimator, x, y, ax=ax)
    plt.title("RoC Curve")

    return pr_curve, roc_curve

    

