from text_classifier.model_training.train_model import train
from text_classifier.evaluation.metrics_ import get_performance_figures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
import pandas as pd

  

if __name__=="__main__":
    data_path = "output/tfidf_features.csv"
    df = pd.read_csv(data_path)

    x_train, x_test, y_train, y_test = train_test_split(df["clean_text"], df["label"], test_size=0.25,random_state=1)

    trained_model = train(x_train, y_train)
    predictions = trained_model.predict(x_test)

    pr_curve, roc_curve = get_performance_figures(trained_model, x_test, y_test)
    pr_curve.savefig("output/Precision_Recall_Curve.png")
    roc_curve.savefig("output/ROC_curve.png")
    report = classification_report(predictions, y_test)
    print(report)
