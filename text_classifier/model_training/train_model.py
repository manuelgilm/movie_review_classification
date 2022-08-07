from multiprocessing.connection import Pipe
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def train(x_train, y_train):
    
    pipeline = Pipeline([
        ("tfidf",TfidfVectorizer()),
        ("classifier",RandomForestClassifier(random_state =1))
    ])

    pipeline.fit(x_train, y_train)

    return pipeline
