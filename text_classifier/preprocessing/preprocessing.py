import string
import re
import os
import pandas as pd 

from text_classifier.utils.read import read_text_files

def clean_text_data(text, stopwords):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("/W+", text)
    text = [token for token in tokens if token not in stopwords]
    return text

def get_dataset(folder_class, stopwords):
    classes = os.listdir(folder_class)
    dataframes = []
    for class_ in classes:
        documents = read_text_files(os.path.join(folder_class, class_))
        documents_ = [clean_text_data(text, stopwords) for text in documents]
        dataframes.append(pd.DataFrame({"raw_text":documents,
                                        "clean_text":documents_,
                                        "label":[class_ for _ in range(len(documents_))]}))
    df = pd.concat(dataframes, axis=0)
    return df
        


