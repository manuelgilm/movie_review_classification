from text_classifier.preprocessing.preprocessing import get_dataset
import nltk

if __name__=="__main__":
    data_path = "data"
    path_dest = "output/tfidf_features.csv"
    stopwords = nltk.corpus.stopwords.words("english")
    df = get_dataset(data_path, stopwords)
    df.to_csv(path_dest, index=False, header=True)
