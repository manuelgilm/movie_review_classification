import os

def read_text_file(filepath):
    with open(filepath,"r") as f:
        file = f.read().strip()
    return file

def read_text_files(folderpath):
    files = os.listdir(folderpath)
    documents = []
    for file in files:
        document = read_text_file(os.path.join(folderpath, file))
        documents.append(document)
    return documents