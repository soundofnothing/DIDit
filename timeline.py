import csv
from docarray import DocArray

class Doc:
    def __init__(self):
        self.id = None
        self.source = None
        self.target = None
        self.tweet_id = None
        self.type = None

def import_csv_to_docarray(csv_file_path):
    docarray = DocArray()
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            doc = Doc()
            doc.id = row['id']
            doc.source = row['source']
            doc.target = row['target']
            doc.tweet_id = row['tweet_id']
            doc.type = row['type']
            docarray.append(doc)
    return docarray
