import pandas as pd

def ingest_data(file_path):
    data = pd.read_csv(file_path)
    return data
