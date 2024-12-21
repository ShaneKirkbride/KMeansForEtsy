import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_csv(self):
        self.data = pd.read_csv(self.filepath)
        return self.data