from utils import *
import pandas as pd

class BracketEvaluator:
    def __init__(self, year):
        self.bracket_data = pd.read_csv(f'{DATA_DIR}\\data_cleaned.csv')
        self.bracket_data = self.bracket_data[self.bracket_data.YEAR == year]
    
    def evaluate(self, bracket):
        print(bracket)