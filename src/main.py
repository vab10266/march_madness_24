import os
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\data'))
print(DATA_DIR)

df = pd.read_csv(f'{DATA_DIR}\\training_data.csv')
print(df.head())