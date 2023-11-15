# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd
from time import time

from component_store import *

class Pipeline:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def run(self):
        data = None
        for component in self.components:
            t0 = time()
            data = component.execute(data)
            print(f'Time spent: {time() - t0}')
        return data

class MadnessPipeline(Pipeline):
    def __init__(self, train_years, test_year, cols, model_key='rfc', param_dict={"random_state": 69}, rand=True, num_groups=5, brackets_per_group=3):
        super().__init__()

        # Define data ingestion component
        data_ingestion = TrainingDataIngestionComponent(years=train_years, cols=cols)
        self.add_component(data_ingestion)

        # Define model training component
        # model = RandomForestClassifier(random_state=42)
        model_training = ModelTrainingComponent(model_key=model_key, param_dict=param_dict)
        self.add_component(model_training)

        # Define model inference component
        model_inference = InferenceBracketComponent(year=test_year, cols=cols, rand=rand, num_groups=num_groups, brackets_per_group=brackets_per_group)
        self.add_component(model_inference)

        # # Define genetic algorithm component
        # genetic_algorithm = GeneticAlgorithmComponent(population_size=population_size, generations=generations)
        # self.add_component(genetic_algorithm)


