# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd
from time import time

from component_store import *

class Pipeline:
    def __init__(self):
        self.fit_components = []
        self.pred_components = []
        self.data = {}

    def add_component(self, component, fit=True, pred=True):
        if fit:
            self.fit_components.append(component)
        if pred:
            self.pred_components.append(component)

    def fit(self, data_dict={}):
        self.data.update(data_dict)
        for component in self.fit_components:
            t0 = time()
            self.data = component.execute(self.data)
            print(f'Time spent: {time() - t0}')
        return self.data
    
    def predict(self, data_dict={}):
        self.data.update(data_dict)
        for component in self.pred_components:
            t0 = time()
            self.data = component.execute(self.data)
            print(f'Time spent: {time() - t0}')
        return self.data

class MadnessPipeline(Pipeline):
    def __init__(self, cols, model_key='rfc', param_dict={"random_state": 69}, rand=True, num_groups=5, brackets_per_group=3):
        super().__init__()

        # Define data ingestion component
        data_ingestion = DataIngestionComponent(cols=cols)
        self.add_component(data_ingestion, pred=False)

        # Define model training component
        # model = RandomForestClassifier(random_state=42)
        model_training = ModelTrainingComponent(model_key=model_key, param_dict=param_dict)
        self.add_component(model_training, pred=False)

        # Define model inference component
        model_inference = InferenceBracketComponent(cols=cols, rand=rand, num_groups=num_groups, brackets_per_group=brackets_per_group)
        self.add_component(model_inference, fit=False)

        # # Define genetic algorithm component
        # genetic_algorithm = GeneticAlgorithmComponent(population_size=population_size, generations=generations)
        # self.add_component(genetic_algorithm)

class ModelPipeline(Pipeline):
    def __init__(self, train_years, test_year, cols, model_key='rfc', param_dict={"random_state": 69}, rand=True, num_groups=5, brackets_per_group=3):
        super().__init__()

        # Define data ingestion component
        data_ingestion = DataIngestionComponent(cols=cols)
        self.add_component(data_ingestion)

        # Define model training component
        # model = RandomForestClassifier(random_state=42)
        model_training = ModelTrainingComponent(model_key=model_key, param_dict=param_dict)
        self.add_component(model_training, pred=False)

        # Define model inference component
        model_evaluation = ModelEvaluatorComponent(cols=cols)
        self.add_component(model_evaluation, fit=False)
