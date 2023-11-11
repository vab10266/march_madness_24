from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from component_store import *

class Pipeline:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def run(self):
        data = None
        for component in self.components:
            data = component.execute(data)

class MadnessPipeline(Pipeline):
    def __init__(self, data_path, model_features, model_target, population_size, generations):
        super().__init__()

        # Define data ingestion component
        data_ingestion = TrainingDataIngestionComponent(data_path=data_path)
        self.add_component(data_ingestion)

        # Define model training component
        model = RandomForestClassifier(random_state=42)
        model_training = ModelTrainingComponent(model=model, features=model_features, target=model_target)
        self.add_component(model_training)

        # Define model inference component
        model_inference = ModelInferenceComponent()
        self.add_component(model_inference)

        # Define genetic algorithm component
        genetic_algorithm = GeneticAlgorithmComponent(population_size=population_size, generations=generations)
        self.add_component(genetic_algorithm)

if __name__ == "__main__":
    # Create an instance of MadnessPipeline
    madness_pipeline = MadnessPipeline(
        data_path="your_data.csv",
        model_features=["feature1", "feature2"],
        model_target="target",
        population_size=10,
        generations=5
    )

    # Run the MadnessPipeline
    madness_pipeline.run()

