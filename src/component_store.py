from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from utils import *
from evaluation import ModelEvaluator

class DataIngestionComponent:
    def __init__(self, cols=[1, 2, 3, 4, 5, 18, 19]):
        # self.years = years
        self.cols = cols

    def execute(self, passthrough_dict):
        print("TrainingDataIngestionComponent")
        years = passthrough_dict['years']
        print(years)

        data = get_training_data(years)

        X = column_selector(data, self.cols)
        y = data['result']
        print(X.shape, y.shape)
        
        passthrough_dict['x'] = X
        passthrough_dict['y'] = y
        return passthrough_dict

class ModelTrainingComponent:
    model_dict = {
        "rfc": RandomForestClassifier
    }
    def __init__(self, model_key, param_dict={}):
        self.model = ModelTrainingComponent.model_dict[model_key](**param_dict)

    def execute(self, passthrough_dict):
        print("ModelTrainingComponent")
        # print(input)
        X, y = passthrough_dict['x'], passthrough_dict['y']
        # Example: Training the model
        self.model.fit(X, y)
        passthrough_dict['model'] = self.model
        return passthrough_dict

class InferenceBracketComponent:
    def __init__(self, cols=[1, 2, 3, 4, 5, 18, 19], rand=True, num_groups=100, brackets_per_group=50):
        self.cols = cols
        self.rand = rand
        self.num_groups = num_groups
        self.brackets_per_group = brackets_per_group

    def execute(self, passthrough_dict):
        # Returns a numpy array of shape (num_groups, brackets_per_group, 63)

        model = passthrough_dict['model']
        year = passthrough_dict['years'][0]
        self.team_data = get_team_data(year)
        print(self.team_data.columns)
        if not 'SEED' in self.team_data.columns:
            df = pd.read_csv(f'{DATA_DIR}\\teams{year}.csv')
            self.team_data = pd.merge(df, self.team_data, 'left', on='TEAM')
            
        print("InferenceBracketComponent")
        # Time: 0.1 seconds per bracket
        groups = []
        for i in range(self.num_groups):
            brackets = []
            for j in range(self.brackets_per_group):
                r1_preds = predict_round(self.team_data, model, self.cols, prob=self.rand)
                r2_teams = get_winning_teams(self.team_data, r1_preds)

                r2_preds = predict_round(r2_teams, model, self.cols, prob=self.rand)
                r3_teams = get_winning_teams(r2_teams, r2_preds)
                
                r3_preds = predict_round(r3_teams, model, self.cols, prob=self.rand)
                r4_teams = get_winning_teams(r3_teams, r3_preds)
                
                r4_preds = predict_round(r4_teams, model, self.cols, prob=self.rand)
                r5_teams = get_winning_teams(r4_teams, r4_preds)
                
                r5_preds = predict_round(r5_teams, model, self.cols, prob=self.rand)
                r6_teams = get_winning_teams(r5_teams, r5_preds)
                
                r6_preds = predict_round(r6_teams, model, self.cols, prob=self.rand)
                r7_teams = get_winning_teams(r6_teams, r6_preds)
                
                brackets.append(np.concatenate([r1_preds, r2_preds, r3_preds, r4_preds, r5_preds, r6_preds], axis=0))
            groups.append(np.stack(brackets, axis=0))
        passthrough_dict['brackets'] = np.stack(groups, axis=0)
        print("+-="*100)
        print(passthrough_dict)
        return passthrough_dict

class InferenceGameComponent:
    def __init__(self) -> None:
        pass

    def execute(self, passthrough_dict):
        model = passthrough_dict['model']
        X = passthrough_dict['x']
        pred = model.predict(X)
        
        passthrough_dict['preds'] = pred
        return passthrough_dict

class ModelEvaluatorComponent:
    def __init__(self, cols):
        self.cols = cols

    def execute(self, passthrough_dict):
        years = passthrough_dict['years']
        eval = ModelEvaluator(years=years, cols=self.cols)
        model = passthrough_dict['model']
        acc = eval.evaluate(model=model)
        print(acc)
        passthrough_dict['acc'] = acc

        return passthrough_dict

class GeneticAlgorithmComponent:
    def __init__(self, population_size, generations):
        self.population_size = population_size
        self.generations = generations

    def execute(self, passthrough_dict, initial_population, X_test, y_test):
        model = passthrough_dict['model']
        population = initial_population

        for generation in range(self.generations):
            # Evaluate fitness of each individual in the population
            fitness_scores = self.evaluate_fitness(population, X_test, y_test)

            # Select top performers
            selected_indices = self.selection(fitness_scores)

            # Crossover to create new individuals
            new_population = self.crossover(population, selected_indices)

            # Mutate new individuals
            new_population = self.mutation(new_population)

            population = new_population

        # Return the final evolved population
        return population

    def evaluate_fitness(self, population, X_test, y_test):
        fitness_scores = []

        for individual in population:
            # Assuming individual is a set of parameters for the model
            model = RandomForestClassifier(random_state=42, **individual)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            fitness_scores.append(accuracy)

        return np.array(fitness_scores)

    def selection(self, fitness_scores):
        # Select top performers based on fitness scores
        selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
        return selected_indices

    def crossover(self, population, selected_indices):
        new_population = []

        for _ in range(self.population_size // 2):
            parent1 = population[np.random.choice(selected_indices)]
            parent2 = population[np.random.choice(selected_indices)]

            # Simple crossover: Take the average of parameters
            child = {}
            for key in parent1:
                child[key] = (parent1[key] + parent2[key]) / 2.0

            new_population.append(child)

        return population + new_population

    def mutation(self, population):
        # Randomly mutate some parameters
        for individual in population:
            for key in individual:
                if np.random.rand() < 0.1:  # 10% chance of mutation
                    individual[key] += np.random.normal(scale=0.1)  # Small random change

        return population
